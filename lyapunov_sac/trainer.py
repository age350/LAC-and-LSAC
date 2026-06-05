"""Train LSAC entirely by episodes and record every episode."""

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import agent
import config
import envs
import plotting
import replay_buffer


METRIC_FIELDS = [
    "q1_loss",
    "q2_loss",
    "lyapunov_loss",
    "sac_actor_loss",
    "actor_loss",
    "alpha_loss",
    "labda_loss",
    "l_delta",
    "alpha",
    "labda",
    "entropy",
    "q_value",
    "lyapunov_value",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_metrics(metrics):
    return {
        key: float(np.mean(values))
        for key, values in metrics.items()
        if values
    }


def moving_average(values, window):
    return float(np.mean(values[-min(window, len(values)) :]))


def evaluate(lsac_agent, env, episodes):
    rewards = []
    costs = []
    for test_episode in range(1, episodes + 1):
        obs, _ = env.reset(seed=10_000 + test_episode)
        terminated = truncated = False
        total_reward = 0.0
        total_cost = 0.0
        while not (terminated or truncated):
            action = lsac_agent.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_cost += info["cost"]
        rewards.append(total_reward)
        costs.append(total_cost)
    return float(np.mean(rewards)), float(np.mean(costs))


def resolve_run_directory(output_dir):
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = Path(__file__).resolve().parent / output_root
    run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train(train_config):
    train_config.validate()
    set_seed(train_config.seed)

    env = envs.make_env(train_config.env_id, train_config.seed)
    test_env = envs.make_env(train_config.env_id, train_config.seed + 1)
    lsac_agent = agent.LSACAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_sizes=train_config.hidden_sizes,
        gamma=train_config.gamma,
        polyak=train_config.polyak,
        alpha=train_config.alpha,
        alpha3=train_config.alpha3,
        labda=train_config.labda,
        actor_lr=train_config.actor_lr,
        q_lr=train_config.q_lr,
        lyapunov_lr=train_config.lyapunov_lr,
        alpha_lr=train_config.alpha_lr,
        labda_lr=train_config.labda_lr,
        device=train_config.device,
    )
    memory = replay_buffer.ReplayBuffer(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        capacity=train_config.replay_size,
        device=lsac_agent.device,
    )

    run_dir = resolve_run_directory(train_config.output_dir)
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(train_config), file, indent=2)

    episodes_path = run_dir / "episodes.csv"
    reward_curve_path = run_dir / "reward_curve.png"
    model_path = run_dir / "lsac_model.pt"
    episode_fields = [
        "episode",
        "total_steps",
        "episode_reward",
        "episode_cost",
        "episode_length",
        "moving_average_reward",
        "test_reward",
        "test_cost",
        *METRIC_FIELDS,
    ]

    total_steps = 0
    all_episode_rewards = []

    print("LSAC network: SAC Actor + Q1 + Q2 + Lyapunov Critic")
    print(f"Training episodes: {train_config.episodes}")
    print(f"Device: {lsac_agent.device}")
    print(f"Run directory: {run_dir.resolve()}")

    with episodes_path.open(
        "w", newline="", encoding="utf-8"
    ) as episodes_file:
        writer = csv.DictWriter(episodes_file, fieldnames=episode_fields)
        writer.writeheader()

        for episode_number in range(1, train_config.episodes + 1):
            obs, _ = env.reset()
            terminated = truncated = False
            episode_reward = 0.0
            episode_cost = 0.0
            episode_length = 0
            episode_metrics = defaultdict(list)

            while not (terminated or truncated):
                if total_steps < train_config.start_steps:
                    action = env.action_space.sample()
                else:
                    action = lsac_agent.get_action(obs)

                obs_next, reward, terminated, truncated, info = env.step(action)
                cost = info["cost"]
                memory.store(
                    obs,
                    action,
                    reward,
                    cost,
                    obs_next,
                    float(terminated),
                )

                obs = obs_next
                total_steps += 1
                episode_reward += reward
                episode_cost += cost
                episode_length += 1

                should_update = (
                    total_steps >= train_config.update_after
                    and memory.size >= train_config.batch_size
                    and (total_steps - train_config.update_after)
                    % train_config.update_every
                    == 0
                )
                if should_update:
                    for _ in range(train_config.gradient_steps):
                        diagnostics = lsac_agent.update(
                            memory.sample(train_config.batch_size)
                        )
                        for key, value in diagnostics.items():
                            episode_metrics[key].append(value)

            all_episode_rewards.append(episode_reward)
            average_reward = moving_average(
                all_episode_rewards, train_config.reward_smoothing
            )
            diagnostics = mean_metrics(episode_metrics)

            should_evaluate = (
                train_config.test_episodes > 0
                and (
                    episode_number % train_config.eval_interval == 0
                    or episode_number == train_config.episodes
                )
            )
            if should_evaluate:
                test_reward, test_cost = evaluate(
                    lsac_agent, test_env, train_config.test_episodes
                )
            else:
                test_reward = float("nan")
                test_cost = float("nan")

            row = {
                "episode": episode_number,
                "total_steps": total_steps,
                "episode_reward": episode_reward,
                "episode_cost": episode_cost,
                "episode_length": episode_length,
                "moving_average_reward": average_reward,
                "test_reward": test_reward,
                "test_cost": test_cost,
                **{
                    name: diagnostics.get(name, float("nan"))
                    for name in METRIC_FIELDS
                },
            }
            row["alpha"] = float(lsac_agent.alpha.detach().cpu())
            row["labda"] = float(lsac_agent.labda.detach().cpu())
            writer.writerow(row)
            episodes_file.flush()

            plotting.save_reward_curve(
                all_episode_rewards,
                reward_curve_path,
                train_config.reward_smoothing,
                title=f"LSAC on {train_config.env_id}",
            )
            lsac_agent.save(model_path)

            print(
                f"Episode {episode_number:04d}/{train_config.episodes:04d} | "
                f"steps {total_steps:07d} | "
                f"reward {episode_reward:10.3f} | "
                f"cost {episode_cost:10.3f} | "
                f"length {episode_length:4d} | "
                f"avg reward {average_reward:10.3f} | "
                f"alpha {float(lsac_agent.alpha.detach().cpu()):.4f} | "
                f"lambda {float(lsac_agent.labda.detach().cpu()):.4f}"
            )
            if should_evaluate:
                print(
                    f"  Evaluation | reward {test_reward:10.3f} | "
                    f"cost {test_cost:10.3f}"
                )

    env.close()
    test_env.close()
    print("Training complete.")
    print(f"Episode CSV: {episodes_path}")
    print(f"Reward curve PNG: {reward_curve_path}")
    print(f"Model: {model_path}")
    return lsac_agent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train episode-based SAC with a Lyapunov constraint."
    )
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-steps", type=int, default=1000)
    parser.add_argument("--update-after", type=int, default=1000)
    parser.add_argument("--update-every", type=int, default=50)
    parser.add_argument("--gradient-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--alpha3", type=float, default=0.2)
    parser.add_argument("--labda", type=float, default=0.1)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--q-lr", type=float, default=3e-4)
    parser.add_argument("--lyapunov-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--labda-lr", type=float, default=3e-4)
    parser.add_argument("--test-episodes", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--reward-smoothing", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="runs")
    return parser.parse_args()


def main():
    args = parse_args()
    train_config = config.TrainConfig(
        env_id=args.env_id,
        episodes=args.episodes,
        start_steps=args.start_steps,
        update_after=args.update_after,
        update_every=args.update_every,
        gradient_steps=args.gradient_steps,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        hidden_sizes=tuple(args.hidden_sizes),
        gamma=args.gamma,
        polyak=args.polyak,
        alpha=args.alpha,
        alpha3=args.alpha3,
        labda=args.labda,
        actor_lr=args.actor_lr,
        q_lr=args.q_lr,
        lyapunov_lr=args.lyapunov_lr,
        alpha_lr=args.alpha_lr,
        labda_lr=args.labda_lr,
        test_episodes=args.test_episodes,
        eval_interval=args.eval_interval,
        reward_smoothing=args.reward_smoothing,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )
    train(train_config)


if __name__ == "__main__":
    main()
