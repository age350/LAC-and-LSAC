# LAC：李雅普诺夫 Actor-Critic

该目录保留单 Lyapunov Critic 的 LAC：

```text
随机高斯 Actor
        |
        +-- 非负 Lyapunov Critic
```

核心网络、损失和默认超参数对照
`rickstaa/stable-learning-control` 当前 `main` 分支的 PyTorch LAC。
为了满足本项目的使用要求，训练循环按 episode 组织和打印，而不是按 epoch 记录。

## PyCharm

运行文件：

```text
C:\Users\16057\Desktop\stable-learning-control-main\lyapunov_sac_clean\trainer.py
```

Working directory：

```text
C:\Users\16057\Desktop\stable-learning-control-main\lyapunov_sac_clean
```

快速测试：

```text
--episodes 2 --update-after 20 --update-every 20
--gradient-steps 1 --batch-size 16 --hidden-sizes 32 32
--test-episodes 0
```

正式训练示例：

```text
--episodes 500 --device cpu --eval-interval 10
```

默认使用与原项目一致的线性学习率衰减。可用
`--lr-decay-type constant` 关闭衰减。

## 输出

每次训练在 `runs/日期时间/` 下生成：

- `episodes.csv`：每个 episode 的奖励、代价、长度、损失和学习率
- `reward_curve.png`：单条蓝色平滑 Returns 曲线
- `lac_model.pt`：模型参数
- `config.json`：本次训练参数

Pendulum 的原始 reward 非正，因此环境使用
`cost = -reward`，绘图使用 `episode_reward = -episode_cost`。

核心 Actor 损失：

```text
L_delta = mean(L(s_next, pi(s_next)) - L(s, action) + alpha3 * cost)
actor_loss = lambda * L_delta + alpha * mean(log_pi)
```
