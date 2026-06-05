# LSAC：SAC 加李雅普诺夫约束

该目录是独立的 LSAC 实现：

```text
随机高斯 Actor
    + Q1 Critic
    + Q2 Critic
    + Lyapunov Critic
```

训练按 episode 组织，每个 episode 结束后打印并写入 CSV。

## PyCharm

运行文件：

```text
C:\Users\16057\Desktop\stable-learning-control-main\lyapunov_sac_lsac\trainer.py
```

Working directory：

```text
C:\Users\16057\Desktop\stable-learning-control-main\lyapunov_sac_lsac
```

快速测试：

```text
--episodes 2 --start-steps 20 --update-after 20
--update-every 20 --gradient-steps 1 --batch-size 16
--hidden-sizes 32 32 --test-episodes 0
```

正式训练示例：

```text
--episodes 500 --device cpu --eval-interval 10
```

## 输出

每次训练在 `runs/日期时间/` 下生成：

- `episodes.csv`：每个 episode 的奖励、代价、长度和损失
- `reward_curve.png`：单条蓝色平滑 Returns 曲线
- `lsac_model.pt`：模型参数
- `config.json`：本次训练参数

核心目标：

```text
Q_target = reward + gamma * (1-done)
           * (min(Q1_target, Q2_target) - alpha * log_pi)

L_delta = mean(
    L(s_next, pi(s_next)) - L(s, action) + alpha3 * cost
)

actor_loss = SAC_loss + lambda * L_delta
```
