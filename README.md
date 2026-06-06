# LAC 与 LSAC 项目说明

本项目包含两个连续控制强化学习算法：

```text
LAC-and-LSAC
├── lyapunov_ac    # LAC：最大熵随机策略 + Lyapunov Critic
└── lyapunov_sac   # LSAC：完整 SAC + Lyapunov Critic
```

两套算法默认使用 `Pendulum-v1` 环境，并按 episode 训练、打印和记录结果。

## 1. LAC 是不是标准 SAC

原开源项目中的 LAC 全称为：

```text
Lyapunov (Soft) Actor-Critic
```

它不是标准 SAC，也不是简单地在完整 SAC 上增加一个 Lyapunov 网络。

LAC 使用了 SAC 中的以下机制：

- 随机高斯策略
- 重参数化采样
- `tanh` 动作压缩
- 最大熵正则项
- 自动温度系数 `alpha`

但是，LAC 没有标准 SAC 中的：

- `Q1`、`Q2` 双 Q 网络
- `min(Q1, Q2)` 目标
- SAC 的 Q Bellman 更新
- `alpha * log_pi - Q` 形式的 Actor 损失

LAC 使用 Lyapunov Critic 代替传统 Q Critic，其 Actor 损失为：

```text
actor_loss = lambda * L_delta + alpha * mean(log_pi)
```

其中：

```text
L_delta = mean(
    L(s_next, pi(s_next))
    - L(s, action)
    + alpha3 * cost
)
```

因此，更准确的描述是：

> LAC 在 Actor-Critic 框架中使用 Lyapunov 稳定性约束，并采用 SAC 的最大熵随机策略机制，但它不是完整的双 Q SAC。

参考实现：

https://github.com/rickstaa/stable-learning-control/blob/main/stable_learning_control/algos/pytorch/lac/lac.py

## 2. LSAC 与 LAC 的区别

本项目中的 LSAC 保留完整的 SAC 结构，再增加 Lyapunov Critic：

```text
随机高斯 Actor
├── Q1 Critic
├── Q2 Critic
└── Lyapunov Critic
```

标准 SAC 部分的目标为：

```text
Q_target = reward + gamma * (1-done)
           * (min(Q1_target, Q2_target) - alpha * log_pi)
```

SAC Actor 损失为：

```text
sac_actor_loss = mean(alpha * log_pi - min(Q1, Q2))
```

加入 Lyapunov 约束后的总 Actor 损失为：

```text
actor_loss = sac_actor_loss + lambda * L_delta
```

所以：

| 算法 | 随机最大熵 Actor | Q1/Q2 | Lyapunov Critic |
|---|---:|---:|---:|
| LAC | 是 | 否 | 是 |
| LSAC | 是 | 是 | 是 |

## 3. 三个网络类的作用

### `SquashedGaussianActor`

这是策略网络，负责根据状态生成连续动作。

网络首先输出高斯分布的均值和标准差：

```text
mu(s), sigma(s)
```

然后进行重参数化采样：

```text
u ~ Normal(mu(s), sigma(s))
```

最后通过 `tanh` 将动作压缩到有限范围：

```text
a = action_scale * tanh(u) + action_bias
```

它还会返回动作的对数概率 `log_pi`，用于最大熵训练和自动调节 `alpha`。

### `LyapunovCritic`

这是李雅普诺夫评价网络。

输入为状态和动作：

```text
(s, a)
```

输出为：

```text
L(s, a)
```

网络对最后一层特征进行平方求和：

```text
L(s,a) = sum(features^2)
```

因此可以保证：

```text
L(s,a) >= 0
```

该网络用于估计累计代价，并构造 Lyapunov 稳定性约束。

### `LyapunovActorCritic`

这是组合网络，用于统一管理：

```text
LyapunovActorCritic
├── SquashedGaussianActor
└── LyapunovCritic
```

它不是第三种算法，只是将 Actor、Critic、目标网络、动作生成和模型参数组织在一起。

## 4. 当前控制环境

两个算法默认使用：

```text
Pendulum-v1
```

这是一个单摆连续控制环境。智能体需要向转轴施加连续力矩，使摆杆转到竖直向上的位置并保持稳定。

### 状态

状态为 3 维：

```text
cos(theta)
sin(theta)
角速度
```

### 动作

动作为 1 维连续力矩：

```text
torque in [-2, 2]
```

### 控制目标

- 减小摆杆与竖直向上位置的角度误差
- 减小角速度
- 避免使用过大的控制力矩

Pendulum 的奖励近似为：

```text
reward = -(
    angle_error^2
    + 0.1 * angular_velocity^2
    + 0.001 * torque^2
)
```

奖励越接近 `0`，控制效果越好。

## 5. LAC 中的 reward 与 cost

LAC 是代价最小化算法，训练时只使用非负 `cost`：

```text
cost = -reward
```

Lyapunov Critic 的目标为：

```text
L_target = cost + gamma * (1-done) * L_target(s_next, a_next)
```

代码中记录：

```text
episode_reward = -episode_cost
```

该 `episode_reward` 主要用于打印、CSV 和绘图，不单独参与 LAC 的网络更新。

## 6. 当前 LSAC 中的 reward 与 cost

当前 LSAC 同时使用：

```text
reward = Pendulum 原始奖励
cost = -reward
```

因此它们只是符号相反：

```text
最大化 reward <=> 最小化 cost
```

在当前实现中：

- `Q1`、`Q2` 使用 reward 学习性能目标
- Lyapunov Critic 使用 cost 构造稳定性约束

因为 `cost = -reward`，两者实际上来自同一个函数。也就是说，同一个控制目标一边用于最大化 Q 值，一边用于构造 Lyapunov 约束。

这种设计可以用于验证算法和稳定性正则化，但它不属于严格意义上的独立安全约束。

## 7. 独立 reward 和 cost 的设计

如果要研究真正的约束强化学习，建议将性能奖励和安全代价分开。

例如：

```text
reward = -(
    angle_error^2
    + control_performance_term
)
```

安全代价可以定义为：

```text
cost =
    angular_velocity_limit_violation
    + torque_limit_violation
    + unsafe_angle_violation
```

这样两部分具有不同作用：

```text
Q1、Q2：
    最大化控制性能

Lyapunov Critic：
    限制安全代价并满足稳定性约束
```

## 8. Lambda 的含义

`lambda` 是 Lyapunov 约束的拉格朗日乘子，用于调节约束项在 Actor 损失中的权重。

```text
lambda 越大：
    Actor 越重视 Lyapunov 约束

lambda 越小：
    Actor 越重视原始策略优化目标
```

LAC 默认从约 `0.99` 开始，是为了与原开源项目默认参数一致。

LSAC 默认从 `0.1` 开始，是为了避免训练初期 Lyapunov 项压过 SAC 的 Q 值优化。

即使两个算法使用相同的初始 `lambda`，由于损失函数和数值尺度不同，它们的 `lambda` 也不能直接进行横向比较。

## 9. 训练输出

每次训练都会生成：

```text
episodes.csv
reward_curve.png
模型参数文件
config.json
```

奖励图只保存 PNG。

曲线含义：

```text
横轴：Episodes
纵轴：Returns
蓝线：平滑后的 episode reward
```

