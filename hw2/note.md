## 指标
| 指标                    | 含义                    | 典型表现趋势              |
|-------------------|------------------|------------------|
| **Eval_AverageReturn** | 评估平均回报         | 越来越高               |
| **Eval_StdReturn**     | 评估回报标准差       | 先高后低（稳定后变小） |
| **Eval_MaxReturn**     | 评估最高回报       | 越来越高               |
| **Eval_MinReturn**     | 评估最低回报       | 越来越高               |
| **Eval_AverageEpLen**  | 评估平均episode长度 | 越来越长               |
| **Train_AverageReturn**| 训练平均回报         | 越来越高               |
| **Train_StdReturn**    | 训练回报标准差       | 先高后低               |
| **Train_MaxReturn**    | 训练最高回报       | 越来越高               |
| **Train_MinReturn**    | 训练最低回报       | 越来越高               |
| **Train_AverageEpLen** | 训练平均长度       | 越来越长               |
| **Actor Loss**         | 策略网络损失       | 先大后小，逐渐平稳 |
| **Train_EnvstepsSoFar**| 累计环境步数       | 单调增加               |
| **TimeSinceStart**     | 训练总耗时       | 单调增加               |
### 使用整条轨迹作为return
```bash
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
--exp_name cartpole
```
```
Eval_AverageReturn : 9.25
Eval_StdReturn : 0.772392988204956
Eval_MaxReturn : 11.0
Eval_MinReturn : 8.0
Eval_AverageEpLen : 9.25
Train_AverageReturn : 9.392523765563965
Train_StdReturn : 0.7071993350982666
Train_MaxReturn : 11.0
Train_MinReturn : 8.0
Train_AverageEpLen : 9.392523364485982
Actor Loss : 9.237488120561466e-05
Train_EnvstepsSoFar : 100405
TimeSinceStart : 21.20573902130127
```
### 使用return to go
```bash
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
-rtg --exp_name cartpole_rtg
```
```
Eval_AverageReturn : 9.325581550598145
Eval_StdReturn : 0.7061500549316406
Eval_MaxReturn : 11.0
Eval_MinReturn : 8.0
Eval_AverageEpLen : 9.325581395348838
Train_AverageReturn : 9.364485740661621
Train_StdReturn : 0.7284924983978271
Train_MaxReturn : 11.0
Train_MinReturn : 8.0
Train_AverageEpLen : 9.36448598130841
Actor Loss : 1.8799379176925868e-05
Train_EnvstepsSoFar : 100402
TimeSinceStart : 21.482080698013306
```
### 使用advance归一化
```bash
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
-na --exp_name cartpole_na
```
```
Eval_AverageReturn : 9.44186019897461
Eval_StdReturn : 0.7250445485115051
Eval_MaxReturn : 11.0
Eval_MinReturn : 8.0
Eval_AverageEpLen : 9.44186046511628
Train_AverageReturn : 9.373831748962402
Train_StdReturn : 0.8809868097305298
Train_MaxReturn : 11.0
Train_MinReturn : 8.0
Train_AverageEpLen : 9.373831775700934
Actor Loss : 1.3601615891735744e-15
Train_EnvstepsSoFar : 100432
TimeSinceStart : 21.48967170715332
```

### 同时使用return to go和advance归一化
```python
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
-rtg -na --exp_name cartpole_rtg_na
```
```
Eval_AverageReturn : 9.571428298950195
Eval_StdReturn : 0.7604151964187622
Eval_MaxReturn : 11.0
Eval_MinReturn : 8.0
Eval_AverageEpLen : 9.571428571428571
Train_AverageReturn : 9.392523765563965
Train_StdReturn : 0.758219838142395
Train_MaxReturn : 11.0
Train_MinReturn : 8.0
Train_AverageEpLen : 9.392523364485982
Actor Loss : -9.04969894967376e-16
Train_EnvstepsSoFar : 100398
TimeSinceStart : 21.996335744857788
```

### No baseline
```python
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
--exp_name cheetah
```
```
Eval_AverageReturn : -3966029.0
Eval_StdReturn : 0.0
Eval_MaxReturn : -3966029.0
Eval_MinReturn : -3966029.0
Eval_AverageEpLen : 1000.0
Train_AverageReturn : -3833606.5
Train_StdReturn : 437.410400390625
Train_MaxReturn : -3833047.25
Train_MinReturn : -3833975.0
Train_AverageEpLen : 1000.0
Actor Loss : -1606616832.0
Train_EnvstepsSoFar : 500000
TimeSinceStart : 139.9972860813141
```
### Baseline
```python
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
-n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
--use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline
```
```
val_AverageReturn : -3962683.5
Eval_StdReturn : 0.0
Eval_MaxReturn : -3962683.5
Eval_MinReturn : -3962683.5
Eval_AverageEpLen : 1000.0
Train_AverageReturn : -3828384.75
Train_StdReturn : 309.4350280761719
Train_MaxReturn : -3828048.25
Train_MinReturn : -3828879.0
Train_AverageEpLen : 1000.0
Actor Loss : -1532090752.0
Baseline Loss : 5585714688.0
Train_EnvstepsSoFar : 500000
TimeSinceStart : 145.9169261455536
```
### GAE with lambda_0
```python
python cs285/scripts/run_hw2.py \
--env_name LunarLander-v2 --ep_len 1000 \
--discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
--use_reward_to_go --use_baseline --gae_lambda 0 \
--exp_name lunar_lander_lambda_0
```
```
Eval_AverageReturn : -236.57147216796875
Eval_StdReturn : 58.10466766357422
Eval_MaxReturn : -160.9412841796875
Eval_MinReturn : -324.0888366699219
Eval_AverageEpLen : 111.0
Train_AverageReturn : -216.97317504882812
Train_StdReturn : 111.0882797241211
Train_MaxReturn : 22.875511169433594
Train_MinReturn : -372.0850830078125
Train_AverageEpLen : 114.47368421052632
Actor Loss : 0.16357816755771637
Baseline Loss : 5113.09033203125
Train_EnvstepsSoFar : 625936
TimeSinceStart : 199.10666799545288
```
### GAE with lambda_0.95
```python
python cs285/scripts/run_hw2.py \
--env_name LunarLander-v2 --ep_len 1000 \
--discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
--use_reward_to_go --use_baseline --gae_lambda 0.95 \
--exp_name lunar_lander_lambda_0.95
```
```
Eval_AverageReturn : -703.693359375
Eval_StdReturn : 165.2461700439453
Eval_MaxReturn : -436.8841857910156
Eval_MinReturn : -914.60302734375
Eval_AverageEpLen : 72.66666666666667
Train_AverageReturn : -557.3521118164062
Train_StdReturn : 170.95262145996094
Train_MaxReturn : -340.83526611328125
Train_MinReturn : -996.9075317382812
Train_AverageEpLen : 65.35483870967742
Actor Loss : 1.9451850652694702
Baseline Loss : 17415.619140625
Train_EnvstepsSoFar : 623449
TimeSinceStart : 199.12210321426392
```
### GAE with lambda_0.98
```python
python cs285/scripts/run_hw2.py \
--env_name LunarLander-v2 --ep_len 1000 \
--discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
--use_reward_to_go --use_baseline --gae_lambda 0.98 \
--exp_name lunar_lander_lambda_0.98
```
```
Eval_AverageReturn : -721.1944580078125
Eval_StdReturn : 77.77672576904297
Eval_MaxReturn : -657.9852294921875
Eval_MinReturn : -871.8558959960938
Eval_AverageEpLen : 102.8
Train_AverageReturn : -1019.3593139648438
Train_StdReturn : 900.0762329101562
Train_MaxReturn : -359.409423828125
Train_MinReturn : -3640.85302734375
Train_AverageEpLen : 137.66666666666666
Actor Loss : -1.2667990922927856
Baseline Loss : 82444.8046875
Train_EnvstepsSoFar : 620948
TimeSinceStart : 202.87654829025269
```
### GAE with lambda_1.0
```python
python cs285/scripts/run_hw2.py \
--env_name LunarLander-v2 --ep_len 1000 \
--discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
--use_reward_to_go --use_baseline --gae_lambda 1.0 \
--exp_name lunar_lander_lambda_1.0
```
```
Eval_AverageReturn : -648.7041015625
Eval_StdReturn : 150.25228881835938
Eval_MaxReturn : -478.6739196777344
Eval_MinReturn : -880.2762451171875
Eval_AverageEpLen : 102.0
Train_AverageReturn : -1030.9447021484375
Train_StdReturn : 780.5755004882812
Train_MaxReturn : -396.80255126953125
Train_MinReturn : -3152.17529296875
Train_AverageEpLen : 135.2
Actor Loss : -0.008382154628634453
Baseline Loss : 61681.1796875
Train_EnvstepsSoFar : 620937
TimeSinceStart : 201.46896505355835
```
### all the things together
```python
python cs285/scripts/run_hw2.py \
--env_name Humanoid-v4 --ep_len 1000 \
--discount 0.99 -n 1000 -l 3 -s 256 -b 50000 -lr 0.001 \
--baseline_gradient_steps 50 \
-na --use_reward_to_go --use_baseline --gae_lambda 0.97 \
--exp_name humanoid --video_log_freq 5
```
|## 各实验总结表

| 实验 | 环境 | 主要设置 | 主要观察 |
|---|---|---|---|
| **CartPole-整轨迹回报** | CartPole-v0 | 无rtg、无baseline | 最简单，平均回报9.25，表现一般 |
| **CartPole-RTG** | CartPole-v0 | `-rtg` | 加了RTG，略好，平均回报9.33 |
| **CartPole-优势归一化** | CartPole-v0 | `-na` | 加优势归一化，回报9.44，略好于纯RTG |
| **CartPole-RTG+归一化** | CartPole-v0 | `-rtg -na` | RTG+归一化最好，9.57，说明归一化对小任务有效 |
| **HalfCheetah-无baseline** | HalfCheetah-v4 | 纯PG+RTG | 回报非常负（-396万），完全不行 |
| **HalfCheetah-有baseline** | HalfCheetah-v4 | baseline+RTG | 仍然非常负（-396万），说明PG对高维控制力不足 |
| **LunarLander-GAE λ=0** | LunarLander-v2 | GAE无平滑 | 回报-236，表现比高λ好很多 |
| **LunarLander-GAE λ=0.95** | LunarLander-v2 | GAE λ=0.95 | 回报-703，λ过高导致优势估计过于不稳定 |
| **LunarLander-GAE λ=0.98** | LunarLander-v2 | GAE λ=0.98 | 回报-721，继续变差，说明高λ对LunarLander完全不适合 |
| **LunarLander-GAE λ=1.0** | LunarLander-v2 | GAE λ=1.0 | 回报-648，比0.98略好，但仍然很差 |
| **Humanoid-全配置大作业** | Humanoid-v4 | GAE+RTG+baseline+归一化+50步baseline更新 | 结果待补充（请补充结果） |

---

## 经验总结

| 经验 | 解释 |
|---|---|
| **RTG适合短程和稀疏奖励环境** | 因为RTG只考虑未来，适合长周期稀疏奖励 |
| **优势归一化对所有环境都有帮助** | 归一化能有效防止数值爆炸，尤其是小batch任务 |
| **PG不适合高维控制（如HalfCheetah）** | 高维环境PG的探索效率极低，几乎必崩 |
| **GAE λ过高反而有害** | 特别是高噪声环境，λ=1会导致优势估计剧烈波动 |
| **baseline训练次数要足够** | 否则baseline不准，反而拖累优势估计 |

---