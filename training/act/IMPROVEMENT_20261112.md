# ACT 固定场景折叠改进（2026-11-12）

本轮保留原有物理、抓取规则和验收线：平均配对误差 <25 mm、最大配对误差 <50 mm、折后宽度比例 <0.68，且双侧实际抓取、最终释放。策略输入仍仅为三路 RGB 和实际关节状态，没有加入时钟、布料角点、专家回放或运行中专家接管。

## 训练与工程修改

从 `outputs/act_stage1/rollout_correction_z0/checkpoint` 继续训练 6000 步，使用原始 815 帧和已有纠偏 558 帧，共 1373 帧；两组各占一半采样概率。官方 ACT 网络、batch size 2、学习率 1e-5，使用已有的零潜变量手臂/夹爪 L1 微调目标。原数据和旧模型保留。

```bash
./run_act_stage1.sh \
  --dataset outputs/collect/20261109_152842_33675/dataset \
  --recovery-dataset outputs/collect/rollout_correction_frame20/dataset \
  --init-checkpoint outputs/act_stage1/rollout_correction_z0/checkpoint \
  --balance-recovery --finetune-z0 --steps 6000 \
  --output outputs/act_stage1/新的运行目录
```

模型保存在 `outputs/act_stage1/correction_refined_retry_20261112/checkpoint`。固定训练片段 z=0 L1 从 0.200102 降至 0.059768，保存/重载最大差为 0；该指标不是独立验证集指标。

工程新增：

- 微调加载原 checkpoint 的输入/输出处理器，避免多 episode 数据统计变化使旧权重的数值含义改变。本轮仍使用原数据统计，不能把本次性能改善归因于该修复。
- 每 1000 步保存可部署中途模型；`--save-every` 可调整，`--learning-rate` 可覆盖学习率。
- 批量采集支持 `--resume`，保留中断目录、已完成轮次和原随机序列。
- `training/act/evaluate_closed_loop.py` 独立重复运行完整任务，并检查原有所有验收条件。
- 默认启动优先读取 `training/act/deployment.json` 中经过闭环验证的权重与执行参数，并核对权重 SHA-256，防止新的试验训练自动替换已验证模型。

此前启动的 12 轮采集和首次训练因中断未完成；其输出保留为诊断材料，未用于本轮成功模型。恢复后串行执行训练与仿真，避免同时运行 Isaac、训练和测试造成主机内存争用。

## 闭环验证

每次独立启动物理与策略进程，从标准初态开始，持续 29 秒仿真时间，每次执行 30 步动作块。没有使用恢复计划或分阶段切换模型；专家调用为 0。

首次运行（seed 0）通过：平均误差 16.617 mm、最大误差 36.533 mm、宽度比例 0.554965。完整结果位于 `outputs/act_rollout/correction_refined_retry_20261112/run_000/evaluation.json`，并保留三路视频各 871 帧。

重复验证结果以 `outputs/act_rollout/correction_refined_repeat3_20261112/report.json` 为准；未完成的轮次不计为通过。

本结果范围为固定位置、相同机器人与布料参数、运动学机器人及已有辅助抓取规则的仿真。不同初始位置、扰动和真实机械臂未在本轮验收中验证。

## 最终验收记录

首次测试加三次独立重复，4/4 通过。每轮均为同一权重，30 步动作块，29 秒仿真时间，870 个控制周期，三路视频各 871 帧。

| seed | 平均误差 mm | 最大误差 mm | 宽度比例 | 专家调用 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 16.617 | 36.533 | 0.554965 | 0 | 通过 |
| 1 | 9.190 | 14.930 | 0.560477 | 0 | 通过 |
| 2 | 6.707 | 6.857 | 0.567012 | 0 | 通过 |
| 3 | 11.417 | 20.567 | 0.562452 | 0 | 通过 |

默认部署配置已保存。运行：

```bash
./run_act_live.sh
```

该命令选择上述已验证模型，执行 30 步动作块，评价 29 秒；不会选取时间较新的未验证试验模型。

权重 SHA-256：`0c60029024bcc6e1e69720df3bfa2089d7fef4c3eab28099fb0e4c99c399d634`。

原自主基线平均误差 50.963 mm、最大误差 77.688 mm；本轮四次均低于原验收线。这是固定场景的重复实测结果，不是对任意初态成功率的保证。
