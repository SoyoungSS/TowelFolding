# ACT第二阶段：无专家实时闭环

2026-11-12 的模型已在固定场景连续完成 4 次无专家折叠验收，见 [改进与验收记录](IMPROVEMENT_20261112.md)。早期 200 次更新模型的失败记录保留于 [ROLLOUT_VALIDATION.md](ROLLOUT_VALIDATION.md)。

## 日常使用

在工程根目录执行：

```bash
./run_act_live.sh
```

脚本优先使用 `deployment.json` 指定且哈希匹配的已验证模型，启动Newton物理进程、独立ACT推理进程和Isaac窗口。当前部署参数为30步动作块、29秒**仿真时间**，然后停止控制并保留窗口供观察。只有不存在部署配置时才回退到最近一次第一阶段验收通过的checkpoint（30秒）。可通过 `--checkpoint`、`--action-steps` 和 `--duration` 显式覆盖参数。

窗口中的 `ACT policy evaluation` 面板提供：

- `Play / Pause`：暂停或继续；暂停时不继续推进Newton。
- `Show / hide cameras`：显示模型使用的头部、左腕、右腕RGB。
- `Exit all processes`：关闭本次运行与后台worker。

终端出现 `ACT EVALUATION COMPLETE` 后读取 `task_success`，或打开本次输出中的 `evaluation.json`。`false` 表示评价已运行完毕，但策略没有完成任务。Isaac可能在关闭时直接以0退出，所以**不要用终端退出码判断折叠成功**。提前关闭窗口会在 `result.json` 中标记 `aborted`；异常标记 `error`，此时没有完整任务结论。

需要运行完自动关闭窗口：

```bash
./run_act_live.sh --exit-on-done
```

指定模型（必须同时保留模型旁的处理器与runtime.json）：

```bash
./run_act_live.sh --checkpoint outputs/act_stage1/20261110_165233_89112/checkpoint
```

首次启动Isaac可能出现许可证确认；按本机安装要求处理。训练环境仍是 `.runtime/act-env311`，不向Isaac环境安装LeRobot。

## 每次控制发生什么

1. Newton返回当前实际关节和毛巾状态，Isaac同步机器人与布料位置。
2. 同一次渲染获取三路640×480 RGB；相机时间必须相同且递增。
3. **只有三路RGB与16维实际关节状态传给ACT**。毛巾粒子真值只用于显示与独立结果评价。
4. 图像完整视野缩放为320×240，再使用checkpoint自带的标准归一化处理器。
5. 官方ACT以eval模式、z=0输出30步动作块；默认 `n_action_steps=30`，按序执行整块后，用最新观测重新预测。`--action-steps 1` 可测试每步重新预测，但不属于当前通过验收的部署参数。
6. 后处理器反归一化动作；转换关节顺序，将夹爪closed预测按0.5阈值转换为开度0.5或0.04。关节目标经过URDF限位，运动速度沿用仿真原有约束。
7. 保持该动作推进两个60Hz物理步，即一个30Hz控制周期，再重复上述流程。

这里的30Hz表示**仿真时钟中的控制频率**。物理、推理、渲染采用同步等待，机器渲染较慢时，运行30秒仿真会花费更多实际时间。当前未证明30Hz墙钟实时性能。

策略不读取训练视频、示范动作、专家阶段或毛巾几何。ACT的VAE专家动作编码器加了推理调用拦截；Newton专家启动、专家动作与规划入口同样加拦截，调用即报错。结果中 `expert_calls` 必须为0。

`run_collect.sh` 仍用于专家数据采集；`run_act_live.sh` 用于模型独立控制，两者是不同入口。这里不会失败后自动调用专家，也不会自动修改权重。

## 对折通过条件

在指定评价时长结束时，同时满足：

- 左右夹爪都曾通过原有局部抓取规则实际抓到毛巾粒子。
- 结束时粒子全部释放。
- 对称列粒子在桌面平面上的平均配对误差小于25mm。
- 最大配对误差小于50mm。
- 毛巾沿折叠方向的占地宽度小于初始宽度的68%。

几何判据沿用固定工况的专家评价定义，抓取与释放检查用于排除未操作或仍夹持的情况。机器人仍采用运动学控制，夹爪区域内粒子有辅助绑定。这能检验当前仿真任务，不能直接代表真实摩擦抓取或真机性能。`--seed` 只记录种子，当前初态没有随机化；重复同一初态不等于泛化测试。

## 输出位置与含义

```text
outputs/act_rollout/<日期_时间_进程号>/
├── evaluation.json          完整任务结论、几何误差、抓取记录、推理耗时
├── result.json              程序是否完成/异常/提前中止，worker是否退出
├── frames.jsonl             每个控制周期的预测、执行目标、帧编号与误差
├── trajectory.npz           实际状态、毛巾粒子和预测/执行动作
├── videos/                  三路30Hz RGB视频（仿真时钟）
├── head_initial.png / head_final.png
├── left_wrist_initial.png / left_wrist_final.png
├── right_wrist_initial.png / right_wrist_final.png
├── metadata.json            本次物理与相机配置
├── training_scene.json      训练采集时配置，用于一致性检查
├── policy_ready.json        模型与接口信息
├── live_status.json         窗口运行中的最新进度
└── newton.log / policy.log  两个独立worker日志
```

当前29秒部署运行有870次动作、871个状态和每路871帧视频；30秒运行则有900次动作、901个状态和每路901帧视频。最后一帧是最后一步执行后的结果，不再输入下一次动作。`frames.jsonl` 第t行对应 `trajectory` 中状态t到状态t+1的动作。

`trajectory.npz` 的 `joint_state` 与 `action` 使用Newton内部顺序：左臂7、右臂7、左开度、右开度。`canonical_policy_action` 使用训练顺序：左臂7、左closed、右臂7、右closed。两者不可混用。该文件用于闭环诊断，不是自动生成的专家LeRobot训练数据。

## 接下来如何根据结果行动

先检查独立控制能否完成抓取、抬起、翻折和释放，再定位失败阶段。单条示范加200次更新只做过接口与小样本拟合检查；即使离线误差下降，闭环偏离也可能不断累积。应根据轨迹诊断，再决定延长训练、补充成功示范与恢复示范，并保留完整episode作为独立评价数据。
