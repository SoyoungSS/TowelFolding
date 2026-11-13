# ACT第一阶段：标准数据加载、短程训练与模型重载

此目录使用官方 `lerobot==0.4.4` 的 `LeRobotDataset`、`ACTPolicy` 和预处理/后处理流水线。Python训练循环负责本工程的单条数据验收、日志及保存重载检查，没有复制或缩小官方ACT网络。

本阶段不接管Newton，不发送真机命令，不声称单条示范具有泛化能力。

## 运行

在工程根目录执行：

```bash
./run_act_stage1.sh
```

默认运行200次更新，batch size为2；训练完成自动执行模型与处理器保存、重载、无专家动作推理及数值一致性检查。终端最后出现 `ACT STAGE1 PASSED`，且输出目录存在 `SUCCESS.json`，才表示整个阶段通过。

环境位于 `.runtime/act-env311`，是独立Python 3.11虚拟环境，`include-system-site-packages=false`。只借用已有Isaac环境的解释器创建venv，不使用或修改Isaac的site-packages。LeRobot 0.4.4支持此Python版本；不跟随main分支升级。首次安装可能下载数GB依赖，后续使用READY标记检查现有环境。

依赖安装入口 `training/act/setup.sh`，固定主依赖见 `environments/act-requirements.txt`。安装后完整依赖快照保存在 `.runtime/act_setup/installed-freeze.txt`。如有本地wheel目录，可设置 `ACT_WHEELHOUSE=/绝对路径` 供安装器复用。

训练需要宿主CUDA访问；沙箱或容器无法访问GPU时会明确报错，不自动转CPU。需要手动选择CPU时可传 `--device cpu`，但标准网络会明显更慢。

自定义示例：

```bash
./run_act_stage1.sh --steps 300 --batch-size 2
./run_act_stage1.sh --dataset /绝对路径/dataset --steps 200
```

每次创建新输出目录。可指定 `--output outputs/act_stage1/自定义名称`，该目录必须尚不存在，避免覆盖已有模型。

## 数据与训练约定

- 默认读取 `outputs/collect/*/dataset/SUCCESS.json` 中最近成功的数据，仅接受单条30Hz示范。
- 原始数据保留不动，复制到本次输出的 `dataset/` 后处理任务文本的Pandas索引兼容问题。
- 原始数据所有文件记录SHA-256；副本除明确记录的元数据修正外，保持一致。
- 使用官方数据读取器和PyAV视频后端，训练和数据读取不需要访问Hugging Face Hub，不上传模型或数据。
- 三路图像等比例缩至320×240，保留完整视野，不使用中心裁剪；配置保存到模型旁的 `runtime.json`。
- 输入为三路RGB和16维实际状态，输出为30步16维目标动作。参数维度及夹爪编码沿用原数据。
- 标准ACT配置：ResNet-18 ImageNet预训练、512维、8头、4层观测编码器、1层解码器、4层VAE编码器、32维潜变量、KL权重10。视觉骨干参与训练。
- 标准处理器从当前训练数据读取统计量，统一做归一化、设备放置和动作反归一化。
- 使用官方参数分组及默认学习率1e-5，AdamW、梯度裁剪1，不使用教学网络权重初始化。
- 每次查询只执行1步的配置仅用于推理接口验收；本阶段没有仿真动作执行。

## 自动验收内容

1. 官方接口读取三路RGB、状态、动作和任务文字。
2. 检查首帧、普通帧及末尾两帧的动作块、填充掩码与当前目标对齐。
3. 标准ACT执行前向、CVAE损失、反向传播和参数更新，拒绝非有限损失或梯度。
4. 记录固定8个训练片段的z=0推理L1，要求短程训练后低于初始值。
5. 无答案推理不调用VAE编码器。
6. 保存官方模型配置、safetensors权重、标准预处理和后处理配置及状态。
7. 重载后检查归一化动作块、反归一化动作块与保存前一致。
8. 清空动作队列后，单步select_action与预测块第一步一致。

固定片段全部来自当前训练episode。曲线叫训练监测，不是独立验证集或测试集，不计算闭环成功率。

## 输出文件

```text
outputs/act_stage1/<日期_时间_进程号>/
├── dataset/                  官方接口读取的训练副本
├── data_preparation.json     原始哈希和副本兼容调整记录
├── dataset_check.json        数据形状和边界检查
├── environment.json          Python/LeRobot/PyTorch/GPU版本
├── train_metrics.jsonl       每步损失、梯度、耗时及定期z=0监测
├── checkpoint/
│   ├── config.json           官方ACT配置
│   ├── model.safetensors     官方模型权重
│   ├── policy_preprocessor.json / 对应safetensors
│   ├── policy_postprocessor.json / 对应safetensors
│   └── runtime.json          相机/关节顺序、完整视野缩放及频率
├── optimizer_state.pt        优化器状态；暂未提供一键续训
├── inference_example.npz     验收用预测示例
├── training.png              固定训练片段推理误差曲线
└── SUCCESS.json              整体通过标记
```

失败会尽可能写出 `FAILED.json` 并以非零状态退出，不生成成功标记。仅有checkpoint文件不等于验收全部通过。

## 单独重载推理

训练完成后使用实际输出路径：

```bash
./run_act_predict.sh --checkpoint outputs/act_stage1/实际运行目录/checkpoint --frame 100
```

该命令在独立进程中加载模型和处理器，读取保存路径对应的数据，打印动作块形状和第一步反归一化目标；不执行动作。当前需保留训练副本，若移动数据则需更新 `runtime.json` 中的副本路径。

输出关节目标为弧度，夹爪预测是连续数值，即使训练标签为0/1也可能超出此范围。接入仿真前还必须做关节重排、夹爪编码转换、限位限速和独立任务结束评价。

## 第二阶段：实时闭环

### 持续训练与重复任务验收

微调时使用 `--init-checkpoint` 加载权重，现同时沿用该 checkpoint 的输入归一化和动作反归一化处理器。新增数据不会改变已有模型的数值坐标系；从零训练仍从完整主示范计算统计量。`--learning-rate` 可单独调整本次优化器学习率。

默认每 1000 步保存 `checkpoint_step_XXXXXX/`，内含权重、处理器、运行配置和优化器状态。可用 `--save-every N` 修改间隔或以 0 禁用。中途模型可通过 `--init-checkpoint` 继续微调，但这会重建优化器与采样器，不是逐位一致的训练续跑；中途保存也不代表训练或折叠验收通过。

完成训练后，使用独立进程重复评价原有折叠标准：

```bash
.runtime/act-env311/bin/python training/act/evaluate_closed_loop.py \
  --checkpoint outputs/act_stage1/实际模型/checkpoint \
  --output outputs/act_rollout/实际模型_repeat3 --repeats 3 --action-steps 30
```

每次从标准初态开始，保存视频和评价。只有全部运行完成、无专家调用、双侧抓取及释放、平均/最大边缘误差与宽度比例全部达标，汇总 `report.json` 才会标为通过。固定场景重复通过不等于不同初始位置上的泛化通过。

阶段1通过后，运行 `./run_act_live.sh` 接入ACT实时闭环。该入口使用独立worker，禁用专家，以当前三路RGB和实际关节状态控制Newton，并独立判断任务是否成功。用法与结果说明见 [ROLLOUT.md](ROLLOUT.md)。`run_collect.sh` 仍保留专家数据采集行为。阶段1通过不代表闭环折叠成功，后续仍需要完整任务评价、多episode数据和必要的恢复示范。

## 已完成的本机验收

200次更新、batch 2的第一阶段已通过，训练片段推理L1从1.0197降至0.2552，独立进程重载第0帧预测与保存值一致。输出为 `outputs/act_stage1/20261110_165233_89112`，完整结果见 [VALIDATION.md](VALIDATION.md)。该验收只针对短程训练与接口；闭环评价独立记录于第二阶段。
