# 双臂叠毛巾：实时仿真与数据采集

**2026-11-12：ACT 已在当前固定场景连续完成 4 次无专家折叠验收。** 平均边缘误差为 6.7–16.6 mm，最大误差为 6.9–36.5 mm，未放宽原有标准。直接运行 `./run_act_live.sh` 使用已验证权重；训练、结果及适用范围见 [本轮改进记录](training/act/IMPROVEMENT_20261112.md)。

默认工况为固定位置的 30×30 cm 平铺毛巾，由双臂轨迹专家完成对折。批量采集可改变每轮毛巾的初始平移。Newton负责物理，Isaac Sim打开实时窗口并同步采集头部、左右腕三路RGB、实际关节状态和目标动作，输出LeRobot v3结构的数据。

## 运行

在图形桌面终端执行：

```bash
cd /home/shendongran/code/newton_towel_sim
./run_collect.sh
```

脚本自动启动两个环境，无需手动激活Conda。首次使用Isaac Sim时，需按程序提示阅读并接受许可。基础环境配置见 [环境说明](docs/ENVIRONMENT.md)，Isaac安装记录见 [实验记录](docs/ISAAC_ACT_LEARNING.md)。脚本会检查采集用PyArrow，缺失时自动联网安装固定版本19.0.1到 `.runtime/collector_deps`，不修改Isaac环境；清理 `outputs` 不会删除运行依赖。

运行后：

1. 等待Isaac窗口打开，首次启动和编译可能较慢。
2. 专家自动执行一次对折，同时采集数据；无需点击开始。
3. 可用 `Play / Pause` 暂停，用 `Show / hide cameras` 查看三路画面。采集模式不支持窗口内重置；需要再采一次时重新运行脚本。
4. 等终端出现 `DATASET SAVED`，数据已保存，再关闭窗口或点击 `Exit both processes`。

窗口实时反映仿真状态；渲染速度较慢时，仿真会等待相机，实际耗时可能长于任务的约27秒仿真时间。提前关闭窗口或折叠失败不会产生成功数据标记。

## 批量采集、训练与仿真

```bash
cd /home/shendongran/code/newton_towel_sim
./run_collect.sh --episodes 5 --offset-mm 3 --output outputs/collect/my_batch
./run_act_stage1.sh --collection-batch outputs/collect/my_batch/batch.json \
  --steps 2000 --output outputs/act_stage1/my_batch_model
./run_act_live.sh --checkpoint outputs/act_stage1/my_batch_model/checkpoint \
  --action-steps 30 --goal fold --duration 29
```

批量采集使用无窗口渲染。第一轮使用原始位置，后续每轮按种子在 X 和 Y 方向分别采样不超过指定范围的初始平移。每轮生成独立的 episode 目录，包含三路 RGB、16 维状态和 16 维动作。每轮完成后更新 `batch.json`，记录偏移、验收状态和日志位置。训练只读取清单中通过折叠验收并写出成功标记的数据。输出目录须是新的目录。

若批量采集中断，保持原来的 `--episodes`、`--seed`、`--offset-mm` 和 `--output` 参数，再加 `--resume` 可继续。已记录完成的轮次会跳过，未完成目录会保留为 `interrupted_*` 并重新采集该轮。建议采集、训练和闭环评价依次运行，避免 Isaac 与训练进程争用主机内存。

训练将各 episode 的样本一起采样，动作块在各自 episode 内构造，末尾不足 30 步的部分按原有掩码处理。归一化统计从纳入训练的完整示范合并计算。最后一条命令从标准毛巾位置启动实时窗口，查看输出的 `evaluation.json` 判断自主策略是否通过。2000 步是起始训练预算，效果由闭环测试决定。

## 数据在哪里

每次创建独立目录，终端会打印实际路径：

```text
outputs/collect/日期_时间_进程号/
├── dataset/
│   ├── SUCCESS.json                 成功采集标记
│   ├── data/chunk-000/file-000.parquet
│   ├── videos/observation.images.head/...
│   ├── videos/observation.images.left_wrist/...
│   ├── videos/observation.images.right_wrist/...
│   └── meta/                        info、stats、tasks、episodes、字段说明
├── completed_episode.json           最终折叠质量
├── console.log                      主进程日志
└── newton.log                       物理进程日志
```

只使用包含 `dataset/SUCCESS.json` 的数据。没有该文件时，查看日志；失败或中途关闭产生的文件不能当作成功示范。

数据按均匀30 Hz仿真时间采集；三路RGB均为640×480。每条观测对应执行前的图像和实际状态，动作是随后保持一个控制周期的关节目标。状态和动作均为16维：

```text
[左臂7关节, 左夹爪closed, 右臂7关节, 右夹爪closed]
```

关节单位为弧度；夹爪closed以仿真开度小于0.1转换为1，否则为0。字段和目录结构对齐参考数据 `/home/shendongran/code/FoldedTowelsDataset/fold-flat-towel`；视频使用MPEG-4编码，参考数据为AV1，不复制其删帧和时间压缩。

实测已完成一次带实时窗口的完整采集：815条记录、30 Hz、约27.17秒仿真时间，三路视频均逐帧解码检查通过，最终平均对齐误差16.2 mm。当前训练使用的成功数据位于 `outputs/collect/20261109_152842_33675/dataset`。

## 用于ACT的范围

这一步生成视觉模仿学习所需的RGB、状态和动作数据。当前采集的是固定初态的一条示范，适合读取检查和小样本拟合，不足以证明泛化能力。标准LeRobot加载与ACT短程训练已通过第一阶段验收；模型控制仿真使用下方独立闭环入口。

抓取采用夹持区域内粒子的辅助绑定，机器人为运动学控制。专家与动作接口共用抓取规则。需要批量数据前，应先验证重复成功率，再加入初态变化。

其他底层调试命令见 [动作重放说明](docs/ACTION_REPLAY.md)和 [实时桥接说明](docs/NEWTON_ISAAC_LIVE.md)，日常采集只使用上面的脚本。

## 主要文件

- `run_collect.sh`：日常采集唯一入口。
- `simulation/scene.yaml`：固定工况、毛巾、相机及物理参数。
- `scripts/isaac/live_bridge.py`：实时窗口与同步采集。
- `scripts/isaac/collect_dataset.py`：数据集写入与视频帧数检查。
- `dual_arm/`：机器人模型与网格。

重装采集依赖时使用：

```bash
$HOME/miniforge3/envs/dexonomy-isaac/bin/python -m pip install pyarrow==19.0.1 --no-deps --target .runtime/collector_deps
```

## 学习资料与教学实验

所有教学脚本和Python代码已集中到 `learning/`，按课程顺序阅读 [学习回顾手册](learning/STUDY_GUIDE.md)。根目录保留实时仿真采集入口 `./run_collect.sh`。

```bash
# 在工程根目录运行；教学实验不控制机器人
./learning/scripts/learn_data.sh
./learning/scripts/learn_batch.sh
```

`learning/scripts/` 是13课的Shell入口，`learning/python/` 是教学实现与共享模块。旧的根目录 `learn_*.sh` 已移动，请使用新路径。教学图表、页面和模型仍写入采集目录旁的 `learning_*` 输出目录，依赖仍在 `.runtime/`。

## 标准ACT第一阶段

运行 `./run_act_stage1.sh`，依次完成独立环境检查、官方数据加载、标准ACT短程训练、保存及重载推理验收。默认只使用当前单条固定工况示范，尚不控制仿真。详细命令、版本、数据副本与输出说明见 [ACT第一阶段](training/act/README.md)。

第一阶段已实测通过：标准ACT约5158万参数，200次更新后固定训练片段z=0 L1从1.0197降至0.2552，模型和处理器重载预测一致；尚不代表仿真任务成功。详细设备与检查结果见 [验收记录](training/act/VALIDATION.md)。

## ACT独立控制实时仿真

```bash
./run_act_live.sh
```

优先加载 `training/act/deployment.json` 中经过闭环验收的模型和执行参数，并校验权重哈希；没有部署记录时才选择最近一次第一阶段验收通过的模型。默认每次执行30步预测动作，再用当前RGB和实际关节状态重新预测。打开Isaac窗口后专家关闭，窗口可暂停、查看三路相机和退出；运行结束保留窗口。物理与渲染同步，实际运行可能慢于仿真时钟。指定 `--checkpoint` 可测试其他权重，`--action-steps` 和 `--duration` 可显式覆盖执行参数。

结果保存到 `outputs/act_rollout/<本次运行>/`，查看 `evaluation.json` 的 `task_success` 判断是否完成对折；`result.json` 区分完成、提前退出和异常。任务失败也会保存视频、预测动作与实际状态供诊断。完整用法、控制信号、评价标准和输出格式见 [ACT闭环说明](training/act/ROLLOUT.md)。

本机首次完整评价：900次模型控制、专家调用0次，三路视频各901帧；当前200次更新的权重未完成抓取与对折，平均配对误差165mm。闭环已运行，策略能力尚未通过，详情见 [闭环实测记录](training/act/ROLLOUT_VALIDATION.md)。
