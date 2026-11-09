# 双臂叠毛巾：实时仿真与数据采集

当前只运行一种工况：固定位置的30×30 cm平铺毛巾，由双臂轨迹专家完成一次对折。Newton负责物理，Isaac Sim打开实时窗口并同步采集头部、左右腕三路RGB、实际关节状态和目标动作，输出LeRobot v3结构的数据。

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

实测已完成一次带实时窗口的完整采集：815条记录、30 Hz、约27.17秒仿真时间，三路视频均逐帧解码检查通过，最终平均对齐误差16.2 mm。验收数据位于 `outputs/collect_validation_02/dataset`。

## 用于ACT的范围

这一步生成视觉模仿学习所需的RGB、状态和动作数据。当前只打通固定初态的一条示范，不自动训练ACT；单条数据适合读取检查和小样本拟合，不足以证明泛化能力。LeRobot实际加载及训练兼容性仍需在训练环境中验证。

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
