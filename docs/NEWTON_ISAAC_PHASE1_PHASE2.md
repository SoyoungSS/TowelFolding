# Newton + Isaac Sim：阶段 1、阶段 2 实操笔记

日期：2026-11-06。目标：Newton 完成物理仿真，Isaac Sim 回放同一段物理状态并输出相机视频。本次不包含在线进程通信、ACT 训练或策略闭环。

## 一、先理解这次的系统分工

```text
phase1_record_newton.py（towel-newton 环境）
    → NewtonTowelEnv：reset / expert_action / step / get_state
    → Newton：运动学双臂 + VBD 毛巾 + 碰撞 + 辅助抓取
    → trajectory.npz + metadata.json + result.json

phase2_replay_isaac.py（dexonomy-isaac 环境）
    → 读同一份 trajectory.npz
    → 更新机器人各链接世界变换、毛巾顶点
    → Isaac Sim RTX / Replicator 同步捕获
    → 三路相机视频 + 总览 + 四宫格 + 帧映射 + 校验结果
```

两个程序先后运行，使用文件交换状态。这样可以分别验证物理、坐标和渲染。Isaac 中的机器人和毛巾没有刚体、碰撞或 articulation 物理属性，不会被第二个物理引擎重新求解。

当前机器人仍然采用原工程的限速运动学控制；接触只影响毛巾。自动专家知道网格角点，抓取使用区域检查后的节点绑定。这里只是在既有物理假设下跑通联合路线。

## 二、新增的代码分别做什么

| 文件 | 职责 | 建议阅读顺序 |
|---|---|---|
| `simulation/newton_env.py` | 将原 Newton 场景封装成明确的控制/状态接口 | 1 |
| `scripts/isaac/phase1_record_newton.py` | 检查 reset/动作，运行一次专家对折并记录 | 2 |
| `scripts/isaac/phase2_replay_isaac.py` | 读取轨迹、创建显示模型、同步捕获并编码视频 | 3 |
| `scripts/isaac/check_bridge_outputs.py` | CPU 验收：状态映射、同步时间、视频完整解码与可见性 | 4 |

原有 `simulation/towel_scene.py` 和 URDF 未修改。此前 `step01_import_robot.py`、`step02_robot_scene.py` 是独立 Isaac/PhysX 基础实验；本次回放不依赖它们生成的 USD。

### 为什么这次没有直接复用导入的机器人 USD？

URDF importer 可能重新对齐物体或关节坐标轴。将 Newton 的原始链接位姿直接写进这些调整后的链接，容易让视觉几何错位。

本次回放直接读取原 URDF 的 visual 网格、颜色、origin 和 scale。每个链接创建为同一级 Xform，再写入 Newton 世界位姿；visual 相对链接的原始变换保留。这样每个显示链接与 Newton 链接一一对应，能直接检查世界位置与旋转。

关节状态依然完整记录，但回放机器人时使用链接世界位姿，以免两个系统各自重新计算正运动学产生差异。

## 三、阶段 1：封装 Newton，并记录完整对折

### 1. reset 在做什么？

```python
state = env.reset(seed=0, scene_config=None)
```

- 从原 `scene.yaml` 创建或重置场景。
- 恢复机器人、布料、抓取标记、仿真时钟。
- 新建专家控制器，去掉上一轮的阶段及 IK 历史。
- 帧号归零、episode 编号递增。
- 返回一个完整状态快照。

`scene_config` 若提供，必须是完整配置。seed 目前用于记录与建立随机数生成器，本次没有隐式随机化。reset 检查保证初始状态恢复；不承诺 GPU 柔性仿真的完整轨迹逐位一致。实际多次运行均成功，但最终毫米级误差会变化，应以对应结果文件为准。

### 2. action 的 16 个数字是什么？

```text
[左臂 J1…J7, 右臂 J1…J7, 左夹爪开度, 右夹爪开度]
```

前 14 项单位为 rad，最后两项是归一化开度：0 关闭，1 到 URDF 开启极限。适配器将左夹爪的负角度范围与右夹爪的正角度范围统一转换。

```python
next_state = env.step(action)
```

step 只推进一个原场景帧，即 1/60 秒；内部仍然包含原有 10 个物理子步。目标受关节限位限制，实际关节受到速度限制，因此目标与实际状态并不相同。NaN、Inf 或维度错误的动作会被拒绝。

step 不会偷偷更新专家，也不渲染相机。专家动作通过单独的 `expert_action()` 获得。

### 3. 为什么专家接口需要单独说明？

`expert_action()` 除了计算关节目标，还会在阶段边界执行原来的绑定/释放操作。节点绑定目前仍由知道角点的专家触发，**仅靠 16 维动作重放并不能独立重建这些抓取事件**。

这不影响本阶段“完整物理状态回放”，因为毛巾顶点和抓取拥有者都已记录。后续采集 ACT 数据前，需要把抓取改成专家与策略共用的几何触发控制器；否则会出现专家能抓、策略无法抓的问题。

### 4. get_state 返回什么？

| 字段 | 单帧形状/含义 |
|---|---|
| `joint_state` | `(16,)`，统一顺序的实际角度和开度 |
| `body_pose` | `(26,7)`，含 world 的 Newton body 世界 xyz + xyzw 四元数 |
| `cloth_points` | `(441,3)`，毛巾世界顶点坐标 |
| `cloth_velocity` | `(441,3)`，顶点速度 |
| `grasp_owner` | `(441,)`，-1 表示自由，其余为绑定的 body 索引 |
| `sim_time` / `frame_id` | Newton 仿真秒数/物理场景帧号 |
| `episode_id` / `phase` | 当前 episode/专家阶段 |

返回的是 CPU 数组副本，便于文件记录。本阶段优先可读性，不优化 GPU 零拷贝。

### 5. 实际执行命令

在工程根目录：

```bash
mkdir -p outputs/bridge
/home/shendongran/miniforge3/envs/towel-newton/bin/python \
  scripts/isaac/phase1_record_newton.py \
  > outputs/bridge/phase1.log 2>&1
```

也可以使用新增入口：

```bash
make bridge-record
```

Makefile 的 `CONDA` 默认是 `$HOME/miniforge3/bin/conda`，可覆盖；Newton 环境可通过 `SIM_ENV` 覆盖。

程序先检查：无效动作拒绝、有效动作推进帧号和关节、reset 恢复初态和释放抓取。随后专家逐步完成对折，记录所有 60 Hz 状态，不运行 Newton 相机渲染。

### 6. 实际结果

最终保留的 `episode_000/result.json`：

| 项目 | 结果 |
|---|---|
| 任务结束阶段 | `done` |
| 场景步数 | 1625 |
| 完整状态数 | 1626 |
| 仿真时长 | 27.0833 秒 |
| 平均成对顶点平面对齐误差 | 10.87 mm |
| 最大成对顶点平面对齐误差 | 21.55 mm |
| X 方向投影宽度比 | 0.5653 |
| 抓取状态 | 完成后全部释放 |

通过原场景的有限数值、关节范围、毛巾质量、桌面接触和完整折叠检查。

这里的误差是原专家基于网格配对的评价，并非相机视觉估计误差；宽度比也不是独立的折叠成功判据。

输出在 `outputs/bridge/episode_000/`：

- `trajectory.npz`：约 8 MB 的完整状态与命令。
- `metadata.json`：配置、拓扑、body 名称、关节顺序、软件版本、URDF 校验值和限制。
- `result.json`：折叠结果。

### 7. 理解 T+1 个状态和 T 个动作

```text
s0 --a0--> s1 --a1--> s2 … --a(T-1)--> sT
```

所以 `joint_state[t]` 和其他状态对应执行 `action[t]` 前的状态，最后一个状态没有后续动作。专家在选动作时产生的绑定/释放事件，反映在推进后的抓取记录中；本阶段文件还不是可直接训练 ACT 的完整数据集。

可以用 Newton 环境查看文件：

```bash
/home/shendongran/miniforge3/envs/towel-newton/bin/python - <<'PY'
import numpy as np
d = np.load('outputs/bridge/episode_000/trajectory.npz', allow_pickle=False)
for key in d.files:
    print(key, d[key].shape)
print('第一帧实际状态:', d['joint_state'][0])
print('第一步目标动作:', d['action'][0])
print('末帧仿真时间:', d['sim_time'][-1])
PY
```

## 四、阶段 2：Isaac Sim 回放与三路相机

### 1. 创建显示场景

读取原 URDF visual，创建机器人显示链接；按同一配置创建桌面、支腿、支架和地面。毛巾使用 441 个顶点、800 个三角形的双面网格，避免翻折后背面消失。

毛巾顶点是世界坐标，显示 Mesh 保持单位变换。每次更新 points 时同步更新包围盒 extent，避免旧包围盒把抬起后的毛巾裁掉。

### 2. 每个回放帧在做什么？

```text
读取一个 Newton 状态索引
→ 写入 25 个 URDF 链接的世界位置和朝向
→ 写入 441 个毛巾顶点和包围盒
→ Replicator 同步完成渲染
→ 读取 RGB 与 ReferenceTime
→ 校验三路相机和总览时间一致、时间没有重复
→ 编码视频，并保存源帧索引
```

相机挂在原链接下，局部 position/target/up 沿用原工程。另加 overview，仅用于检查任务全局过程，不属于原定 ACT 的三路输入。

本阶段使用参考时间 annotator 判断捕获更新；ReferenceTime 属于渲染器时间，**Newton 的物理时间始终以轨迹中的 sim_time 为准**。二者不能混作同一个时间戳。

### 3. 执行命令

用户已在本次会话同意 NVIDIA EULA，因此以下启动命令携带接受变量：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
/home/shendongran/miniforge3/envs/dexonomy-isaac/bin/python -u \
  scripts/isaac/phase2_replay_isaac.py \
  > outputs/bridge/phase2.log 2>&1
```

或者：

```bash
OMNI_KIT_ACCEPT_EULA=YES make bridge-replay
```

第一次调试可以只渲染三帧，保存到独立目录：

```bash
OMNI_KIT_ACCEPT_EULA=YES make bridge-replay \
  ARGS='--max-frames 3 --output outputs/bridge/replay_smoke'
```

`ISAAC_ENV` 可覆盖现有环境名；两套环境不混装 Newton/Warp。Isaac 环境中的 ffmpeg 用于生成 H.264 视频，未额外安装软件。

### 4. 输出文件怎么读？

输出目录为 `outputs/bridge/replay_000/`：

| 文件 | 内容 |
|---|---|
| `head.mp4` | 头部 RGB，320×240 |
| `left_wrist.mp4` / `right_wrist.mp4` | 左右腕部 RGB，320×240 |
| `overview.mp4` | 辅助总览，320×240 |
| `multiview.mp4` | 四宫格，640×480，标注源仿真时间和阶段 |
| `checkpoint_*.png` | 各阶段开始和最终画面的四宫格 |
| `{camera}_{phase}.png` | 对应相机的关键帧 |
| `frames.json` | 每个视频帧对应的源索引、源时间、相机参考时间和世界位姿 |
| `calibration.json` | 分辨率、视场角、针孔内参和坐标约定 |
| `final_scene.usdc` | 最终静态显示场景，内嵌几何，可独立打开 |
| `result.json` | 完整回放是否通过，以及误差和画面更新统计 |

正常回放从 60 Hz 轨迹中每三帧取一帧，视频为 20 fps。另补末帧保留 `done` 状态，末尾采样间隔略短，精确时间看 `frames.json`。这不是均匀重采样后的训练数据。

### 5. 做了哪些校验？

- 显示对象没有 RigidBody、Collision、ArticulationRoot 属性。
- 每 50 个回放帧检查链接世界平移与旋转矩阵，以及相机挂载世界位置。
- 同时检查写入的毛巾顶点与源数组一致。
- 三路和总览相机的参考时间一致，每个输出帧都比前一帧更新。
- 图像维度正确、有内容变化，视频编码进程正常结束。

位置/旋转检查的门槛为 `1e-5`，实际结果见 `result.json`。这些检查证明坐标和数据传递一致，不代表毛巾材料参数已物理标定。

最终完整回放与独立验收结果：

| 项目 | 实测 |
|---|---|
| 输出视频帧数 | 每个视频 543 帧 |
| 视频帧率 | 20 fps |
| 物理轨迹时长 | 27.0833 秒 |
| 链接变换最大分量差 | `6.19e-6`，含平移分量与旋转矩阵元素，不能整体解释成米 |
| 相机位置最大差 | `5.50e-7 m` |
| 显示链接数 | 25（Newton 另有一个 world body） |
| 相机输出 | 四路均有 542 次相邻帧内容变化 |
| 视频格式与完整性 | 五个视频全部为 H.264，完整解码通过 |
| 头部画面毛巾颜色像素 | 开始 8083，结束 3112 |
| 最终 USD | 约 32 MB，重新打开无 composition error |

建议先打开 [四宫格视频](../outputs/bridge/replay_000/multiview.mp4)，再对照以下关键帧：

![翻折中，源仿真时间 15 秒](../outputs/bridge/replay_000/mid_fold.png)

![完成并释放](../outputs/bridge/replay_000/checkpoint_done.png)

`mid_fold.png` 从最终视频第 15 秒提取，可复现：

```bash
/home/shendongran/miniforge3/envs/dexonomy-isaac/bin/ffmpeg \
  -y -v error -ss 15 -i outputs/bridge/replay_000/multiview.mp4 \
  -frames:v 1 outputs/bridge/replay_000/mid_fold.png
```

完整回放之后执行独立验收：

```bash
make bridge-check
```

该步骤不需要 GPU。它检查完整状态与动作长度、源帧映射、抓取保持/释放、三路捕获同步，使用 ffprobe 统计视频帧数，再用 ffmpeg 完整解码全部五个视频。还检查头部画面在开始和结束时均能看到绿色毛巾；颜色阈值仅适用于本次固定材质的调试，不是正式任务成功评价。结果写入 `acceptance.json`。

## 五、这次排错中值得学习的地方

1. **网格先创建完整初态再渲染。** 只写拓扑、晚些时候才创建 points，会让 Hydra 首次读取到无效网格。
2. **NumPy 与 USD 类型要明确转换。** 某些 Gf 构造器不接受 numpy.float32；数组使用 `Vt.*Array.FromNumpy`，标量使用 Python float。
3. **相机首次需要预热。** 创建传感器后立即读 RGB，可能得到空数组。
4. **Camera 的缓存帧号不能直接当采集确认。** 本次改用 Replicator 显式捕获和独立 ReferenceTime annotator；不能只靠 sleep 或 get_rgba 返回了数组就判断是新帧。
5. **数学上的视角相同不保证极端参数量级渲染相同。** 过小的 USD aperture/focal 参数出现异常放大；改成常规 USD 相机量级，保持 aperture/focal 比值后，65° 视角恢复。这里使用理想针孔投影，未标定真实镜头。
6. **退出码不替代结果校验。** Kit 关闭行为可能掩盖异常退出；脚本在异常时写 `status: failed` 并打印 traceback，验收以结果文件和视频检查为准。
7. **物理引擎与显示引擎对四元数的容差不同。** 首轮全回放发现 Newton float32 四元数轻微偏离单位长度，转换后的矩阵最大偏差为约 `1.15e-5`，超过 `1e-5` 门槛。写入 USD 前显式归一化后重新验收，避免把轻微数值缩放带进显示层。首次失败日志保存在 `outputs/bridge/phase2_numeric_failure.log`。
8. **分清相机缓存和直接 annotator。** 回放使用同步 RGB/ReferenceTime annotator，不依赖 Camera 的旧版仿真时间插值回调；暂停该回调后仍可读取同步捕获的 RGB，避免让没有 PhysX 状态历史的显示场景做错误时间插值。

## 六、阶段边界与下一步

本阶段已经能显示 Newton 算出的机器人与柔性毛巾，并生成对应视角的视频。它仍是离线回放。

下一阶段才建立两个进程的实时锁步接口，以及统一的策略抓取控制器。之后再做多工况采集、LeRobot 转换、ACT 训练与在线闭环。不能把历史轨迹的视频回放称作 ACT 已经控制机器人。

学习时建议先看四宫格视频，再查看 frames.json 中某一帧的 source_index，最后在 trajectory.npz 中读取同一索引的 joint_state、body_pose 和 cloth_points，理解“物理状态如何变成图像”。
