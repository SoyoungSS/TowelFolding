# Isaac Sim → 示范数据 → ACT：叠毛巾学习实验记录

记录日期：2026-11-06。目标仅为仿真学习，不涉及真机。

最新进展：Newton + Isaac 离线联合路线的阶段 1、2 已实施，详见 [阶段 1、2 实操笔记](NEWTON_ISAAC_PHASE1_PHASE2.md)。下文较早的“待执行”指原先独立 Isaac 毛巾路线；当前优先保留 Newton VBD 并在 Isaac 中渲染。

2026-11-07 更新：新增两个进程的实时锁步运行与 Isaac UI，见 [同步运行使用说明](NEWTON_ISAAC_LIVE.md)。ACT 数据集与训练仍未实施。

这份文档区分“已运行的实验”和“后续实验设计”。有源码不等于验证通过，导入成功不等于已经能折叠，训练 loss 下降也不等于策略成功。每个阶段先保留可复现命令和结果，再进入下一阶段。

## 1. 全流程在做什么

```text
URDF + 网格 → USD 机器人资产 → 机器人控制、桌面和相机
→ 毛巾物理与抓取 → 专家完成一次对折 → 记录同步示范
→ LeRobot 数据集 → ACT 训练 → ACT 控制同一个仿真 → 独立场景评价
```

URDF 描述机器人结构，不包含毛巾、任务逻辑、真正的相机传感器和训练数据。USD 保存 Isaac Sim 能加载的场景资产。ACT 从图像和机器人状态预测未来动作；它首先需要成功示范，而不是靠启动仿真自己学会折叠。

## 2. 实验 00：检查运行条件（已执行）

### 目的

找到可用的解释器、Isaac Sim 和 GPU，避免重装已有环境或把 Isaac 的依赖装进 Newton 环境。

### 实际检查

```bash
nvidia-smi
/home/shendongran/miniforge3/bin/conda env list
/home/shendongran/miniforge3/envs/dexonomy-isaac/bin/python -m pip show isaacsim isaacsim-core
df -h .
```

本次发现：

| 项目 | 实际结果 |
|---|---|
| Isaac Sim | 已有 `dexonomy-isaac` 环境，版本 5.1.0.0 |
| Isaac Python | 3.11，沿用已有 5.1 环境，不套用其他版本 Python 要求 |
| GPU | RTX 5060 Laptop GPU，8151 MiB 显存 |
| 驱动 | 580.178.04 |
| Newton | 独立环境 `towel-newton`，未修改其依赖 |
| 工程所在磁盘 | 检查时约剩余 31 GB |

普通沙箱内 `nvidia-smi` 报错，扩大设备访问后成功。这是本次执行环境访问差异，不能直接判定主机驱动坏了。原工程环境文档中提到 RTX 5080 是交付机记录，不是本机实测配置。

首次启动遇到 Omniverse EULA；用户已明确授权接受，随后使用 `OMNI_KIT_ACCEPT_EULA=YES` 启动。该变量代表接受协议，不应在未同意时机械复制使用。

## 3. 实验 01：把原始 URDF 导入为 USD（已通过）

### 目的

验证机器人资产完整，并把原始机器人描述转换成 Isaac Sim 能加载的资产。这个实验不包含动力学跟踪和毛巾。

### 脚本

`scripts/isaac/step01_import_robot.py`

脚本先做不依赖 GPU 的检查：解析 XML，把 `package://dual_arm/` 网格路径改为绝对路径，检查文件存在，检查可动关节集合。结果写入 `outputs/isaac/step01/resolved.urdf`，原始 URDF 保持原样。

接着创建 `SimulationApp`，启用 URDF importer，固定基座、不合并固定关节、导入惯量、关闭机器人自碰撞，调用 `URDFParseAndImportFile` 导出 USD。保留固定链接便于后续挂相机；自碰撞设置属于当前简化，后续任务仍需验证合理的碰撞配置。

### 复现命令

在工程根目录执行：

```bash
mkdir -p outputs/isaac/step01
OMNI_KIT_ACCEPT_EULA=YES \
/home/shendongran/miniforge3/envs/dexonomy-isaac/bin/python \
  scripts/isaac/step01_import_robot.py \
  > outputs/isaac/step01/run.log 2>&1
```

### 实际结果

- 50 处网格引用全部找到。
- 16 个可动关节：双臂各 7 个，夹爪各 1 个。
- GPU 被 Isaac/Warp 识别，应用正常启动并退出。
- `dual_arm.usd` 及 `configuration/` 下子资产成功生成；主 USD 很小，因为真实几何通过子文件引用，不能只复制主文件。
- `import_result.json` 记录 `status: imported`。

Importer 提示若干固定关节轴不与主轴对齐并自动调整。这些提示不等于导入失败，后续需要用姿态和相机画面检查变换。

### 一个很容易犯的错误：关节顺序

原始 URDF 先右臂后左臂；仿真实际 DOF 顺序可能左右交错。未来策略统一采用：

```text
左臂 joint_1…7，右臂 joint_1…7，左夹爪，右夹爪
```

始终按名字转换索引，不能直接把长度相同的数组互相发送。左夹爪角范围为负、右夹爪为正；未来数据集可把两者都转成 0～1 开度，但要提供可逆转换。

## 4. 实验 02：机器人控制、桌面与三路相机（已执行，结果见下节）

### 目的

验证“能加载”之后“能推进物理、能发送关节目标、能拿到 RGB”。这仍然不是叠毛巾任务。

### 脚本与命令

`scripts/isaac/step02_robot_scene.py`

```bash
OMNI_KIT_ACCEPT_EULA=YES \
/home/shendongran/miniforge3/envs/dexonomy-isaac/bin/python \
  scripts/isaac/step02_robot_scene.py \
  > outputs/isaac/step02.log 2>&1
```

脚本中的操作顺序：

1. 创建米制场景，物理步长设为 1/120 秒。
2. 引用刚导出的机器人，沿用原工程 1 m 安装高度。
3. 根据 `simulation/scene.yaml` 创建桌子与初始关节姿态。
4. 在头部及左右腕部的父链接下创建真正的 Camera。
5. 用名称找到各 DOF，发送关节位置目标，并给左臂第一个关节一个 0.05 rad 的小幅变化。
6. 运行 240 个物理步，约 2 秒仿真时间，每 6 步渲染一次。
7. 检查关节数值有限、计算命令跟踪误差、保存三路 RGB 和场景。

相机采用 USD/OpenGL 的 -Z 前向、+Y 上向。配置中的 position/target/up 是父链接局部坐标，需要先构造旋转矩阵，再转成 Camera 使用的 wxyz 四元数。

### 实际发现与修改

第一次使用 Kp=400、Kd=40，最大关节目标偏差达到约 0.543 rad（31°）。虽然没有崩溃、相机也有图像，这个偏差不能作为合格控制器。原始结果保存在 `initial_low_gain_result.json`，其中早期 `status: passed` 只反映冒烟检查；判定逻辑已修改为还必须满足最大误差 <0.1 rad。

随后把 Kp/Kd 改为 10000/1000 复测。这个调整用于本次仿真学习基线，不代表真实电机参数，也不能直接推广到其他机器人。

复测结果：240 步执行完成，最大跟踪误差 **0.018589 rad，约 1.07°**，满足本次冒烟门槛 0.1 rad。这个门槛仅用于检查明显异常，并不是抓角精度指标。当前只给一个关节施加小幅变化，其余关节保持初始目标；没有完成全工作空间或夹爪开合验收。

图像核对还发现一个 reset 问题：直接在 USD 根节点写 1 m 平移，随后创建 articulation/reset 时高度丢失，头部相机落到约 0.2 m。改用 articulation 的 position、default state 和 reset 后 world pose 设置安装位置，重新运行成功。导出场景中头部相机世界高度为约 1.2 m，左右腕相机约为 0.797/0.791 m；桌面恢复到相机视野。这说明检查真实图像和世界坐标，比只检查退出码更有用。

输出文件：

| 路径 | 用途 |
|---|---|
| `outputs/isaac/step02/result.json` | 最新运行结果、关节顺序、误差、图像统计 |
| `outputs/isaac/step02/head.png` | 头部相机 |
| `outputs/isaac/step02/left_wrist.png` | 左腕相机 |
| `outputs/isaac/step02/right_wrist.png` | 右腕相机 |
| `outputs/isaac/step02/scene.usda` | 带桌面和相机的场景；依赖步骤 01 资产 |
| `outputs/isaac/step02.log` | 实际运行日志 |

图像有非零变化只能证明传感器产生图像，仍需人工确认是否看得到毛巾操作区域。当前尚无毛巾，所以不能证明三路视角适合抓角。

本次实际保存的头部画面（在支持本地图片的 Markdown 查看器中打开）：

![头部相机实际输出](../outputs/isaac/step02/head.png)

当前完成到机器人与传感器基础验证。下面实验 03～06 是后续执行顺序，尚未采集折叠示范、训练 ACT 或实现 Isaac 毛巾。

## 5. 实验 03：毛巾物理与抓取（待执行）

这是下一项实验，不应把 Newton 的参数直接复制成同名 PhysX 参数。当前 Newton 使用 VBD 和节点绑定辅助；Isaac Sim 5.1 的布料实现、参数和抓取机制需要单独验证。

官方最新物理限制页将 Particle Cloth 列为计划弃用并建议不使用；这不等于本机 5.1 没有该功能。学习实验可以在固定版本中评估旧方案，但必须记录版本、限制和实际表现，不能未经测试就宣称已有可替代的表面可变形方案。

按下面次序进行，每一步保存视频或关键帧和数值：

1. 30×30 cm 布片落桌静置，检查尺寸、质量、伸长、抖动和穿透。
2. 单夹爪抓角抬起并释放，确认抓取可逆。
3. 双夹爪抓角，保持一段时间，检查约束冲突和拉伸。
4. 翻折形成双层，验证自碰撞与释放后形态。

首版允许辅助抓取：闭合且布料位于夹持区域内才绑定，张开即释放，不能瞬间把远处角点吸过来。专家和 ACT 执行必须使用同一规则。

## 6. 实验 04：专家与同步数据（待执行）

先迁移现有自动对折的阶段逻辑，完成一次“平铺 → 双臂抓两角 → 抬起 → 翻折 → 释放”。成功后再接 VR，暂时不增加随机揉皱、多次折叠等任务难度。

采集频率先统一为 20 Hz；仿真每 6 个物理步产生一条记录。每条记录的语义是：

```text
当前状态和图像 o_t → 专家目标命令 a_t → 保存(o_t,a_t) → 推进物理
```

不能把执行后的图像与执行前的动作随意配对。状态是实际角度，动作是命令目标，两者不是一个量。

第一版数据约定：

| 数据 | 形状/含义 |
|---|---|
| 三路 RGB | 每路 H×W×3，记录颜色顺序、尺寸和时间戳 |
| observation.state | 14 个实际关节角 + 2 个实际归一化夹爪开度 |
| action | 14 个目标关节角 + 2 个目标归一化夹爪开度 |
| 元信息 | episode、帧号、仿真时间、任务、随机种子、成功标记、配置 |
| 评价真值 | 布料顶点/角点，独立保存，不输入视觉策略 |

先录一条并回放，确认图像、状态、命令和夹爪方向正确，再批量采集。重复相同初态的同一条轨迹不会带来足够的泛化数据。

## 7. 实验 05：LeRobot 与 ACT（待执行）

使用独立训练环境，固定 LeRobot 版本/提交。先用其 Dataset API 把一条示范转成数据集并读取，验证字段、形状、归一化与动作时间对齐，再扩大到约 50～100 条成功示范作为起点。这个数量不是成功保证。

按完整 episode 和不同初始场景划分训练/验证集，不能随机拆相邻视频帧。以 RGB + 16 维状态预测 16 维动作，动作块先测试约 1～2 秒范围。

官方命令形式如下，仅作为后续入口，尚未在本工程执行；必须按实际固定版本核对参数：

```bash
lerobot-train \
  --dataset.repo_id=YOUR_ACCOUNT/dual_arm_towel_fold \
  --policy.type=act \
  --output_dir=outputs/train/act_towel_fold \
  --job_name=act_towel_fold \
  --policy.device=cuda
```

先对一个小数据子集检查能否拟合，再做正常训练。训练 loss 用于排错，最终依据仍然是闭环成功率。

## 8. 实验 06：策略闭环评价（待执行）

自定义 Isaac 执行器负责图像和状态预处理 → ACT → 动作反归一化 → 限位限速 → 原位置控制器。使用与采集一致的频率、关节顺序、图像顺序和抓取规则。每次 reset 清空动作队列与时间集成状态。

测试独立初始位置/旋转，记录抓角成功、掉落、穿透、对折边角误差、目标轮廓重合度和完成时间。面积减半不能单独代表折好，因为揉成团也可能面积变小。

## 9. 官方参考

- [Isaac Sim 5.1 Python 环境](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
- [URDF 导入与 USD 结构](https://docs.isaacsim.omniverse.nvidia.com/latest/openusd_tuning_tutorials/tutorial_01_asset_structure.html)
- [物理功能限制，包括 Particle Cloth](https://docs.isaacsim.omniverse.nvidia.com/latest/physics/physics_resources.html)
- [LeRobot ACT](https://huggingface.co/docs/lerobot/act)

`latest` 网页会变化；本次实际执行依据本机 Isaac Sim 5.1 的 importer 示例与 API 源码。后续实验应继续追加真实结果，不把计划改写成已完成。

## 10. Newton + Isaac 联合路线（2026-11-06 调研；阶段 1、2 已实施）

用户提出保留 Newton 毛巾物理，使用 Isaac 场景和相机采集 ACT 示范。这个方案可行，而且新版本已有官方集成；不能将本机 Isaac Sim 5.1 的现状推广到新版本。

- [Isaac Sim Newton backend](https://docs.isaacsim.omniverse.nvidia.com/latest/physics/newton_physics.html)：支持将 Newton 设为物理后端，并同步 Fabric 用于渲染；该模式替换 PhysX，同一时刻只激活一个后端。后端启用不自动等于任务所需 VBD 布料、抓取和耦合已经可用。
- [Isaac Lab Newton](https://isaac-sim.github.io/IsaacLab/main/source/concepts/physics-backends/newton.html)：提供 VBD 和耦合求解配置。
- [官方 tablecloth 示例](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/demos.html)：Newton VBD 布料/接触抓取，脚本 IK 机器人，并可选择 Kit 视口。它是验证联合路线的参考，不是本双臂任务现成实现，也不能仅凭视口展示推断相机训练数据接口已经打通。
- [版本兼容表](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/compatibility.html)：新版本应使用匹配的 Isaac Sim / Isaac Lab / Newton，不能把现有 Newton 1.7 随意装入 Isaac 环境。

本机检查：现有 Isaac Sim 5.1 环境扩展目录中未发现 `isaacsim.physics.newton`；Newton 在独立 Python 3.12 环境，版本 1.7.0 / Warp 1.18.0。前文步骤 02 使用 PhysX，不代表 Newton 集成已完成。

学习原型可先采用两进程锁步桥接：Newton 作为唯一物理状态来源，保留当前机器人运动学、VBD、辅助抓取与专家逻辑；Isaac Sim 5.1 只镜像机器人链接世界变换和毛巾网格顶点，用 RTX 相机渲染。显示副本不参与 PhysX 求解，避免两个引擎各自推进同一个物体。该方案不含毛巾对机器人的反作用力；足以验证当前简化条件下的视觉模仿学习。

为避免两个独立 Python/Warp 环境混装，首版通过共享内存或本地 IPC 交换数组。每个状态包携带 episode_id、frame_id、sim_time、实际关节、目标关节、链接变换、毛巾顶点；拓扑只需初始化传一次。接收方完成渲染和相机帧确认后再推进，不能用墙钟 sleep 假定同步。

最终记录必须满足：同一状态 s_t 的图像 + 实际关节构成 o_t，专家在该状态选择的命令构成 a_t，随后才执行 a_t。不能把更新 USD 后尚未完成的旧相机帧记录成新观测。首版先串行完成轨迹采集、RTX 重渲染、离线训练；闭环评价仍需实时锁步桥接，单纯播放历史轨迹不算策略控制。

后续验证顺序：静态毛巾顶点镜像 → 动态落桌与帧号同步 → 当前专家对折重渲染 → 小规模多初态示范 → 数据格式与 ACT → ACT 在线闭环。若重点转向动力学耦合，则另建匹配版本环境验证官方 VBD/Kit 示例后再迁移本机器人。
