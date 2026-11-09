# Newton + Isaac Sim：同步仿真与 UI 操作

日期：2026-11-07。本阶段将文件回放改为两个进程的锁步运行。Newton 仍是唯一物理状态来源，Isaac Sim 显示当前状态并同步获取 RGB。

## 1. 一条命令启动

在主机图形桌面的终端执行：

```bash
cd /home/shendongran/code/newton_towel_sim
OMNI_KIT_ACCEPT_EULA=YES make bridge-live
```

本会话已获得用户接受 NVIDIA EULA 的授权。其他使用者应先阅读并同意协议，再设置接受变量。

不需要先执行 `bridge-record` 或 `bridge-replay`，也不需要手动激活两个 Conda 环境。这次不会读取以前的 `trajectory.npz`。

启动器在 `dexonomy-isaac` 环境运行，自动拉起 `towel-newton` 的 Python 子进程。窗口标题为 **Isaac Sim Python 5.1.0**，默认显示总览相机并自动开始一次折叠。首次启动、模型创建及 RTX 预热需要等待一段时间。

如果环境路径不同：

```bash
OMNI_KIT_ACCEPT_EULA=YES make bridge-live \
  ISAAC_ENV=你的Isaac环境名 \
  NEWTON_PYTHON=/你的Newton环境/bin/python
```

也可以直接调用脚本：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
/home/shendongran/miniforge3/envs/dexonomy-isaac/bin/python -u \
  scripts/isaac/live_bridge.py \
  --newton-python /home/shendongran/miniforge3/envs/towel-newton/bin/python
```

## 2. 窗口里的按钮

请使用新增的 **Newton Live Controls** 面板控制物理任务。编辑器原有的 Timeline 播放按钮不是 Newton 专家的控制开关。

| 按钮 | 操作 |
|---|---|
| Play / Pause | 开始或暂停后续 Newton 推进；已经提交的一小步先完成并显示 |
| Step | 暂停时推进一个显示周期，默认包含 3 个 Newton 场景帧，即 0.05 秒仿真 |
| Reset + Fold | 同时重置机器人、毛巾、专家、物理时间和 episode 编号，然后重新折叠 |
| Save views | 保存当前头部、左右腕部及总览 PNG |
| Show / hide cameras | 显示或隐藏三路 RGB 预览窗口 |
| Exit both processes | 关闭 Isaac Sim，并通知/清理 Newton 子进程 |

折叠完成后自动暂停，窗口保持打开。要重新观看，点击 **Reset + Fold**。也可以关闭主窗口，或在启动终端按 Ctrl+C，启动器会进入清理流程。

想先观察初始场景再手动开始：

```bash
OMNI_KIT_ACCEPT_EULA=YES make bridge-live ARGS='--paused'
```

想完成一次后自动退出：

```bash
OMNI_KIT_ACCEPT_EULA=YES make bridge-live ARGS='--exit-on-done'
```

## 3. 同步具体如何工作

```text
Newton 子进程                    Isaac Sim 主进程
reset，返回 metadata + state ──→ 创建显示网格和相机
                                更新状态，完成同步 RGB 捕获
                            ←── advance，带当前 episode/frame 确认
专家更新 + 推进 3 个场景帧
返回新的 state ────────────────→ 更新网格与相机，再次捕获
等待下一条命令                  更新 UI / 等待用户操作
```

### 时间与帧编号

- Newton 原场景为 60 Hz，每个场景帧内部保留 10 个物理子步。
- 默认同步捕获周期为 20 Hz，即每次请求推进 3 个场景帧。
- 发送下一次 advance 请求前，当前状态必须已经显示并被相机捕获。
- 命令包含递增 request_id，以及确认过的 episode_id/frame_id。
- Newton 拒绝错误或重复的确认，避免偷偷多走一步。
- reset 后 episode 编号加一、frame_id 与物理时间归零；渲染器时间保持独立。

“20 Hz”描述仿真时间的采样间隔，不保证每秒墙钟完成 20 次捕获。运行速度由 Newton、RTX、UI 和数据拷贝共同决定。即使慢于实时，状态对应关系仍保持正确。

### 为什么 UI 不会被等待网络锁住？

本地请求/响应在一个后台线程中处理，每次最多一个未完成请求。主线程继续更新 Kit UI；收到响应后才更新显示并获取图像。按钮在主线程设置操作标记，不能从后台线程直接操作 USD 或 UI。

### 传输了什么？

使用临时目录下的 Unix domain socket，只在本机通信，没有暴露 TCP 端口。消息采用 4 字节长度前缀 + JSON，数组先以列表传输；禁止 pickle，限制单条消息大小。当前网格只有 441 个顶点，这个实现足够用于验证流程，后续需要时再改共享内存。

每个状态包含实际关节、链接世界位姿、毛巾顶点/速度、抓取标记、物理时间、帧号和阶段。元信息、拓扑及配置只在启动时传输一次。

当前专家仍在 Newton 中运行；Isaac 完成捕获后发出的 advance 是“当前帧已消费”的确认。后续 ACT 可以在该位置根据图像和实际状态选动作，但本阶段尚未实现 ACT 或把图像传回 Newton 做策略推理。

## 4. 代码入口

| 文件 | 职责 |
|---|---|
| `simulation/bridge_transport.py` | 本地消息分帧、大小限制和断连检查 |
| `scripts/isaac/live_newton_worker.py` | 持有唯一物理状态；执行 advance/reset/quit |
| `simulation/isaac/visual_scene.py` | 创建纯显示场景、写链接/毛巾、同步相机捕获和坐标核验 |
| `scripts/isaac/live_bridge.py` | 自动启动进程、UI、锁步调度、结果与退出清理 |
| `tests/test_live_bridge.py` | 分包读取、超长消息拒绝、锁步/reset/quit、错误确认不推进物理 |

为了保持阶段 2 离线回放稳定，本次未修改其运行方式。实时显示模块以已验证的离线场景构造代码为基础，额外将链接 orient 写成双精度 USD quaternion；这修复了长序列中 float32 四元数转换出现的矩阵误差。

UI 联调还验证了 `ByteImageProvider.set_bytes_data` 的输入要求：本机版本需要像素整数列表，直接传 bytes 会报类型转换错误。预览使用 RGBA 列表，保存图像仍使用 RGB 数组。窗口截图通过 Kit 自身的 renderer capture 获取，避免依赖 Wayland/X11 桌面截图权限。

## 5. 输出与观察运行状态

默认每次运行创建新的 `outputs/bridge/live_日期_时间/`。也可指定目录：

```bash
OMNI_KIT_ACCEPT_EULA=YES make bridge-live \
  ARGS='--output outputs/bridge/my_live_run'
```

| 文件 | 内容 |
|---|---|
| `newton.log` | 物理子进程日志与专家阶段 |
| `metadata.json` | 配置、拓扑、关节顺序与 Newton 依赖版本 |
| `frames.jsonl` | 每次捕获对应的物理帧、episode、时间、阶段与相机参考时间 |
| `live_status.json` | 最近一次显示状态，以及两个进程 PID |
| `ep*_相机_阶段.png` | 阶段变化时的相机关键帧 |
| `snapshot_*.png` | 点击 Save views 保存的画面 |
| `ui_started.png` / `ui_completed.png` | 原生窗口开始/完成截图（由 renderer capture 生成） |
| `completed_episode.json` | 一次完整折叠结束即写入；窗口尚未关闭也能查看 |
| `result.json` | 程序退出时写入，包含子进程是否已停止及异常信息 |

当前阶段不默认写完整视频或完整 ACT 数据集，以免把实时显示验证与训练数据格式混在一起。`frames.jsonl` 的相机参考时间用于确认同步，物理时间仍以 Newton sim_time 为准。

## 6. 验证与排错命令

协议与状态机单元测试，不需要 GPU，但需要本地 Unix socket 权限：

```bash
/home/shendongran/miniforge3/envs/towel-newton/bin/python tests/test_live_bridge.py
```

无窗口的真实双进程 GPU 集成测试，在第 8 次捕获后自动 reset，总共捕获 20 次：

```bash
OMNI_KIT_ACCEPT_EULA=YES make bridge-live \
  ARGS='--headless --max-updates 20 --reset-after 8 --output outputs/bridge/live_tests/reset'
```

完整无窗口同步折叠，完成后退出：

```bash
OMNI_KIT_ACCEPT_EULA=YES make bridge-live ARGS='--headless'
```

已执行的集成 reset 测试：20 次捕获、1 次跨进程 reset，通过帧号和相机时间检查，正常退出且 Newton 子进程已停止。四项协议/状态机测试通过。

2026-11-07 已完成的原生 UI 全流程验证：

| 项目 | 结果 |
|---|---|
| 实际运行目录 | `outputs/bridge/live_ui_final/` |
| 结束状态 | `done`，UI 保持打开并自动暂停 |
| Newton 场景帧 | 1625 |
| 同步捕获次数 | 543 |
| 仿真时间 | 27.0833 秒 |
| 平均/最大配对平面误差 | 7.37 / 11.28 mm |
| 链接变换最大分量差 | `7.88e-10` |
| 相机位置最大差 | `2.98e-9 m` |
| 帧核验 | 543 条物理帧递增记录，四路捕获参考时间逐条一致 |

本次窗口完成截图：

![原生 Isaac UI 已完成同步对折](../outputs/bridge/live_ui_final/ui_completed.png)

完整结果见该目录的 `completed_episode.json`。因为窗口留给用户操作，最终退出结果 `result.json` 要等用户关闭程序后才产生。

常见情况：

- 看不到窗口：确认没有加 `--headless`，并在主机图形桌面运行；无显示环境只能使用无窗口测试。
- 暂停后稍微动了一下：已经提交的 3 帧物理计算允许完成，之后停止请求；不会自由运行。
- `LIVE FINISHED` 报错：查看该次 `result.json` 和 `newton.log`，不要仅凭 Kit 的进程退出码判断成功。
- 运行较慢：锁步优先保证正确性。可以暂不打开三路预览；这不会改变相机捕获流程。
- 画面完成后不继续运动：这是设计行为，点击 Reset + Fold 开始下一轮。

仍然沿用当前仿真假设：运动学机械臂、VBD 毛巾和专家辅助抓取，不含布料对机械臂的动力学反作用。正式 ACT 采集前还需要实现专家/策略共用的抓取触发逻辑。
