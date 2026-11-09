# 使用说明

所有命令在交付目录执行。关闭桌面窗口或按 Ctrl+C 退出。

## 桌面与自动折叠

```bash
make sim
make sim-fold
make sim ARGS="--viewer null --num-frames 180 --test"
make sim-fold ARGS="--viewer null --test"
```

`make sim-fold` 默认运行上限为 2400 帧，图形窗口的运行行为由 Newton 查看器控制。
自动对折约需 27 秒仿真时间；无窗口 180 帧只验证落稳，不能验证完整折叠。

| 控件 | 行为 |
|---|---|
| Left arm / Right arm | 调整对应手臂的七个关节目标 |
| Left/Right opening (%) | 旋转夹指开口，0% 闭合，100% 到 URDF 最大角度 |
| Reset + fold | 重置整个场景并执行一次双臂对折 |
| Stop | 停止自动/VR 控制并释放毛巾 |
| Home | 释放抓持、限速回到初始关节姿态，毛巾不重置 |
| Reset scene | 重置机器人、毛巾、抓持和仿真时间 |
| Pause | 暂停物理推进 |

默认初始开口为 50%。左右夹指为旋转关节，开口百分比不是毫米。
自动流程仅执行一次对折，结果由对齐误差和 X 方向跨度判断；失败原因会显示在界面和终端。

## VR 动作控制

先完成环境和证书配置，再运行 `make sim-vr`，在头显浏览器打开：

```text
https://<电脑局域网IP>:8012?ws=wss://<电脑局域网IP>:8012
```

进入 `Virtual Reality`，确认两只手柄被追踪。无需在终端按 Enter。
先松开左右扳机一次，再按住对应扳机移动。仅发送动作，不传输仿真画面到头显。

| 输入 | 行为 |
|---|---|
| 按住左/右扳机 | 对应手臂跟随手柄相对位移与旋转；松开停止跟随 |
| 左/右 squeeze 短按 | 对应夹爪开闭切换，近处有布料节点时辅助抓取 |
| 右 A | 释放抓持并回到 `scene.yaml` 初始姿态 |
| 右 B | 切换快速/精细动作映射 |
| 左 A/B | 本仿真入口未分配 |
| VR control | 开关 VR 控制；重新启用后释放扳机再接管 |

从桌面主视窗和腕部相机观察，分别将夹指移动到近侧两角后闭合。
`Held nodes: L ... / R ...` 对应侧大于 0 才表示抓取成功；未抓到则张开、调整位置，再闭合。
抬起两角后沿弧线移到远侧，放低，再分别张开释放。新版长夹爪可先使用约 8–9 cm 抬升弧线。
旋转的三个自由度均开放；姿态和位置组合受机械臂可达范围限制。

追踪/数据超时约 250 ms 会停止跟随，保留已抓住的毛巾；恢复后重新释放扳机。
手动调滑块、Home 或 Reset + fold 会退出 VR，避免多个控制源同时驱动机器人。
零位独立于真机。不要在此目录运行任何真机启动命令。

## 相机与导出

头部、左腕、右腕三路 RGB-D 默认 320 × 240、15 Hz，在桌面查看器显示。
相机窗口关闭后，可在 `Logged Images` 中重新选择 Cameras。

```bash
make sim ARGS="--viewer null --num-frames 180 --camera-output outputs/cameras"
make sim-fold ARGS="--headless --test --screenshot outputs/folded.png --camera-output outputs/folded-cameras"
make sim ARGS="--no-cameras"
```

导出每路 PNG 彩色图、`*_depth.npy` 深度（米）和 `cameras.json` 内外参。
`--camera-output` 保存最后一组观测，不是连续录像；重复使用输出目录会覆盖同名文件。
`--no-cameras` 关闭传感器渲染，URDF 中的相机外壳仍保留。
这些相机当前不参与自动抓点决策。

## 修改参数与测试

场景参数在 `simulation/scene.yaml`：毛巾尺寸、质量、网格密度、初始关节、桌高、相机、物理频率。
VR 倍率在 `teleop/VR_controller/config/gripper_pd.yaml` 的 `fast_*` / `fine_*` 字段。
TCP 和开合阈值在 `simulation/robot_model.py`，修改几何后需要重新验证。

```bash
make test
make test-vr-input   # 需要 VR 环境、证书和空闲 8012，使用合成事件，不需要实体头显
```

更改参数后，不应假设原有折叠轨迹仍然可达。测试不替代实体头显操作验收。
