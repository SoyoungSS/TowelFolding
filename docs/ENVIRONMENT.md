# 环境配置

## 主机要求

已验证主机为 Ubuntu 24.04 x86_64、NVIDIA RTX 5080 Laptop GPU（约 16 GB 显存）、驱动 595.99.02。
这是一组验证配置，不代表必须使用同型号 GPU。当前交付以 Linux + NVIDIA CUDA 为运行目标。
需要 Conda、GNU Make；桌面查看器需要可用的 OpenGL 桌面会话。VR 还需要 OpenSSL 和支持 WebXR 的头显浏览器。

```bash
nvidia-smi
conda --version
make --version
```

当前使用 Newton 1.7.0、Warp 1.18.0 和 Python 3.12。不要直接跟随最新版升级。
本机通过 Python wheel 安装，无需额外编译 Newton 或安装系统 CUDA Toolkit。
其他 GPU/驱动组合以 `make check` 的实际 CUDA 内核检查为准。

## 安装仿真环境

收到压缩包后可先在其所在目录执行：

```bash
sha256sum -c newton_towel_sim.tar.gz.sha256
tar -xzf newton_towel_sim.tar.gz
cd newton_towel_sim
```

在解压后的 `newton_towel_sim` 目录执行：

```bash
conda env create -f environments/towel-newton.yml
conda run -n towel-newton python -m pip install -r environments/newton-requirements.txt
conda run -n towel-newton python -m pip check
make verify
make check
make sim ARGS="--viewer null --num-frames 180 --test"
make sim
```

若环境已存在，可跳过创建，直接安装对应 requirements。不要 source 原项目的 ROS 环境。
Makefile 使用 `conda run`，无需预先 `conda activate`；Conda 可执行文件必须在 PATH 中。
环境名称不同则在命令后附加 `SIM_ENV=你的环境名`，并在安装命令中使用同一名称。

`newton-requirements.txt` 固定直接使用的包；`newton-tested-freeze.txt` 是交付机的完整 pip 依赖快照。
需要重建相同 pip 版本集合时，可改为安装该 freeze 文件。它不包含操作系统驱动、Conda 本体或所有系统库。

## 安装可选 VR 环境

普通查看和自动折叠不需要这一环境。VR 输入与仿真分进程运行，避免 NumPy 1.x/2.x 依赖冲突。

```bash
conda env create -f environments/towel-vr.yml
conda run -n towel-vr python -m pip install -r environments/vr-requirements.txt
conda run -n towel-vr python -m pip check
make certificate HOST_IP=192.168.1.10
make sim-vr
```

将示例 IP 改成电脑实际局域网 IPv4 地址。证书只在本机生成，交付包不携带任何证书或私钥。
已有证书时脚本不会覆盖；换 IP 后需要自行备份并移走旧文件，再重新生成。
头显必须信任本地证书，电脑和头显应能通过局域网连接 TCP 8012。访问页面后还需要进入 WebXR。
自签名证书的信任方式取决于头显浏览器；若浏览器不允许自签名 HTTPS 进入 XR，应部署该设备信任的证书。

默认 `VR_ENV=towel-vr`；可使用 `make sim-vr VR_ENV=自定义环境名`。
也可以直接指定已有、依赖兼容的解释器：`make sim-vr VR_PYTHON=/绝对路径/bin/python`。
该入口不依赖 `.venv-robot`，不发送 ROS 或真机控制指令。

## 常见问题

| 现象 | 处理 |
|---|---|
| `conda: command not found` | 安装/初始化 Conda，使新终端能运行 `conda --version` |
| CUDA unavailable | 在主机终端确认 `nvidia-smi`，检查驱动和 GPU 权限；不能将跳过 GPU 测试看作通过 |
| 首次启动耗时较长 | Warp 首次编译内核，观察终端编译输出；后续会使用缓存 |
| OpenGL/显示连接失败 | 在图形桌面运行；服务器先使用 `--viewer null`，`--headless` 仍需要 OpenGL |
| 8012 已被占用 | 退出另一份 Vuer/VR 服务后启动；不要同时开两份 VR 输入服务 |
| 找不到网格 | 保留整个目录结构并执行 `make verify`，不要只复制 URDF |
| NumPy 或模块导入冲突 | 使用此处两套独立环境，检查是否误用了 ROS/Python 环境 |

安装依赖需能访问 Conda/PyPI；头显页面还可能访问 Vuer 的前端资源，本包不保证离线 WebXR。
