# 来源与第三方组件

- Newton 与 NVIDIA Warp 通过 pip 安装，本包不重新分发其安装文件；对应许可证随上游软件包提供。
- `teleop/src/televuer` 源自当前项目的 Unitree TeleVuer 适配，包含控制器输入及无图像模式的本地修改。
  保留上游 MIT 许可全文于 `teleop/LICENSE`。
- `dual_arm/urdf` 与 `dual_arm/meshes` 是项目提供的 SolidWorks 导出机器人资产。
  源目录未附独立资产许可证，本包没有为这些模型声明新的许可。
- 仿真、抓取、VR 接入与测试为当前项目的交付快照；本说明不改变原有代码和资产的权利归属。

不包含当前电脑的私钥、真实机器人标定文件、用户数据、Conda 环境目录或第三方二进制缓存。
