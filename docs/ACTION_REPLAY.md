# 第一阶段：专家动作采集与独立重放

运行：

```bash
/home/shendongran/miniforge3/envs/towel-newton/bin/python scripts/action_replay.py
```

需要 CUDA。输出目录为 `outputs/action_replay/episode_000`。

- `trajectory.npz`：T+1 状态、T 动作，内部顺序为左7、右7、左开度、右开度。
- `canonical_samples.npz`：T 条观测和动作，顺序与参考数据一致：左7、左closed、右7、右closed。关节为弧度，夹爪以开度小于0.1转换为二值closed。
- `metadata.json`：机器人、版本、场景和控制语义。
- `verification.json`：仅当专家成功且独立重放通过后生成。

专家使用现有分段平滑末端轨迹和IK。NewtonTowelEnv 中的抓取由 step() 调用通用几何规则：关闭命令持续1秒、实际夹爪足够关闭且粒子位于夹持区域时，绑定距离指尖中心最近的至多两个粒子；打开命令释放。抓取不使用角点编号或专家阶段。专家本身仍以已知网格角点规划目标。

重放重置相同初态，禁止 fold.update 和 fold.attach 调用，仅执行记录动作。检查发生抓取、最终释放、折叠指标达标，并报告逐帧布料位置最大差异；GPU布料轨迹不要求逐位复现，成功以独立重放的折叠指标判断。

这是固定初态、原生仿真频率的控制接口验收。NPZ 是中间数据，不是完整 LeRobot 数据集；尚未包含同步 RGB、30Hz重采样、随机场景、LeRobot导出或ACT训练。不能直接用 canonical_samples.npz 替代视觉训练数据。
