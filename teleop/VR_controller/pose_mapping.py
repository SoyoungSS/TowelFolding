"""Shared robot-frame pose mapping for hardware and simulation VR input."""

import numpy as np


def matrix_to_quaternion(matrix):
    """从 4x4 变换矩阵提取单位四元数，返回 [w, x, y, z]。"""
    rotation_matrix = matrix[:3, :3]

    trace = rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
    else:
        if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
            w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            x = 0.25 * s
            y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
            w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            y = 0.25 * s
            z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
            w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            z = 0.25 * s

    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm > 0:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array([w, x, y, z])


def quaternion_to_euler_zyx(quaternion):
    """四元数 [w, x, y, z] 转 ZYX 欧拉角，返回 [yaw, pitch, roll]。"""
    w, x, y, z = quaternion

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    return np.array([yaw, pitch, roll])


def _head_yaw_rotation(head_pose):
    """Return the horizontal (yaw-only) head rotation in robot coordinates."""
    if head_pose is None:
        return np.eye(3)
    rotation = np.asarray(head_pose, dtype=float)[:3, :3]
    # Robot axes are x=front, y=left, z=up.  Ignore headset pitch/roll so
    # looking up/down cannot turn a horizontal hand motion into vertical motion.
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _head_rotation(head_pose):
    """Return a proper full head rotation, with a safe identity fallback."""
    if head_pose is None:
        return np.eye(3)
    rotation = np.asarray(head_pose, dtype=float)[:3, :3]
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        return np.eye(3)
    # XR tracking matrices should already be orthonormal.  Re-orthogonalize
    # defensively so noisy frames cannot inject scale/shear into a quaternion.
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation


def _rotate_xy(delta, cy, sy):
    """Rotate the horizontal (x=front, y=left) plane of a 3-vector by a yaw.

    The vertical z component is left unchanged.  Used to cancel a constant
    angular offset between the hand frame and the robot frame so that straight
    hand motion maps to straight robot motion.
    """
    x, y, z = delta[0], delta[1], delta[2]
    return np.array([cy * x - sy * y, sy * x + cy * y, z])


def _limit_increment(delta, enabled, hard_max_norm, max_speed, sample_dt):
    """Reject tracking jumps and cap speed without rescaling displacement."""
    delta = np.asarray(delta, dtype=float).copy()
    norm = float(np.linalg.norm(delta))
    if not enabled or not np.all(np.isfinite(delta)) or norm > hard_max_norm:
        return np.zeros(3)
    max_step = max_speed * sample_dt
    if norm > max_step and norm > 1e-12:
        delta *= max_step / norm
    return delta


def _conservative_deadband(delta, residual, threshold, enabled):
    """Keep sub-threshold movement until it accumulates into an increment."""
    if not enabled:
        residual.fill(0.0)
        return np.zeros(3)
    residual += delta
    output = residual.copy()
    output[np.abs(output) < threshold] = 0.0
    residual -= output
    return output


def robot_frame_position_delta(
    current_pose,
    previous_pose,
    current_head_pose,
    previous_head_pose,
    head_reference_rotation,
):
    """Compute a head-relative hand displacement and express it in robot axes."""
    if current_pose is None or previous_pose is None:
        return np.zeros(3)
    waist_offset = np.array([0.15, 0.0, 0.45])
    current = np.asarray(current_pose, dtype=float)[:3, 3] - waist_offset
    previous = np.asarray(previous_pose, dtype=float)[:3, 3] - waist_offset
    # Position follows the horizontal head frame only.  Looking up/down or
    # tilting the headset must not turn a horizontal hand motion into Z motion.
    current_head = _head_yaw_rotation(current_head_pose)
    previous_head = _head_yaw_rotation(previous_head_pose)
    reference = np.asarray(head_reference_rotation, dtype=float)
    if reference.shape != (3, 3) or not np.all(np.isfinite(reference)):
        reference = np.eye(3)
    reference_yaw = np.arctan2(reference[1, 0], reference[0, 0])
    c, s = np.cos(reference_yaw), np.sin(reference_yaw)
    reference = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    delta_in_head = current_head.T @ current - previous_head.T @ previous
    return reference @ delta_in_head


def robot_frame_orientation_delta(
    current_arm_pose,
    previous_arm_pose,
    current_head_pose,
    previous_head_pose,
    head_reference_rotation,
):
    """Return the controller rotation increment in the fixed robot frame.

    The tracker reports both controller and headset in its world frame.  The
    controller must be relative to the headset so body/head motion does not
    command the robot.  That relative increment is expressed in head axes,
    however, while robot_control_hw left-multiplies every received increment
    in fixed robot axes.  Conjugating with the headset orientation captured at
    teleoperation start performs this frame conversion explicitly.
    """
    if current_arm_pose is None or previous_arm_pose is None:
        return np.zeros(3)

    current_arm = _head_rotation(current_arm_pose)
    previous_arm = _head_rotation(previous_arm_pose)
    current_head = _head_rotation(current_head_pose)
    previous_head = _head_rotation(previous_head_pose)
    reference = np.asarray(head_reference_rotation, dtype=float)
    if reference.shape != (3, 3) or not np.all(np.isfinite(reference)):
        reference = np.eye(3)

    current_relative = current_head.T @ current_arm
    previous_relative = previous_head.T @ previous_arm
    delta_in_head = current_relative @ previous_relative.T
    delta_in_robot = reference @ delta_in_head @ reference.T

    transform = np.eye(4)
    transform[:3, :3] = delta_in_robot
    return quaternion_to_euler_zyx(matrix_to_quaternion(transform))
