"""Robot state extraction and action building for Albert robot."""

import numpy as np
import pybullet as p


def yaw_from_quat(q):
    """Extract yaw angle from quaternion [x, y, z, w]."""
    x, y, z, w = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


# Robot body ID in PyBullet (set after environment creation)
_robot_body_id = None


def set_robot_body_id(body_id):
    """Set the PyBullet body ID for the robot."""
    global _robot_body_id
    _robot_body_id = body_id


def extract_base_state(_ob=None):
    """
    Extract robot base state (x, y, yaw) from PyBullet.
    
    The robot's URDF has facing_direction='-y', meaning in the robot's
    local frame, forward is -y. We need to correct for this when reading
    the orientation from PyBullet.
    
    Args:
        _ob: Observation (kept for API compatibility, but ignored)
    
    Returns:
        Array [x, y, yaw] where yaw is in radians, representing the
        direction the robot is actually facing (its forward direction)
    """
    global _robot_body_id
    
    if _robot_body_id is None:
        raise RuntimeError("Robot body ID not set. Call set_robot_body_id() first.")
    
    position, quaternion = p.getBasePositionAndOrientation(_robot_body_id)
    
    x = float(position[0])
    y = float(position[1])
    yaw_world = yaw_from_quat(quaternion)
    
    # CRITICAL FIX: Robot's forward direction is -y in local frame
    # This is a -90° rotation from the world frame orientation
    yaw = yaw_world - np.pi/2
    
    # Wrap to [-pi, pi]
    yaw = (yaw + np.pi) % (2*np.pi) - np.pi
    
    return np.array([x, y, yaw], dtype=float)


def vw_to_wheel_speeds(v, w, wheel_radius, wheel_distance):
    """
    Convert unicycle commands (v, w) to differential drive wheel speeds.
    
    Args:
        v: Linear velocity (m/s)
        w: Angular velocity (rad/s)
        wheel_radius: Radius of wheels (m)
        wheel_distance: Distance between wheels (m)
    
    Returns:
        (omega_right, omega_left): Wheel angular velocities (rad/s)
    """
    v_right = v + 0.5 * wheel_distance * w
    v_left = v - 0.5 * wheel_distance * w
    
    omega_right = v_right / wheel_radius
    omega_left = v_left / wheel_radius
    
    return omega_right, omega_left


def build_action(env_n, v, w, wheel_radius=0.08, wheel_distance=0.494):
    """
    Build action array for environment from velocity commands.
    
    Args:
        env_n: Size of environment action space
        v: Linear velocity command (m/s)
        w: Angular velocity command (rad/s)
        wheel_radius: Wheel radius (m)
        wheel_distance: Distance between wheels (m)
    
    Returns:
        Action array for environment.step()
    """
    action = np.zeros(env_n, dtype=float)
    
    # For GenericDiffDriveRobot with mode="vel"
    # action[0] = v (linear velocity)
    # action[1] = w (angular velocity)
    action[0] = v
    action[1] = w
    
    return action