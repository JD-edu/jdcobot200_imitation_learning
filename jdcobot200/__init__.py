"""JDCobot200 shared runtime package."""

from .config import JointConfig, RobotConfig, load_robot_config

__all__ = ["JointConfig", "RobotConfig", "load_robot_config"]
__version__ = "0.2.0"
