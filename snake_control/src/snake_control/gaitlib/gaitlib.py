import yaml
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

# ROS-optional import
try:
    import rospy  # type: ignore
except Exception:
    rospy = None


class Gaitlib(metaclass=ABCMeta):
    """
    Abstract interface for gait libraries.

    ROS-agnostic, ROS-ready:
      - If rospy is available, default gait params can come from ROS param server.
      - Otherwise, defaults are loaded from a YAML file (snake_control/param/snake_params.yaml).
    """

    def __init__(
        self,
        snake_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        params_yaml: Optional[str] = None,
    ):
        self.create_gait()

        # Decide snake_type
        if snake_type is None:
            if rospy is not None:
                snake_type = rospy.get_param("snake_type", "REU")
            else:
                snake_type = "REU"
        self._snake_type_runtime = snake_type

        # Load default gait params
        if params is not None:
            # Explicit dict overrides everything (useful for tests/teleop)
            self.default_gait_params = params.get(snake_type, {}).get("gait_params", {})
        else:
            if rospy is not None:
                self.default_gait_params = rospy.get_param(f"{snake_type}", {}).get("gait_params", {})
            else:
                # YAML fallback (repo-local, no ROS)
                ypath = params_yaml if params_yaml is not None else str(self._default_params_yaml())
                all_params = self.parse_params_yaml(ypath) or {}
                self.default_gait_params = all_params.get(snake_type, {}).get("gait_params", {})

        self.current_gait_params = [0.0] * self.num_gait_param
        self.current_gait = None

    @staticmethod
    def _default_params_yaml() -> Path:
        # snake_control/src/snake_control/gaitlib/gaitlib.py -> go up to snake_control/
        snake_control_dir = Path(__file__).resolve().parents[3]
        return snake_control_dir / "param" / "snake_params.yaml"

    @abstractmethod
    def create_gait(self):
        pass

    @property
    @abstractmethod
    def snake_type(self):
        pass

    @property
    @abstractmethod
    def num_modules(self):
        pass

    @property
    @abstractmethod
    def num_gait_param(self):
        pass

    @property
    @abstractmethod
    def gait_params_filepath(self):
        pass

    def update_params(self, params_dict, params_to_update):
        z = params_dict.copy()
        z.update(params_to_update)
        return z

    @staticmethod
    def parse_params_yaml(yaml_filepath):
        with open(yaml_filepath, "r") as f:
            return yaml.safe_load(f)
