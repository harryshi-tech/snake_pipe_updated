import os
import numpy as np
from snake_control.gaitlib.gaitlib import Gaitlib

"""
Gaits for the ReUnified (ReU) snake.
"""


class ReuGaits(Gaitlib):
    def create_gait(self):
        pass

    snake_type = "REU"
    num_modules = 16
    num_gait_param = 10

    @property
    def gait_params_filepath(self):
        dir_name = os.path.dirname(__file__)
        return os.path.join(dir_name, "reu_gait_parameters.yaml")

    @staticmethod
    def flip_axes(target_angles):
        """Flip sign for hardware convention."""
        direction = 1
        for i in range(len(target_angles[0])):
            target_angles[:, i] *= direction
            if not ((i + 1) % 2):
                direction *= -1
        return target_angles

    # Import gait methods
    from .compound_serpenoid import compound_serpenoid
    from .head_look import head_look
    from .head_look_ik import head_look_ik
    from .lateral_undulation import lateral_undulation
    from .linear_progression import linear_progression
    from .rolling import rolling
    from .rolling_helix import rolling_helix
    from .rolling_in_shape import rolling_in_shape
    from .slithering import slithering
    from .turn_in_place import turn_in_place
    from .conical_sidewinding import conical_sidewinding
    
    # T-junction gaits (window functions must be imported as methods)
    from .t_junction import (
        t_junction,
        gaussian_window,
        sinus_window,
        amplitude_reduced,
        amplitude_reduced_sinus,
        parameter_windowed,
        exp_window,
    )
    from .spiraling import spiraling
    from .windowed_rolling_helix import windowed_rolling_helix