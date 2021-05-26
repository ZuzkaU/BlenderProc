import random

import bpy

from src.camera.CameraSamplerModule import CameraSamplerModule
from src.utility.CameraUtility import CameraUtility
from src.utility.Config import Config
from src.utility.sampler.SuncgPointInRoomSampler import SuncgPointInRoomSampler


class SuncgCameraSampler(CameraSamplerModule):
    """ Samples valid camera poses inside suncg rooms.

    Works as the standard camera sampler, except the following differences:
    - Always sets the x and y coordinate of the camera location to a value uniformly sampled inside a rooms bounding box
    - The configured z coordinate of the configured camera location is used as relative to the floor
    - All sampled camera locations need to lie straight above the room's floor to be valid
    
    See parent class CameraSampler for more details.

    """
    def __init__(self, config):
        CameraSamplerModule.__init__(self, config)

    def run(self):
        self.point_sampler = SuncgPointInRoomSampler()
        super().run()

    def _sample_pose(self):
        """ Samples a new camera pose, sets the parameters of the given camera object accordingly and validates it.

        :return: True, if the sampled pose was valid
        """
        cam2world_matrix = super()._sample_pose()
        cam2world_matrix.translation = self.point_sampler.sample(cam2world_matrix.translation[2])
        return cam2world_matrix
