from math import radians
from pathlib import Path

import bpy
from mathutils import Matrix, Vector, Euler

import numpy as np

from src.camera.CameraInterface import CameraInterface
from src.utility.ItemCollection import ItemCollection
from src.utility.Utility import Utility


class MatterportPoseLoader(CameraInterface):
    def __init__(self, config):
        super().__init__(config)

        self.number_of_arguments_per_parameter = {
            "cam2world_matrix": 16
        }

        self.house_path = Path(Utility.resolve_path(self.config.get_string("house_path")))
        self.cam_pose_collection = ItemCollection(self._add_cam_pose, self.config.get_raw_dict("default_cam_param", {}))

    def convert_to_blender_space(self, matrix):
        # blender_rot_mat = Matrix.Rotation(radians(-90), 4, 'X').to_3x3()
        # print(matrix)

        translation = matrix[:, 3]
        rotation = matrix[:3, :3].copy()
        # print(rotation)
        rotation[:, 1] *= -1
        rotation[:, 2] *= -1
        # print(rotation)
        extrinsic = np.zeros((4,4)).astype(np.float)
        extrinsic[:3, :3] = rotation
        extrinsic[:, 3] = translation

        extrinsic = Matrix(extrinsic)

        # rotation = extrinsic.to_3x3()
        # euler = rotation.to_euler()
        # print("Euler before", euler)
        # euler.x *= -1
        # print("Euler after", euler)

        # translation = Matrix.Translation(extrinsic.to_translation())
        # transformed_rotation = euler.to_matrix().to_4x4()
        # euler = transformed_rotation.
        # transformed_rotation[3][3] = 0
        # extrinsic = translation @ transformed_rotation

        return extrinsic


    def run(self):
        #  parse conf file
        conf_file_path = self.house_path / "undistorted_camera_parameters" / (self.house_path.name + ".conf")
        with open(conf_file_path) as f:
            content = f.readlines()

        num_images = int(content[1].split(" ")[1])
        frames = []

        current_intrinsic_matrix = np.zeros((3, 3))
        for line in content:
            if len(line.strip()) == 0:
                continue

            parts = line.replace("  ", " ").strip().split(" ")
            if parts[0] == "intrinsics_matrix":
                # blender proc needs this as a list
                current_intrinsic_matrix = [float(p) for p in parts[1:]]

            if parts[0] == "scan":
                # parse frame

                # convert extrinsic to blender space
                # rotate around x by 90deg

                frame = {
                    "depth_frame": parts[1],
                    "color_frame": parts[2],
                    "cam2world_matrix": np.identity(4).tolist(),
                    "cam_K": current_intrinsic_matrix
                }

                frame_name = frame["depth_frame"].replace("_d", "_pose_").replace(".png", ".txt")
                # if frame_name != "76c7a665d2b242bfa203e7f394b1353e_pose_2_0.txt" and frame_name != "76c7a665d2b242bfa203e7f394b1353e_pose_1_2.txt": continue

                frames.append(frame)

        # load poses
        poses_path = self.house_path / "matterport_camera_poses"

        for frame_index, frame in enumerate(frames):
            frame_name = frame["depth_frame"].replace("_d", "_pose_").replace(".png", ".txt")

            pose_path = poses_path / frame_name
            pose = np.loadtxt(str(pose_path))
            pose_before = pose.reshape((4, 4))
            pose = self.convert_to_blender_space(pose_before)
            # if frame_name == "76c7a665d2b242bfa203e7f394b1353e_pose_2_0.txt":
            #     print(frame_index, Matrix(pose_before))
            #     print(frame_index, pose)
            #
            # if frame_name == "76c7a665d2b242bfa203e7f394b1353e_pose_1_2.txt":
            #     print(frame_index, pose_before)
            #     print(frame_index, pose)

            elements = []
            for v in pose:
                for e in v:
                    elements.append(e)

            frames[frame_index]["cam2world_matrix"] = elements  # blenderproc needs this as a list

        # add frames to collection
        for frame in frames:
            # frame_name = frame["depth_frame"].replace("_d", "_pose_").replace(".png", ".txt")
            # if frame_name == "76c7a665d2b242bfa203e7f394b1353e_pose_1_2.txt":

            self.cam_pose_collection.add_item(frame)

        # write key-frame index -> frame_name mapping
        frame_index_to_name_mapping = [(frame_index, frame["depth_frame"].replace("_d", "_").replace(".png", "")) for frame_index, frame in enumerate(frames)]
        output_path = Path(super()._determine_output_dir()) / "frame_names.txt"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, "w") as f:
            lines = [f"{line[0]} {line[1]}\n" for line in frame_index_to_name_mapping]
            f.writelines(lines)

    def _add_cam_pose(self, config):
        """ Adds new cam pose + intrinsics according to the given configuration.

        :param config: A configuration object which contains all parameters relevant for the new cam pose.
        """

        # Collect camera object
        cam_ob = bpy.context.scene.camera
        cam = cam_ob.data

        # Set intrinsics and extrinsics from config
        self._set_cam_intrinsics(cam, config)
        self._set_cam_extrinsics(cam_ob, config)

        # Store new cam pose as next frame
        frame_id = bpy.context.scene.frame_end
        self._insert_key_frames(cam, cam_ob, frame_id)

        bpy.context.scene.frame_end = frame_id + 1
