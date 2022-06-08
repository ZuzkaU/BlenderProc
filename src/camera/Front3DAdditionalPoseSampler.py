import json
import random
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import mathutils
from mathutils import Matrix, Vector, Euler
from src.utility.Utility import Utility
from src.camera.Front3DCameraSampler import Front3DCameraSampler

import bpy

from src.camera.CameraSampler import CameraSampler
from src.utility.BlenderUtility import get_bounds, hide_all_geometry, show_collection, world_to_camera

from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull, Delaunay
from PIL import Image


class Front3DAdditionalPoseSampler(Front3DCameraSampler):
    """
    This samples additional camera poses that share a large part of the view with 
    an original camera pose, but look at the view from different angles.
    """

    def __init__(self, config):
        Front3DCameraSampler.__init__(self, config)
        self.img_num = 0


    def run(self):
        super().run()
    
    def compute_mask(self, cam, original_matrix, sampled_matrix):
        """
        computes which pixels in the view of sampled_matrix are also visible in original_matrix
        """
        # test if the mask is correct
        original_matrix = Matrix(np.load('../../sample/359821fc-7594-4482-91c6-51a89cefe2b6/campose_0019.npz')['blender_matrix'])
        sampled_matrix = Matrix(np.load('../../sample/359821fc-7594-4482-91c6-51a89cefe2b6/campose_0020.npz')['blender_matrix'])
    
        frame = cam.view_frame(scene=bpy.context.scene)
        original_frame = [original_matrix @ v for v in frame]
        sampled_frame = [sampled_matrix @ v for v in frame]
        position = sampled_matrix.to_translation()
        
        vec_x = sampled_frame[1] - sampled_frame[0]
        vec_y = sampled_frame[3] - sampled_frame[0]

        mask = np.zeros((240, 320), dtype=np.dtype('uint8'))
        masked_pixels, visible_pixels, nohit = 0, 0, 0
        for x in range(0, 320):
            for y in range(0, 240):
                # Compute current point on plane
                end = sampled_frame[0] + vec_x * x / float(320 - 1) \
                      + vec_y * y / float(240 - 1)
                # Send ray from the camera position through the current point on the plane
                hit, location, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.view_layer, position, end - position)

                if not hit:
                    nohit += 1
                if hit and self.location_inside_frame(location, frame, original_matrix):
                    mask[y,x] = 255
                    visible_pixels += 1
                else:
                    mask[y,x] = 0
                    masked_pixels += 1
                # Possible optimization checks (if the pixels are sampled randomly)
                # if masked_pixels >= 320*240*0.8:
                #     print("too many masked pixels")
                #     return None
                # if visible_pixels == 0 and masked_pixels > 1000:
                #     print("too little visible pixels")
                #     return None
        print(f"masked pixels: {masked_pixels}, visible: {visible_pixels}, nohit: {nohit}")
        # https://stackoverflow.com/questions/32159076/python-pil-bitmap-png-from-array-with-mode-1
        # Using mode 'L' instead of '1'
        return Image.fromarray(mask, mode='L')
    
    def location_inside_frame(self, location, frame, cam2world_matrix):
        """
        determines if location is inside camera view given by frame and matrix.
        """
        frame = [cam2world_matrix @ v for v in frame]
        position = cam2world_matrix.to_translation()
        endings = [position + 10*(f-position) for f in frame]
        
        # Inefficient implementation to see if it works correctly
        hull = ConvexHull([position] + endings)
        new_hull = ConvexHull([position] + endings + [location])
        inside = np.array_equal(new_hull.vertices, hull.vertices)
        return np.array_equal(new_hull.vertices, hull.vertices)    

    def sample_and_validate_cam_pose(self, cam, cam_ob, config):
        """ Samples a new camera pose, sets the parameters of the given camera object accordingly and validates it.

        :param cam: The camera which contains only camera specific attributes.
        :param cam_ob: The object linked to the camera which determines general properties like location/orientation
        :param config: The config object describing how to sample
        :return: True, if the sampled pose was valid
        """
        # Get room
        room_obj, floor_obj, room_index = self.rooms[self.current_room_name]

        # Set intrinsics (should be always the same)
        self._set_cam_intrinsics(cam, config)

        # Sample camera extrinsics (we do not set them yet for performance reasons)
        cam2world_matrix = self._cam2world_matrix_from_cam_extrinsics(config)

        # Make sure the sampled location is inside the room => overwrite x and y and add offset to z
        bounding_box = get_bounds(floor_obj)
        min_corner = np.min(bounding_box, axis=0)
        max_corner = np.max(bounding_box, axis=0)

        cam2world_matrix.translation[0] = random.uniform(min_corner[0], max_corner[0])
        cam2world_matrix.translation[1] = random.uniform(min_corner[1], max_corner[1])
        cam2world_matrix.translation[2] += floor_obj.location[2]


        # Check if sampled pose is valid: compute_mask should return a mask if 
        # there are enough non-masked pixels, and None otherwise
        # (the only requirement for pose validity)
        mask = self.compute_mask(cam, Matrix(self.original_campose["blender_matrix"]), cam2world_matrix)
        if mask:
            name = f"output-mask/{self.img_num}.png"
            print(f"saving mask as {name}")
            mask.save(name)
            self.img_num += 1
            cam_ob.matrix_world = cam2world_matrix
            cam_ob["room_id"] = room_index
            return True
        else:
            return False

    def _sample_cam_poses(self, config):
        """ Samples camera poses according to the given config
        used as add_item_func in ItemCollection cam_pose_collection, which adds camera poses to render
        :param config: The config object
        """
        cam_ob = bpy.context.scene.camera
        cam = cam_ob.data

        # Set global parameters
        self._is_bvh_tree_inited = False
        self.sqrt_number_of_rays = config.get_int("sqrt_number_of_rays", 10)
        self.max_tries = config.get_int("max_tries", 100000000)
        self.proximity_checks = config.get_raw_dict("proximity_checks", {})
        self.excluded_objects_in_proximity_check = config.get_list("excluded_objs_in_proximity_check", [])
        self.min_visible_overlap = config.get_float("min_visible_overlap", 0.0)
        self.min_scene_variance = config.get_float("min_scene_variance", 0.0)
        self.min_interest_score = config.get_float("min_interest_score", 0.0)
        self.interest_score_range = config.get_float("interest_score_range", self.min_interest_score)
        self.interest_score_step = config.get_float("interest_score_step", 0.1)
        self.special_objects = config.get_list("special_objects", [])
        self.special_objects_weight = config.get_float("special_objects_weight", 2)
        self.excluded_objects_in_score_check = config.get_list("excluded_objects_in_score_check", [])
        self.excluded_objects_in_overlap_check = config.get_list("excluded_objects_in_overlap_check", [])
        self.center_region_x_percentage = config.get_float("center_region_x_percentage", 0.5)
        self.center_region_y_percentage = config.get_float("center_region_y_percentage", 0.5)
        self.center_region_weight = config.get_float("center_region_weight", 5)
        self._above_objects = config.get_list("check_if_pose_above_object_list", [])

        if self.proximity_checks:
            # needs to build an bvh tree
            self._init_bvh_tree()

        if self.interest_score_step <= 0.0:
            raise Exception("Must have an interest score step size bigger than 0")

        # Determine the number of camera poses to sample
        number_of_poses = config.get_int("number_of_samples", 1)  # num samples per room
        print("Sampling " + str(number_of_poses) + " cam poses")

        if self.min_interest_score == self.interest_score_range:
            step_size = 1
        else:
            step_size = (self.interest_score_range - self.min_interest_score) / self.interest_score_step
            step_size += 1  # To include last value
        # Decreasing order
        interest_scores = np.linspace(self.interest_score_range, self.min_interest_score, step_size)
        score_index = 0
        
        """
        Up to here all copied from CameraSampler / Front3DCameraSampler.
        Need to get the original camera pose and set the visible room from it.
        The try-sample cycle is again the same as in Front3DCameraSampler.
        """
        
        self.original_campose = np.load(config.get_string("original_campose", "test/campose_0001.npz"))
        room_id = self.original_campose["room_id"]
        for room_name, (room_obj, _, rid) in self.rooms.items():
            if room_id == rid:
                break
        self.current_room_name = room_name
        print(f"Original camera pose is located in {room_obj.name}, sampling views from here.")
        
        all_tries = 0  # max_tries is now applied per each score
        tries = 0

        # hide everything except current room
        print("Hide geometry")
        hide_all_geometry()
        print("display single room")
        show_collection(room_obj)
        bpy.context.view_layer.update()


        self.min_interest_score = interest_scores[score_index]
        print("Trying a min_interest_score value: %f" % self.min_interest_score)
        print("start sampling")
        for i in range(number_of_poses):
            # Do until a valid pose has been found or the max number of tries has been reached
            fraction_tries = self.max_tries // 10

            while tries < self.max_tries:
                if tries % fraction_tries == 0:
                    print(f"Performed {tries} tries")
                tries += 1
                all_tries += 1
                # Sample a new cam pose and check if its valid
                if self.sample_and_validate_cam_pose(cam, cam_ob, config):
                    # Store new cam pose as next frame
                    frame_id = bpy.context.scene.frame_end
                    self._insert_key_frames(cam, cam_ob, frame_id)
                    self.insert_geometry_key_frame(room_obj, frame_id)
                    bpy.context.scene.frame_end = frame_id + 1
                    break

            if tries >= self.max_tries:
                if score_index == len(interest_scores) - 1:  # If we tried all score values
                    print(f"Maximum number of tries reached! Found: {bpy.context.scene.frame_end} poses")
                    break
                # Otherwise, try a different lower score and reset the number of trials
                score_index += 1
                self.min_interest_score = interest_scores[score_index]
                print("Trying a different min_interest_score value: %f" % self.min_interest_score)
                tries = 0

        print(str(all_tries) + " tries were necessary")


