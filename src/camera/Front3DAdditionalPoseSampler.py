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
import os

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
        frame = cam.view_frame(scene=bpy.context.scene)
        original_frame = [original_matrix @ v for v in frame]
        sampled_frame = [sampled_matrix @ v for v in frame]
        position = sampled_matrix.to_translation()

        vec_y = sampled_frame[1] - sampled_frame[0]
        vec_x = sampled_frame[3] - sampled_frame[0]

        x_dim, y_dim = 320, 240
        mask = np.zeros((y_dim, x_dim), dtype=np.dtype('uint8'))
        masked_pixels, visible_pixels, nohit = 0, 0, 0
        for x in random.sample(range(x_dim), k=x_dim):
            for y in random.sample(range(y_dim), k=y_dim):
                # Compute current point on plane
                end = sampled_frame[0] + vec_x * x / float(x_dim - 1) \
                      + vec_y * y / float(y_dim - 1)
                # Send ray from the camera position through the current point on the plane
                hit, location, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.view_layer, position, end - position)

                if not hit:
                    nohit += 1
                if hit and self.location_inside_frame(cam, location, original_matrix):
                    mask[y,x_dim-1-x] = 255
                    visible_pixels += 1
                else:
                    mask[y,x_dim-1-x] = 0
                    masked_pixels += 1
                if masked_pixels >= x_dim*y_dim*0.3:
                    print(f"too many masked pixels; visible {visible_pixels}, masked {masked_pixels}, nohit {nohit}")
                    return None
                if visible_pixels == 0 and masked_pixels > 1000:
                    print(f"too little visible pixels; visible {visible_pixels}, masked {masked_pixels}, nohit {nohit}")
                    return None
                if visible_pixels + masked_pixels > 1000 and visible_pixels / (visible_pixels + masked_pixels) < 0.3:
                    print(f"too low ratio; visible {visible_pixels}, masked {masked_pixels}, nohit {nohit}")
                    return None
        print(f"masked pixels: {masked_pixels}, visible: {visible_pixels}, nohit: {nohit}")
        # https://stackoverflow.com/questions/32159076/python-pil-bitmap-png-from-array-with-mode-1
        # Using mode 'L' instead of '1'
        return Image.fromarray(mask, mode='L')
    
    def location_inside_frame(self, cam, location, cam2world_matrix):
        """
        determines if location is inside camera view given by frame and matrix.
        """
        frame = cam.view_frame(scene=bpy.context.scene)
        frame = [cam2world_matrix @ v for v in frame]
        position = cam2world_matrix.to_translation()
        safe_distance = 10
        # make a pyramid-polygon with apex at the camera and base far enough in the direction of frame
        endings = [position + safe_distance*(f-position) for f in frame]
        
        # Not very efficient implementation to see if it works correctly
        # Maybe try this with linear programming, multiple points at a time, save the camera pyramid...?
        # https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
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
        if self._is_pose_valid(cam, cam_ob, cam2world_matrix):
            self.img_num += 1
            cam_ob.matrix_world = cam2world_matrix
            cam_ob["room_id"] = room_index
            return True
            #mask = self.compute_mask(cam, Matrix(self.original_campose["blender_matrix"]), cam2world_matrix)
            #if mask:
            #    path = os.path.join(config.get_string("output_dir"), f"mask_{str(self.img_num).zfill(4)}.png")
            #    print(f"saving mask as {path}")
            #    mask.save(path)
            #    self.img_num += 1
            #    cam_ob.matrix_world = cam2world_matrix
            #    cam_ob["room_id"] = room_index
            #    return True
        else:
            return False

    
    def _is_pose_valid(self, cam, cam_ob, cam2world_matrix):
        " See _is_pose_valid in CameraSampler.py "
        if not self._perform_obstacle_in_view_check(cam, cam2world_matrix):
            #print("Obstacle in view")
            self.errors["obstacle"] += 1
            self.obstacles[0] += 1
            return False

        #if self._is_ceiling_visible(cam, cam2world_matrix):
        #    #print("Ceiling visible")
        #    self.errors["ceiling"] += 1
        #    return False

        scene_coverage_score, score, _, coverage_info = self._scene_coverage_score(cam, cam2world_matrix)
        scene_variance, variance_info = self._scene_variance(cam, cam2world_matrix)

        line = [f"Final score: {scene_coverage_score:4.3f}\tScores: ", " | ".join(["{0}: {1}".format(k, v) for k, v in coverage_info.items()])]
        line += [f"Variance: {scene_variance:4.3f}", "\tVariance: ", " | ".join(["{0}: {1}".format(k, v) for k, v in variance_info.items()])]
        if scene_coverage_score < self.min_interest_score:
            #print(f"\t\t", " ".join(line))
            #print("\t", dict(coverage_info))
            #print("Low coverage score / variance")
            self.obstacles[1] += 1
            self.errors["coverage"] += 1
            return False
        if scene_variance < self.min_scene_variance:
            self.errors["variance"] += 1
            return False


        if self.check_pose_novelty and (not self._check_novel_pose(cam2world_matrix)):
            #print("not novel")
            self.errors["novelty"] += 1
            return False

        if self._above_objects:
            is_above_some_object = False
            for obj in self._above_objects:
                if self._position_is_above_object(cam2world_matrix.to_translation(), obj):
                    is_above_some_object = True
            if not is_above_some_object:
                #print("not above objects")
                self.errors["above"] += 1
                return False

        #print(" ".join(line))

        if not self._enough_common_pixels(cam, cam2world_matrix, Matrix(self.original_campose["blender_matrix"])):
            #print("not enough common pixels")
            self.errors["common"] += 1
            return False

        output_path = super()._determine_output_dir() + "/scores.txt"
        with open(output_path, "a") as f:
            f.write(" ".join(line) + "\n")

        #print("return true")
        return True
    
    def _enough_common_pixels(self, cam, cam2world_matrix, original_matrix):
        # Get position of the corners of the near plane
        frame = cam.view_frame(scene=bpy.context.scene)
        # Bring to world space
        frame = [cam2world_matrix @ v for v in frame]

        # Compute vectors along both sides of the plane
        vec_x = frame[1] - frame[0]
        vec_y = frame[3] - frame[0]

        position = cam2world_matrix.to_translation()

        hits_inside = 0

        for x in range(0, self.sqrt_number_of_rays):
            for y in range(0, self.sqrt_number_of_rays):
                x_ratio = x / float(self.sqrt_number_of_rays - 1)
                y_ratio = y / float(self.sqrt_number_of_rays - 1)
                end = frame[0] + vec_x * x_ratio + vec_y * y_ratio
                # start = end - offset
                start = position

                # Send ray from the camera position through the current point on the plane
                hit, location, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.view_layer, start, end-start)

                if hit and self.location_inside_frame(cam, location, original_matrix):
                    hits_inside += 1
        if hits_inside < self.sqrt_number_of_rays * self.sqrt_number_of_rays * self.min_common_pixels:
            return False
        return True

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

        self.min_common_pixels = config.get_float("min_common_pixels", 0.4)

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
        
        #self._set_cam_intrinsics(cam, config)
        # (setting in sample_and_validate_cam_pose)

        all_views_in_scene = sorted([f for f in os.listdir(config.get_string("scene_dir")) if f.startswith("campose")])

        view_dispatch = {} # which output belongs to which original campose
        for campose_name in all_views_in_scene[:config.get_int("max_views", 100)]:
            # This will output all additional views from one scene in the same output dir, numbered from 0 onwards.
            # Renaming and splitting directories is handled in a bash script.

            campose_nr = campose_name.split("_")[1].split(".")[0]
            view_dispatch[campose_nr] = []
            #self.original_campose = np.load(config.get_string("original_campose", "test/campose_0001.npz"))
            self.original_campose = np.load(os.path.join(config.get_string("scene_dir"), campose_name))

            #print(self.original_campose["intrinsic"])
            self._set_cam_intrinsics(cam, config)

            room_id = self.original_campose["room_id"]
            for room_name, (room_obj, _, rid) in self.rooms.items():
                if room_id == rid:
                    break
            if not room_id == rid:
                print(f"wrong room id: room with id {room_id} not found!")
                continue
            self.current_room_name = room_name
            print(f"Original camera pose is located in {room_obj.name}, sampling views from here.")
            
            all_tries = 0  # max_tries is now applied per each score
            tries = 0
            self.errors = dict()
            self.errors["obstacle"] = 0
            self.errors["ceiling"] = 0
            self.errors["coverage"] = 0
            self.errors["variance"] = 0
            self.errors["novelty"] = 0
            self.errors["above"] = 0
            self.errors["common"] = 0
            original_frame = bpy.context.scene.frame_end

            # hide everything except current room
            #print("Hide geometry")
            #hide_all_geometry()
            #print("display single room")
            #show_collection(room_obj)
            bpy.context.view_layer.update()

            # add original view to render
            print("adding original view")
            self.img_num += 1
            cam_ob.matrix_world = Matrix(self.original_campose["blender_matrix"])
            room_obj, floor_obj, room_index = self.rooms[self.current_room_name]
            cam_ob["room_id"] = room_index
            frame_id = bpy.context.scene.frame_end
            self._insert_key_frames(cam, cam_ob, frame_id)
            bpy.context.scene.frame_end = frame_id + 1
            view_dispatch[campose_nr].append(self.img_num - 1)

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
                        # (cam_ob.matrix_world set in sample_and_validate_cam_pose)
                        frame_id = bpy.context.scene.frame_end
                        self._insert_key_frames(cam, cam_ob, frame_id)
                        #self.insert_geometry_key_frame(room_obj, frame_id)
                        bpy.context.scene.frame_end = frame_id + 1
                        view_dispatch[campose_nr].append(self.img_num - 1)
                        break

                if tries >= self.max_tries:
                    if score_index == len(interest_scores) - 1:  # If we tried all score values
                        print(f"Maximum number of tries reached! Found: {bpy.context.scene.frame_end - original_frame} poses")
                        break
                    # Otherwise, try a different lower score and reset the number of trials
                    score_index += 1
                    self.min_interest_score = interest_scores[score_index]
                    print("Trying a different min_interest_score value: %f" % self.min_interest_score)
                    tries = 0

            print(str(all_tries) + " tries were necessary")
            print("Error types:", self.errors)

        with open(os.path.join(config.get_string("output_dir"), 'view_dispatch.json'), 'w') as f:
            f.write(json.dumps(view_dispatch, sort_keys=True, indent=4))
        for view in view_dispatch:
                with open(os.path.join(config.get_string("output_dir"), 'views_' + view + '.txt'), 'w') as f:
                    for additional in view_dispatch[view]:
                            f.write(str(additional).zfill(4) + '\n')

