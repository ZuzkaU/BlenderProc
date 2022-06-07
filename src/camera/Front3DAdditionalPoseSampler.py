import json
import random
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import mathutils
from mathutils import Matrix

import bpy

from src.camera.CameraSampler import CameraSampler
from src.utility.BlenderUtility import get_bounds, hide_all_geometry, show_collection, world_to_camera


class Front3DAdditionalPoseSampler(CameraSampler):
    """
    This samples additional camera poses that share a large part of the view with 
    an original camera pose, but look at the view from different angles.
    """

    def __init__(self, config):
        CameraSampler.__init__(self, config)
        self.used_floors = []


    def run(self):
        amount_of_objects_needed_per_room = self.config.get_int("amount_of_objects_needed_per_room", 1)
        self.rooms = {}
        for room_obj in bpy.context.scene.objects:
            # Check if object is from type room and has bbox
            if "is_room" in room_obj and room_obj["is_room"] == 1:
                # count objects
                room_objects = [obj for obj in room_obj.children if "is_3D_future" in obj and obj["is_3D_future"] == 1]
                num_room_objects = len(room_objects)

                floors = list(filter(lambda x: x.name.lower().startswith("floor"), room_obj.children))

                if len(floors) == 0:
                    print(f"Skip {room_obj.name}: 0 floor objects found")
                    continue

                if len(floors) > 1 or len(floors) == 0:
                    print(f"Skip {room_obj.name}: {len(floors)} floor objects found")
                    continue

                floor = floors[0]

                if "num_floors" in floor and floor["num_floors"] > 2:
                    print(f"Skip {room_obj.name}: Too many floors merged ({floor['num_floors']})")
                    continue

                if num_room_objects < amount_of_objects_needed_per_room:
                    print(f"Skip {room_obj.name}: Not enough objects in room ({num_room_objects})")
                    continue

                self.rooms[room_obj.name] = room_obj, floor, room_obj["room_id"]

        output_path = Path(super()._determine_output_dir()) / f"room_mapping.json"
        with open(output_path, "w") as f:
            name_index_mapping = {obj[2]: name for name, obj in self.rooms.items()}
            json.dump(name_index_mapping, f, indent=4)

        print(f"Found {len(self.rooms)} rooms")
        super().run()

    def sample_and_validate_cam_pose(self, cam, cam_ob, config):
        """ Samples a new camera pose, sets the parameters of the given camera object accordingly and validates it.

        :param cam: The camera which contains only camera specific attributes.
        :param cam_ob: The object linked to the camera which determines general properties like location/orientation
        :param config: The config object describing how to sample
        :return: True, if the sampled pose was valid
        """
        # Sample room
        room_obj, floor_obj, room_index = self.rooms[self.current_room_name]

        # Sample/set intrinsics
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

        # Check if sampled pose is valid
        if self._is_pose_valid(floor_obj, cam, cam_ob, cam2world_matrix):
            # Set camera extrinsics as the pose is valid
            cam_ob.matrix_world = cam2world_matrix

            cam_ob["room_id"] = room_index
            return True
        else:
            return False
    
    def get_obj_visible_from_campose(self, cam, room_obj, campose):
        matrix = Matrix(campose["camera2world"])
        valid, visibilities = self._check_visible_overlap(cam, matrix)
        
        return visibilities
    
    def check_view_overlap(self, original_cam2world, sampled_cam2world):
        """
        assume that we want that at least k% rays in the sampled cam2world fall on objects that are also in original cam2world.
        
        """
        pass

    def _check_visible_overlap(self, cam, cam2world_matrix) -> Tuple[bool, Dict]:
        # Get position of the corners of the near plane
        frame = cam.view_frame(scene=bpy.context.scene)
        # Bring to world space
        frame = [cam2world_matrix @ v for v in frame]

        # Compute vectors along both sides of the plane
        vec_x = frame[1] - frame[0]
        vec_y = frame[3] - frame[0]

        position = cam2world_matrix.to_translation()

        objects_hit = {}

        num_rays = self.sqrt_number_of_rays

        for x in range(0, num_rays):
            for y in range(0, num_rays):
                # Compute current point on plane
                end = frame[0] + vec_x * x / float(num_rays - 1) \
                      + vec_y * y / float(num_rays - 1)
                # Send ray from the camera position through the current point on the plane
                hit, _, _, _, hit_object, _ = bpy.context.scene.ray_cast(bpy.context.view_layer, position, end - position)

                if hit:
                    model_id = hit_object.name
                    if model_id not in objects_hit:
                        objects_hit[model_id] = hit_object

        world2camera_matrix = cam2world_matrix.inverted()

        view_frame = cam.view_frame(scene=bpy.context.scene)
        view_frame_center = (view_frame[2] - view_frame[0]) / 2

        position_view = world_to_camera(cam, position, world2camera_matrix)

        object_visibilities = {}
        index = 0
        for model_id, obj in objects_hit.items():
            bounding_box = get_bounds(obj)
            bounding_box_view = np.array([world_to_camera(cam, v, world2camera_matrix) for v in bounding_box])

            min_coord = np.min(bounding_box_view, axis=0)
            max_coord = np.max(bounding_box_view, axis=0)

            min_coord_world = cam2world_matrix @ mathutils.Vector(min_coord)
            max_coord_world = cam2world_matrix @ mathutils.Vector(max_coord)

            min_coord_visible = np.clip(min_coord, a_min=0, a_max=None)
            max_coord_visible = np.clip(max_coord, a_max=1, a_min=None)

            min_coord_visible_world = cam2world_matrix @ mathutils.Vector(min_coord_visible)
            max_coord_visible_world = cam2world_matrix @ mathutils.Vector(max_coord_visible)

            extent_bbox = abs(max_coord - min_coord)
            extent_visible = abs(max_coord_visible - min_coord_visible)

            extent_ratio = extent_visible / extent_bbox

            if extent_ratio[0] < self.min_visible_overlap or extent_ratio[1] < self.min_visible_overlap:
                print(f"{model_id}: {extent_ratio} - Skip frame because overlap is too small")
                return False, object_visibilities
            else:
                object_visibilities[model_id] = extent_ratio

        return True, object_visibilities

    def _sample_cam_poses(self, config):
        """ Samples camera poses according to the given config

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
        
        self.original_campose = np.load(config.get_string("original_campose", "test/campose_0001.npz"))

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

        self.min_interest_score = interest_scores[score_index]
        print("Trying a min_interest_score value: %f" % self.min_interest_score)
        room_id = self.original_campose["room_id"]
        for room_name, (room_obj, _, rid) in self.rooms.items():
            if room_id == rid:
                break
        print(f"Sample views in {room_obj.name}")
        
        all_tries = 0  # max_tries is now applied per each score
        tries = 0
        self.current_room_name = room_name


        # hide everything except current room
        visibilities = self.get_obj_visible_from_campose(cam, room_obj, self.original_campose)
        print("Hide geometry")
        hide_all_geometry()
        print("display single room")
        for obj in room_obj.children:
            if obj.name in visibilities.keys():
                obj.hide_viewport = False
                obj.hide_render = False
        bpy.context.view_layer.update()

        print("start sampling")
        for i in range(number_of_poses):
            # Do until a valid pose has been found or the max number of tries has been reached
            fraction_tries = self.max_tries // 10

            while tries < self.max_tries:
                if tries % fraction_tries == 0:
                    print(f"Performed {tries} tires")
                tries += 1
                all_tries += 1
                # Sample a new cam pose and check if its valid
                if self.sample_and_validate_cam_pose(cam, cam_ob, config):
                    # Store new cam pose as next frame
                    frame_id = bpy.context.scene.frame_end
                    self._insert_key_frames(cam, cam_ob, frame_id)
                    self.insert_geometry_key_frame(room_obj, frame_id)
                    bpy.context.scene.frame_end = frame_id + 1

                    # if frame_id == 0:
                    # self._visualize_rays(cam, cam_ob.matrix_world, center_only=True)
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

    def _insert_key_frames(self, cam, cam_ob, frame_id):
        """ Insert key frames for all relevant camera attributes.

        :param cam: The camera which contains only camera specific attributes.
        :param cam_ob: The object linked to the camera which determines general properties like location/orientation
        :param frame_id: The frame number where key frames should be inserted.
        """
        # As the room id depends on the camera pose and therefore on the keyframe, we also need to add keyframes for the room id
        cam_ob.keyframe_insert(data_path='["room_id"]', frame=frame_id)

        # Set visibility key frames for all objects
        # Select all objects except current room, set obj.hide_render = True
        # obj.keyframe_insert(data_path='hide_render', frame=frame_id)

        # Add the usual key frames
        super()._insert_key_frames(cam, cam_ob, frame_id)

    def _is_pose_valid(self, floor_obj, cam, cam_ob, cam2world_matrix):
        """ Determines if the given pose is valid.

        - Checks if the pose is above the floor
        - Checks if the distance to objects is in the configured range
        - Checks if the scene coverage score is above the configured threshold

        :param floor_obj: The floor object of the room the camera was sampled in.
        :param cam: The camera which contains only camera specific attributes.
        :param cam_ob: The object linked to the camera which determines general properties like location/orientation
        :param cam2world_matrix: The sampled camera extrinsics in form of a camera to world frame transformation matrix.
        :return: True, if the pose is valid
        """
        #it doesn't have to be directly above floor, it can also be above object that are above floor 
        #if not self._position_is_above_object(cam2world_matrix.to_translation(), floor_obj):
        #    print("Not above floor")
        #    return False

        if not self._perform_obstacle_in_view_check(cam, cam2world_matrix):
            return False
        print("no obstacle in view")

        if self._is_ceiling_visible(cam, cam2world_matrix):
            print("Ceiling visible")
            return False

        scene_coverage_score, score, _, coverage_info = self._scene_coverage_score(cam, cam2world_matrix)
        scene_variance, variance_info = self._scene_variance(cam, cam2world_matrix)

        line = [f"Final score: {scene_coverage_score:4.3f}\tScores: ", " | ".join(["{0}: {1}".format(k, v) for k, v in coverage_info.items()])]
        line += [f"Variance: {scene_variance:4.3f}", "\tVariance: ", " | ".join(["{0}: {1}".format(k, v) for k, v in variance_info.items()])]
        if scene_coverage_score < self.min_interest_score or scene_variance < self.min_scene_variance:
            #print(f"\t\t", " ".join(line))
            pass
            #return False


        objects_are_visible, object_visibilities = self._check_visible_overlap(cam, cam2world_matrix)
        if not objects_are_visible:
            print("Object overlap too small")
            #return False
        line.append("\tVisibility: " + " | ".join([f"{k}-{v[0]:4.3f}/{v[1]:4.3f}" for k, v in object_visibilities.items()]))

        if self.check_pose_novelty and (not self._check_novel_pose(cam2world_matrix)):
            print("not novel")
            #return False

        if self._above_objects:
            for obj in self._above_objects:
                if self._position_is_above_object(cam2world_matrix.to_translation(), obj):
                    return True
            #return False

        print(" ".join(line))

        output_path = super()._determine_output_dir() + "/scores.txt"
        with open(output_path, "a") as f:
            f.write(" ".join(line) + "\n")

        return True

    def insert_geometry_key_frame(self, room_obj, frame_id):
        for room_name, (room, _, _) in self.rooms.items():
            should_hide = room_obj.name != room_name
            for obj in room.children:
                obj.hide_viewport = should_hide
                obj.hide_render = should_hide
                obj.keyframe_insert(data_path='hide_viewport', frame=frame_id)
                obj.keyframe_insert(data_path='hide_render', frame=frame_id)

