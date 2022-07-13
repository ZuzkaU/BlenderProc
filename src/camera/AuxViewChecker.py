import bpy
import numpy as np
from mathutils import Matrix, Vector, Euler
from scipy.spatial import ConvexHull, Delaunay
import os

from src.main.Module import Module
from src.utility.Utility import Utility
from src.camera.Front3DCameraSampler import Front3DCameraSampler

class AuxViewChecker(Front3DCameraSampler):
    def __init__(self, config):
        Front3DCameraSampler.__init__(self, config)
    
    def run(self):
        super().run()
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
        """
        creates a viewlist_{id}.txt for each view in the data; contains ids of other images that have an overlapping view
        """
        cam_ob = bpy.context.scene.camera
        cam = cam_ob.data
        
        self.sqrt_number_of_rays = config.get_int("sqrt_number_of_rays", 10)
        self.min_common_pixels = config.get_float("min_common_pixels", 0.2)

        self._set_cam_intrinsics(cam, config)

        all_views_in_scene = sorted([f for f in os.listdir(config.get_string("scene_dir")) if f.startswith("campose")])
        print("scene dir: ", config.get_string("scene_dir"))
        print("views: ", all_views_in_scene)

        for campose_name in all_views_in_scene:
            viewlist = []
            for other_campose in all_views_in_scene:
                campose_orig = np.load(os.path.join(config.get_string("scene_dir"), campose_name))
                campose_other = np.load(os.path.join(config.get_string("scene_dir"), other_campose))

                if not campose_orig["room_id"] == campose_other["room_id"]:
                    continue
                
                matrix_orig = Matrix(campose_orig["blender_matrix"])
                matrix_other = Matrix(campose_other["blender_matrix"])

                if not self._enough_common_pixels(cam, matrix_other, matrix_orig):
                    continue

                other_id = other_campose.split("_")[1].split(".")[0]
                viewlist.append(other_id + '\n')
            
            this_id = campose_name.split("_")[1].split(".")[0]
            filename = os.path.join(config.get_string("scene_dir"), f"viewlist_{this_id}.txt")
            with open(filename, 'w') as f:
                print(filename)
                print(viewlist)
                f.writelines(viewlist)
            
