import blenderproc as bproc
import argparse
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument("front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument("output_dir", help="Path to where the data should be saved")
args = parser.parse_args()

if not os.path.exists(args.front) or not os.path.exists(args.future_folder):
    raise Exception("One of the two folders does not exist!")

bproc.init()
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=50, glossy_bounces=50, max_bounces=50,
                                  transmission_bounces=50, transparent_max_bounces=50)
bproc.renderer.set_max_amount_of_samples(512)

# load the front 3D objects
loaded_objects = bproc.loader.load_front3d(
    json_path=args.front,
    future_model_path=args.future_folder,
    front_3D_texture_path="",#args.front_3D_texture_path,
    label_mapping=mapping
)

# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

def get_coords_fixed(f):
	data = np.load(f)
	return data["camera2world"]

def add_view(location, rotation):
	cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
	bproc.camera.add_camera_pose(cam2world_matrix)

def get_pos(f):
	data = np.load(f)
	return data["blender_location"], data["blender_rotation_euler"], data["intrinsic"]

for i in range(9):
	loc, r, K = get_pos("../../sample/359821fc-7594-4482-91c6-51a89cefe2b6/campose_000" + str(i+1) + ".npz")
	bproc.camera.set_intrinsics_from_K_matrix(K, 320, 240)
	cam2world_matrix = bproc.math.build_transformation_mat(loc, r)
	bproc.camera.add_camera_pose(cam2world_matrix)


# render the whole pipeline
data = bproc.renderer.render(output_dir=args.output_dir)

