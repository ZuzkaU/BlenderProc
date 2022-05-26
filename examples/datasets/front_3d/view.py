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
    front_3D_texture_path=args.front_3D_texture_path,
    label_mapping=mapping
)

# Init sampler for sampling locations inside the loaded front3D house
# point_sampler = bproc.sampler.Front3DPointInRoomSampler(loaded_objects)

# Init bvh tree containing all mesh objects
bvh_tree = bproc.object.create_bvh_tree_multi_objects([o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)])

poses = 0
tries = 0


def check_name(name):
    for category_name in ["chair", "sofa", "table", "bed"]:
        if category_name in name.lower():
            return True
    return False


# filter some objects from the loaded objects, which are later used in calculating an interesting score
special_objects = [obj.get_cp("category_id") for obj in loaded_objects if check_name(obj.get_name())]

# returns height, loxation, rotation
def get_coords_dummy():
	height = 2
	location = np.array([2, 2, height])
	rotation = [1.3, 0, 2]
	return location, rotation

def get_coords_fixed(f):
	data = np.load(f)
	print(data["room_id"])
	print(data["intrinsic"])
	print(data["fov_y"])
	print(data["fov_x"])
	print(data["camera2world"])
	print(data["blender_rotation_euler"])
	print(data["blender_matrix"])
	print(data["blender_location"])
	print("_______________________________________________________________________")
	print("_______________________________________________________________________")
	print(bproc.math.build_transformation_mat(data["blender_location"], data["blender_rotation_euler"]))
	print("vs...")
	print(data["camera2world"])
	return data["camera2world"]

def add_view(location, rotation):
	cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
	bproc.camera.add_camera_pose(cam2world_matrix)

def get_pos(f):
	data = np.load(f)
	return data["blender_location"], data["blender_rotation_euler"], data["intrinsic"]

for i in range(9):
	loc, r, K = get_pos("../panoptic-reconstruction/data/front3d/359821fc-7594-4482-91c6-51a89cefe2b6/campose_000" + str(i+1) + ".npz")
	bproc.camera.set_intrinsics_from_K_matrix(K, 320, 240)
	for j in range(1):
		rot = np.random.uniform(0, np.pi * 2)
		rotation  = r
		cam2world_matrix = bproc.math.build_transformation_mat(loc, rotation)
		bproc.camera.add_camera_pose(cam2world_matrix)

location, rotation = get_coords_dummy()
#add_view(location, rotation)

#add_view(np.array([2.3,2.3,1.9]), [1.25, 0, 0])
#add_view(np.array([2.1,2.3,1.8]), [1.28, 0, 4])

# Also render normals
# bproc.renderer.enable_normals_output()

print("_________________START RENDER_______________")

# render the whole pipeline
bproc.renderer.set_output_format("PNG")
data = bproc.renderer.render()
# data.update(bproc.renderer.render_segmap(map_by="class"))

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)
