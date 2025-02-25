import blenderproc as bproc
import argparse
import os
import numpy as np
from mathutils import Euler

MAX_SCENES = 1
MAX_IN_SCENE = 3
ADDITIONAL_CAMERA_POSITIONS = 3
EXCLUDED_IDS = [95,93,49,21,24,79,31,39,60,38,98,58,44,7,40,63]
EXCLUDED_LABELS = ['wallouter','wallbottom','walltop','pocket','slabside','slabbottom',
				   'slabtop','front','back','baseboard','door','window','baywindow',
				   'hole','wallinner','beam']
EXCLUDED_LABELS = [] # can this be the problem that it doesn't look correctly?
#TODO: how about labels newly added from the previous version?

def get_forward(rotation_euler):
	r = np.array(Euler(rotation).to_matrix())
	forward = r @ [0, 0, -1]
	return forward


proximity_checks = {"min": 0.5, "avg": {"min": 2.0, "max": 4.0}, "no_background": True}
def sample_camera_pose(initial_location, initial_rotation, bvh_tree, num_tries):
	for i in range(num_tries):
		# assume that the objects have an average distance 2.0 from camera
		poi = initial_location + 2 * get_forward(initial_rotation) + np.random.normal([0, 0, 0], 0.3)
		#return poi, initial_rotation
		random_direction = np.random.rand(3)
		random_direction /= np.linalg.norm(random_direction)
		random_direction = -2 * get_forward(initial_rotation) + np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
		# + -2 * get_forward(initial_rotation))
		#random_direction = random_direction / (0.5 * np.linalg.norm(v))
		location = poi + random_direction #TODO: random location in a circle
		#location = initial_location + np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
		
		rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
		return location, rotation_matrix
		cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
		if bproc.camera.scene_coverage_score(cam2world_matrix) > 0.4 and bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, proximity_checks, bvh_tree):
			return location, rotation
	return None

def add_pose_and_write(location, rotation, output_file):
	cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
	bproc.camera.add_camera_pose(cam2world_matrix)
	if not os.path.exists(os.path.dirname(output_file)):
		os.makedirs(os.path.dirname(output_file))
	with open(output_file, 'w') as f:
		f.write(f"location:\n{location}\nrotation:\n{rotation}")


parser = argparse.ArgumentParser()
parser.add_argument("front", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
# textures are not included in Panoptic Reconstruction
# parser.add_argument("front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument("panoptic", help="Path to the 3D front dataset from Panoptic Reconstruction (to render the same poses)")
parser.add_argument("output_dir", help="Path to where the data should be saved")
args = parser.parse_args()

if not os.path.exists(args.front) or not os.path.exists(args.future_folder) or not os.path.exists(args.panoptic):
    raise Exception("One of the folders does not exist!")

# Init & load object mapping
bproc.init()
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

# Set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=50, glossy_bounces=50, max_bounces=50,
                                  transmission_bounces=50, transparent_max_bounces=50)
bproc.renderer.set_max_amount_of_samples(512)


# List the scenes
scenes = sorted(os.listdir(args.front))
num_scenes = 0
for scene in scenes:
	scene_path = os.path.join(args.front, scene)
	scene_in_panoptic = os.path.join(args.panoptic, scene.split('.')[0])
	if not os.path.exists(scene_in_panoptic):
		#print(f"This scene doesn't exist in the panoptic dataset: {scene_in_panoptic}")
		continue
	num_scenes += 1
	if num_scenes > MAX_SCENES:
		break

    # Load the front 3D objects
	loaded_objects = bproc.loader.load_front3d(
		json_path=scene_path,
		future_model_path=args.future_folder,
		front_3D_texture_path="",
		label_mapping=mapping,
		ceiling_light_strength=1.0
	)

	objects = [o for o in loaded_objects if mapping.label_from_id(o.get_cp("category_id")) not in EXCLUDED_LABELS and isinstance(o, bproc.types.MeshObject)]
	bvh_tree = bproc.object.create_bvh_tree_multi_objects(objects)
	
	# Get a view from Panoptic dataset
	view_nums = sorted([v[7:].split('.')[0] for v in os.listdir(scene_in_panoptic) if v.startswith("campose")])
	for view_num in view_nums[:MAX_IN_SCENE]:
		view_path = os.path.join(scene_in_panoptic, "campose" + view_num + ".npz")
		output_dir = os.path.join(args.output_dir, scene.split('.')[0], "view" + view_num)
		data = np.load(view_path)
		bproc.camera.set_intrinsics_from_K_matrix(data["intrinsic"], 320, 240) # TODO: are all intrinsic matrices the same? -> set after init
		#bproc.renderer.enable_depth_output(activate_antialiasing=False, output_dir=output_dir)
		
		add_pose_and_write(data["blender_location"], data["blender_rotation_euler"], os.path.join(output_dir, 'campose_0000.txt'))
		
		for i in range(ADDITIONAL_CAMERA_POSITIONS):
			location = data["blender_location"] + np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
			rotation = data["blender_rotation_euler"] + np.random.uniform([-0.1, 0, -0.2], [0.1, 0, 0.2])
			location, rotation = sample_camera_pose(data["blender_location"], data["blender_rotation_euler"], None, 1)
			add_pose_and_write(location, rotation, os.path.join(output_dir, 'campose_' + str(i + 1).zfill(4) +'.txt'))
			continue
			height = np.random.uniform(1.4, 1.8)
			location = point_sampler.sample(height)
			rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])
			cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
			bproc.camera.add_camera_pose(cam2world_matrix)
			# TODO: find some object position, sample camera randomly so that it looks at the object
			# TODO: render RGB image

		####bproc.renderer.set_output_format("PNG")
		bproc.renderer.render(output_dir=output_dir, return_data=False)
		# WTF!! when return_data=False, it outputs only rgb, and when return_data=True, it outputs only depth?????
		
		####data.update(bproc.renderer.render_segmap(map_by="class"))
		####bproc.writer.write_hdf5(output_dir, data, append_to_existing_output=True)
		bproc.utility.reset_keyframes()






























