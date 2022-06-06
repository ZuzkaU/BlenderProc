import blenderproc as bproc
import argparse
import os
import numpy as np

MAX_SCENES = 1
MAX_IN_SCENE = 10
ADDITIONAL_CAMERA_POSITIONS = 0
EXCLUDED_IDS = [95,93,49,21,24,79,31,39,60,38,98,58,44,7,40,63]
EXCLUDED_LABELS = ['wallouter','wallbottom','walltop','pocket','slabside','slabbottom',
				   'slabtop','front','back','baseboard','door','window','baywindow',
				   'hole','wallinner','beam']
EXCLUDED_LABELS = [] # can this be the problem that it doesn't look correctly?
#TODO: how about labels newly added from the previous version?

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
		bproc.renderer.enable_depth_output(activate_antialiasing=False, output_dir=output_dir)
		
		cam2world_matrix = bproc.math.build_transformation_mat(data["blender_location"], data["blender_rotation_euler"])
		bproc.camera.add_camera_pose(cam2world_matrix)
		
		for i in range(ADDITIONAL_CAMERA_POSITIONS):
			location = data["blender_location"] + np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
			rotation = data["blender_rotation_euler"] + np.random.uniform([-0.1, 0, -0.2], [0.1, 0, 0.2])
			cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
			bproc.camera.add_camera_pose(cam2world_matrix)
			continue
			height = np.random.uniform(1.4, 1.8)
			location = point_sampler.sample(height)
			rotation = np.random.uniform([1.2217, 0, 0], [1.338, 0, np.pi * 2])
			cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)
			bproc.camera.add_camera_pose(cam2world_matrix)
			# TODO: find some object position, sample camera randomly so that it looks at the object
			# TODO: render RGB image

		####bproc.renderer.set_output_format("PNG")
		#print(f"__________rendering {output_dir}")
		data = bproc.renderer.render(output_dir=output_dir, return_data=False)
		####data.update(bproc.renderer.render_segmap(map_by="class"))
		####bproc.writer.write_hdf5(output_dir, data, append_to_existing_output=True)
		bproc.utility.reset_keyframes()






























