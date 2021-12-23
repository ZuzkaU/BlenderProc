import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import bpy

from src.loader.LoaderInterface import LoaderInterface
from src.utility.BlenderUtility import create_color_palette
from src.utility.Config import Config
from src.utility.Utility import Utility


class MatterportLoader(LoaderInterface):
    def __init__(self, config: Config):
        super().__init__(config)

        self.house_path = Path(Utility.resolve_path(self.config.get_string("house_path")))

        self.loaded_regions = []
        self.objects_per_region = defaultdict(list)
        self.unique_instances = set()
        self.unique_instances_old_mapping = set()
        self.instance_mapping = defaultdict(list)

        self.category_mapping = self.load_nyu_mapping(Path("resources/matterport/matterport_nyu_mapping.tsv"))

    def run(self):
        self.load_regions()
        self.load_segmentations()

        # debug: save instance mapping
        instance_mapping_path = os.path.join(self._output_dir, "instance_mapping.json")
        with open(instance_mapping_path, "w") as f:
            json.dump(self.instance_mapping, f, indent=4)

    def load_regions(self):
        regions_path = self.house_path / "region_segmentations"

        # region_mesh_files = [file for file in regions_path.iterdir() if file.suffix == ".ply" and file.stem == "region11"]
        region_mesh_files = [file for file in regions_path.iterdir() if file.suffix == ".ply" and "reduced" in file.name]
        region_mesh_files = sorted(region_mesh_files, key=lambda x: int(x.stem.replace("region", "").replace(".reduced", "")))
        print([file.name for file in region_mesh_files])

        for region_mesh_file in region_mesh_files:
            region_id = self.parse_region_name(region_mesh_file.stem)

            mesh = Utility.import_objects(filepath=str(region_mesh_file))[0]
            mesh.name = region_mesh_file.stem.replace(".reduced", "")
            mesh["region_id"] = region_id

            self.loaded_regions.append(mesh)
            self.load_segmentation(self.house_path / "region_segmentations", mesh)

    def load_segmentations(self):
        segmentation_path = self.house_path / "region_segmentations"

        # for region in self.loaded_regions:
        #     self.load_segmentation(segmentation_path, region)

    def load_segmentation(self, folder, region):
        # load segments
        t1 = time.time()

        fsegs_file_path = folder / (region.name + ".fsegs.json")  # assemble filepath
        face_segments = json.load(open(fsegs_file_path))["segIndices"]  # open fsegs.json, parse segIndices

        # parse segGroups for region
        semsegs_file_path = folder / (region.name + ".semseg.json")

        # fix \o decoding bug
        text = open(semsegs_file_path).readlines()

        for line_number, line in enumerate(text):
            if r"\o" in line:
                replaced = line.replace(r"\o", r"/ o")
                text[line_number] = replaced

        content = "\n".join(text)

        segmentation_groups = json.loads(content)["segGroups"]  # open semsegs.json

        color_palette = create_color_palette()

        region_id = region["region_id"]

        segment_to_object_id = {}
        object_id_to_label = {}
        object_id_to_material_index = {}
        for group in segmentation_groups:
            object_id = group["objectId"]

            # parse id, objectId, label, segments
            object_id_to_label[object_id] = group["label"]

            # add material
            material = bpy.data.materials.new(name=f"region_{region_id}_instance_{object_id}")
            material.use_nodes = True
            # color_index = region_id % (len(color_palette)-1)# debug
            color_index = object_id % (len(color_palette)-1)# debug
            color = [c / 255 for c in color_palette[color_index]]  # (object_id, object_id, object_id)
            nodes = material.node_tree.nodes
            links = material.node_tree.links
            emission_node = nodes.new(type='ShaderNodeEmission')
            output = Utility.get_the_one_node_with_type(nodes, 'OutputMaterial')

            emission_node.inputs['Color'].default_value[:3] = color
            links.new(emission_node.outputs['Emission'], output.inputs['Surface'])
            material_index = len(region.data.materials)
            region.data.materials.append(material)

            object_id_to_material_index[object_id] = material_index

            # mapping segment id -> object-id
            for segment in group["segments"]:
                segment_to_object_id[segment] = object_id

        # iterate over faces, assign object-id
        for face_index, face in enumerate(region.data.polygons):
            segment_id = face_segments[face_index]
            if segment_id in segment_to_object_id:
                mapped_object_id = segment_to_object_id[segment_id]
                mapped_material_index = object_id_to_material_index[mapped_object_id]
                face.material_index = mapped_material_index

        # split objects by object ids
        bpy.ops.object.select_all(action='DESELECT')
        region.select_set(True)
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.separate(type="MATERIAL")  # order not deterministic
        bpy.ops.object.editmode_toggle()

        parent_node = bpy.data.objects.new(f"{region_id}", None)
        parent_node["region_id"] = region_id
        parent_node["is_region"] = True
        bpy.context.scene.collection.objects.link(parent_node)

        # assign labels, room ids, object id
        # sort by object_id
        selected_objects = [o for o in bpy.context.selected_objects]
        sorted_selected_objects = sorted(selected_objects, key=lambda x: int(x.data.materials[0].name.split("_")[-1]))

        # debug only: reproduce old instance ids
        for object_mesh in bpy.context.selected_objects:
            # parse material name to get original object id
            current_instance_id = len(self.unique_instances_old_mapping)
            object_id = int(object_mesh.data.materials[0].name.split("_")[-1])
            instance_identifier = f"region{region_id:02d}.{object_id:04d}"
            self.unique_instances_old_mapping.add(instance_identifier)
            self.instance_mapping[instance_identifier].append(current_instance_id)

        for object_mesh in sorted_selected_objects:
            # parse material name to get original object id
            object_id = int(object_mesh.data.materials[0].name.split("_")[-1])
            object_mesh["region_id"] = region_id
            object_mesh.parent = parent_node

            current_instance_id = len(self.unique_instances)
            object_mesh["instanceid"] = current_instance_id
            object_mesh["object_id"] = object_id
            category_label = object_id_to_label[object_id]
            object_mesh["category"] = category_label
            category_id = self.category_mapping.get(category_label, 40)
            object_mesh["cp_category_id"] = category_id
            self.objects_per_region[region_id].append(object_mesh)

            instance_identifier = f"region{region_id:02d}.{object_id:04d}"
            self.unique_instances.add(instance_identifier)
            object_mesh.name = instance_identifier
            # print(object_mesh.name, current_instance_id, category_id)

            # debug
            self.instance_mapping[instance_identifier].append(current_instance_id)

        t2 = time.time()
        print('Blender segmentation in %.3fs' % (t2 - t1))




    def load_nyu_mapping(self, path: Path):
        mapping = {}

        with open(path) as f:
            reader = csv.DictReader(f, dialect="excel-tab")

            for row in reader:
                key = row["raw_category"]
                value = row["nyu40id"]
                mapping[key] = value

        return mapping

    @staticmethod
    def parse_region_name(region_name):
        region_id = int(region_name.replace("region", "").replace(".reduced", ""))

        return region_id
