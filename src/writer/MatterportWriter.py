from pathlib import Path
import bpy
import numpy as np
import png
from src.writer.WriterInterface import WriterInterface

class MatterportWriter(WriterInterface):
    def __init__(self, config):
        super().__init__(config)

        self.depth_scale = 0.25
        self.base_path = Path(super()._determine_output_dir())

    def run(self):
        cam_ob = bpy.context.scene.camera
        self.cam = cam_ob.data
        self.cam_pose = (self.cam, cam_ob)

        # load frame mapping
        frame_mapping = {}
        with open(self.base_path / "frame_names.txt") as f:
            content = f.readlines()
            for line in content:
                parts = line.strip().split()
                key = int(parts[0])
                value = parts[1]
                frame_mapping[key] = value

        # convert distance to depth
        for frame_id in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
            # Activate frame.
            bpy.context.scene.frame_set(frame_id)

            dist_output = self._find_registered_output_by_key("distance")
            if dist_output is None:
                raise Exception("Distance image has not been rendered.")
            depth, _, _ = self._load_and_postprocess(dist_output['path'] % frame_id, "distance")
            depth_mm = 1000.0 * depth  # [m] -> [mm]
            depth_mm_scaled = depth_mm / float(self.depth_scale)

            # Save the scaled depth image.
            frame_name = frame_mapping[frame_id]
            parts = frame_name.split("_")
            depth_name = f"{parts[0]}_d{parts[1]}_{parts[2]}"
            depth_output_path = str(self.base_path / (depth_name + ".png"))
            save_depth(depth_output_path, depth_mm_scaled)

            # rename rgb, normal, seg
            rgb_frame = Path(self.base_path / f"rgb_{frame_id:04d}.png")
            rgb_destination = Path(self.base_path / f"{parts[0]}_c{parts[1]}_{parts[2]}.png")
            rgb_frame.rename(rgb_destination)

            distance_frame = Path(self.base_path / f"distance_{frame_id:04d}.exr")
            distance_destination = Path(self.base_path / f"{parts[0]}_dist{parts[1]}_{parts[2]}.exr")
            distance_frame.rename(distance_destination)

            normal_frame = Path(self.base_path / f"normals_{frame_id:04d}.exr")
            normal_destination = Path(self.base_path / f"{parts[0]}_n{parts[1]}_{parts[2]}.exr")
            normal_frame.rename(normal_destination)

            seg_frame = Path(self.base_path / f"seg_{frame_id:04d}.exr")
            seg_destination = Path(self.base_path / f"{parts[0]}_seg{parts[1]}_{parts[2]}.exr")
            seg_frame.rename(seg_destination)

            segmap_frame = Path(self.base_path / f"segmap_{frame_id:04d}.npz")
            segmap_destination = Path(self.base_path / f"{parts[0]}_segmap{parts[1]}_{parts[2]}.npz")
            segmap_frame.rename(segmap_destination)


def save_depth(path, im):
    """Saves a depth image (16-bit) to a PNG file.
    From the BOP toolkit (https://github.com/thodan/bop_toolkit).

    :param path: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    """
    if not path.endswith(".png"):
        raise ValueError('Only PNG format is currently supported.')

    im[im > 65535] = 0
    im_uint16 = np.round(im).astype(np.uint16)

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    with open(path, 'wb') as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))