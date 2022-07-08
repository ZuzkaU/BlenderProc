import os, sys, json

scene = sys.argv[1]
output = sys.argv[2]

with open(os.path.join(output, "view_dispatch.json")) as f:
    views = json.load(f)

for v in views:
    view_dir = os.path.join(scene, v)
    os.makedirs(view_dir, exist_ok=True)
    for i, s in enumerate(v):
        if i == 0:
            os.rename(os.path.join(output, "rgb_"+str(s).zfill(4)+".png"), os.path.join(view_dir, "original_"+str(s).zfill(4)+".png"))
        else:
            os.rename(os.path.join(output, "rgb_"+str(s).zfill(4)+".png"), os.path.join(view_dir, "rgb_"+str(s).zfill(4)+".png"))
            os.rename(os.path.join(output, "mask_"+str(s).zfill(4)+".png"), os.path.join(view_dir, "mask_"+str(s).zfill(4)+".png"))
