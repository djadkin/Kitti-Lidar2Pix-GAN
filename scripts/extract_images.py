import os
import tqdm
import pickle  
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import math
from skimage import io
from IPython.display import display, clear_output
from pathlib import Path
from kitti_utils import get_data, get_object_centers, \
get_point_idxs, point_in_cube, load_kitti_calib, \
camera_coordinate_to_point_cloud

# input dirs
IMG_DIR = './data/raw/image_2/'
LABEL_DIR = './data/raw/label_2' 
POINT_CLOUD_DIR = './data/raw/velodyne/'
CALIB_DIR = './data/raw/calib/'

# output dir
OUTPUT_DIR = './data/processed'

# output image dimensions
lim_h = 128
lim_w = 128

dicts_path = os.path.join(OUTPUT_DIR, 'data_dicts')
children_output_paths = ['data_dicts', 'image_2', 'point_clouds', 'centers', 'calib']

for child_path in children_output_paths:
    out_path = Path(os.path.join(OUTPUT_DIR, child_path))
    out_path.mkdir(parents=True, exist_ok=True)

# Crop objects from the images and save
def crop_objects(save_dicts = True, save_objects=True):
    """ Crops lim_h by lim_w sized images of cars and their point clouds
        Saves pickled dicts for quicker repeated processing
    """
    out_id = 0
    for file_id in tqdm.tqdm(range(0, 7481)):
        data = get_data(file_id, IMG_DIR, LABEL_DIR, POINT_CLOUD_DIR, CALIB_DIR)
        if save_dicts:

            DICTS_DIR = os.path.join(OUTPUT_DIR,
                                     'data_dicts/{0:06d}.bin'.format(file_id)
                                    )

            with open(DICTS_DIR, 'wb') as f:
                pickle.dump(data, f)

        if save_objects:
            labels = data['labels']
            for L in range(len(labels)):
                line = labels[L]
                line = line.strip('\n').split()

                label = line[0]
                trunc = float(line[1])
                occlu = int(line[2])

                # center location of the object (3D)
                cx, cy, cz = [float(line[i]) for i in range(11, 14)]

                # Points2Pix cars condition
                # Can be changed to other objects and conditions
                if (label=="Car" and np.sqrt(cx**2 + cy**2 + cz**2) < 60 and 
                    math.isclose(trunc, 0) and occlu < 1):
                    [c_h, c_w] = data['obj_centers'][L] # 2D

                    idxs = data['pc_obj_idxs'][L]

                    image = data['image']
                    img_h = image.shape[0]
                    img_w = image.shape[1]
                    
                    if (int(c_h-lim_h) > 0 and int(c_h+lim_h) < img_h and
                        int(c_w-lim_w) > 0 and int(c_w+lim_w) < img_w) and idxs.shape[0]:
                        clear_output(wait=True)
                        pc = data['point_cloud']
                        
                        # save partial point cloud
                        pc_partial = pc[idxs, :]
                        pc_partial = pc_partial.reshape(pc_partial.shape[0]*pc_partial.shape[1])
                        pc_partial.tofile(os.path.join(OUTPUT_DIR, 'point_clouds/{0:06d}.bin'.format(out_id)))

                        # save partial image
                        img_partial = image[int(c_h-lim_h):int(c_h+lim_h), int(c_w-lim_w):int(c_w+lim_w)]
                        
                        assert img_partial.shape == (256, 256, 3) 
                        
                        # save cropped image to png
                        io.imsave(os.path.join(OUTPUT_DIR, 'image_2/{0:06d}.png'.format(out_id)), img_partial)

                        np.save(os.path.join(OUTPUT_DIR, 'centers/{0:06d}.npy'.format(out_id)), data['obj_centers'][L])

                        with open(os.path.join(CALIB_DIR, '{0:06d}.txt'.format(file_id)), 'r') as f:
                            g = open(os.path.join(OUTPUT_DIR, 'calib/{0:06d}.txt'.format(out_id)), "x")
                            for line in f.readlines():
                                g.write(line)
                        out_id += 1

if __name__ == "__main__":
    crop_objects()
