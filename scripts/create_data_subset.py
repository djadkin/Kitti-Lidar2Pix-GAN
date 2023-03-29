# Create subsets of the dataset filtered 
# on the number of points in an objects cloud 
import os
import numpy as np
import shutil

IMG_DIR = './data/processed/image_2/'
LABEL_DIR = './data/processed/label_2' 
PC_DIR = './data/processed/point_clouds/'
CALIB_DIR = './data/processed/calib/'
CENTERS_DIR = './data/processed/centers/'

MIN_POINTS = 100

NEW_PC_DIR = './data/processed/{}/training/point_clouds/'.format(MIN_POINTS)
NEW_IMG_DIR = './data/processed/{}/training/image_2/'.format(MIN_POINTS)
NEW_CENTERS_DIR = './data/processed/{}/training/centers/'.format(MIN_POINTS)
NEW_CALIB_DIR = './data/processed/{}/training/calib/'.format(MIN_POINTS)

cloud_counts = []
counter = 0
for index in range(8789):
    id_str = f'{"0"*(6 - len(str(index)))}{index}'
    
    # load point cloud
    bin_path = os.path.join(PC_DIR, f'{id_str}.bin')
    point_cloud = np.fromfile(bin_path, dtype=np.float32)
    point_cloud = point_cloud.reshape((-1, 4))[:, 0:3]
    #print(point_cloud.shape[0])
    # count points
    n_points = point_cloud.shape[0]
    if  n_points >= MIN_POINTS:
        # save point cloud, img, centers, and calib
        
        # point cloud
        dst = os.path.join(NEW_PC_DIR, 
                        f'{"0"*(6 - len(str(counter)))}{counter}.bin')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(bin_path, dst)
        
        # img
        src = os.path.join(IMG_DIR, 
                        f'{"0"*(6 - len(str(index)))}{index}.png')
        dst = os.path.join(NEW_IMG_DIR, 
                        f'{"0"*(6 - len(str(counter)))}{counter}.png')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        
        # centers
        src = os.path.join(CENTERS_DIR, 
                        f'{"0"*(6 - len(str(index)))}{index}.npy')
        dst = os.path.join(NEW_CENTERS_DIR, 
                        f'{"0"*(6 - len(str(counter)))}{counter}.npy')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        
        # calib
        src = os.path.join(CALIB_DIR, 
                        f'{"0"*(6 - len(str(index)))}{index}.txt')
        dst = os.path.join(NEW_CALIB_DIR, 
                        f'{"0"*(6 - len(str(counter)))}{counter}.txt')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        
        counter += 1