import numpy as np
import pandas as pd
import scipy as sp
import cv2 as cv

folder = "./dat/real/hand_test"
spot_depth_file = f"{folder}/%08d_spot_depth.csv" % 7
spot_coord_file = f"{folder}/%08d_spot_coor.csv" % 7

if __name__ == '__main__':
    print(spot_coord_file)
    print(spot_depth_file)
    a = pd.read_csv(spot_coord_file, header=None) # coordinate
    b = pd.read_csv(spot_depth_file, header=None) # z
    a_np = a.to_numpy()
    b_rep = b.replace('-nan(ind)', np.NaN)
    b_np = b_rep.to_numpy()
    a_fin = a_np.reshape(12, 12, 2)
    b_fin = b_np.astype('float64')
    c = np.zeros((12, 12, 3), dtype='float64') 
    c[:,:,:2] = a_fin
    c[:,:, 2] = b_fin
    # cv.imwrite(f"{folder}/tmp1.tiff", c) 
    # cv.imwrite(f"{folder}/tmp2.tiff", c.astype(np.float32))
    # cv.imwrite(f"floder/tmp3.exr", c)
    read_tmp1 = cv.imread(f"{folder}/tmp1.tiff", -1)
    read_tmp2 = cv.imread(f"{folder}/tmp2.tiff", -1)
    exit(0)