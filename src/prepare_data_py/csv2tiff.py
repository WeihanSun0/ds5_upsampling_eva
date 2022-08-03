from wsgiref.simple_server import demo_app
import numpy as np
import pandas as pd
import scipy as sp
import cv2 as cv
import math

folder = "./dat/real/hand_test"
spot_depth_file = f"{folder}/%08d_spot_depth.csv" % 7
spot_coord_file = f"{folder}/%08d_spot_coor.csv" % 7

fx = 374.8871819319296
fy = 374.3636185723908
cx = 484.7018651481662
cy = 268.95819709962416



if __name__ == '__main__1':
    print(spot_coord_file)
    print(spot_depth_file)
    a = pd.read_csv(spot_coord_file, header=None) # coordinate
    b = pd.read_csv(spot_depth_file, header=None) # z
    a_np = a.to_numpy() / 100
    b_rep = b.replace('-nan(ind)', np.NaN)
    b_np = b_rep.to_numpy()
    a_fin = a_np.reshape(12, 12, 2)
    b_fin = b_np.astype('float64')
    c = np.zeros((12, 12, 3), dtype='float64') 
    c[:,:,:2] = a_fin
    c[:,:, 2] = b_fin
    cv.imwrite(f"{folder}/tmp1.tiff", c) 
    cv.imwrite(f"{folder}/tmp2.tiff", c.astype(np.float32))
    # cv.imwrite(f"floder/tmp3.exr", c)
    read_tmp1 = cv.imread(f"{folder}/tmp1.tiff", -1)
    read_tmp2 = cv.imread(f"{folder}/tmp2.tiff", -1)
    exit(0)

def pc2depth(pc, guide, cx, cy, fx, fy):
    depth = np.zeros(guide.shape, np.float64)
    height, width = guide.shape
    h, w, c = pc.shape
    for r in range(h):
        for c in range(w):
            X = pc[r, c, 0]
            Y = pc[r, c, 1]
            Z = pc[r, c, 2]
            if Z == 0.0:
                continue
            fpx = X * fx / Z + cx
            fpy = Y * fy / Z + cy
            ipx = round(fpx)
            ipy = round(fpy)
            if ipx < 0 or ipx >= width or ipy < 0 or ipy >= height:
                continue
            depth[ipy, ipx] = Z
    return depth

def drawSparseDepth(guide, depth):
    mask = depth.clone()
    mask[mask!=0.0] = 1.0


if __name__ == '__main__':
    fn_guide = f"{folder}/00000005_rgb_gray_img.png"
    fn_spot_pc = f"{folder}/00000005_spot_depth_pc.tiff"
    # fn_flood_pc = f"{folder}/00000000_flood_depth_pc.tiff"
    fn_spot_coord = f"{folder}/00000005_spot_coor.csv"
    fn_spot_depth = f"{folder}/00000005_spot_depth.csv"
    guide_img = cv.imread(fn_guide, -1)
    spot_pc = cv.imread(fn_spot_pc, -1)
    # flood_pc = cv.imread(fn_flood_pc, -1)
    spot_dmap = pc2depth(spot_pc, guide_img, cx, cy, fx, fy)
    spot_coord = np.loadtxt(fn_spot_coord, delimiter=',')
    spot_depth = np.loadtxt(fn_spot_depth, delimiter=',')
    # flood_dmap = pc2depth(flood_pc, guide_img, cx, cy, fx, fy)
    exit(0)