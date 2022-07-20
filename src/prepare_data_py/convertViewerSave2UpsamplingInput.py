""" convertion for data saved by DSViewer 1.3 to Upsampling input
"""

import numpy as np
import cv2 as cv
from torch import saddmm
from readData import DataSet, GetParam, PosMapping
import os
import shutil

folder_root = "./"
folderIn = "dat/real/fromViewer_hand"
folderOut = "dat/real/UpIn_hand"
folderParam = "dat/real/camParam"
src_folder = f"{folder_root}{folderIn}"
dst_folder = f"{folder_root}{folderOut}"
param_folder= f"{folder_root}{folderParam}"


if not os.path.exists(dst_folder):
    os.mkdir(dst_folder)

intensity_shape = (240, 320)

if __name__ == '__main__':
    #* get param ob
    gp = GetParam(param_folder)
    gp.read_camera_params()
    #* pos mapping ob
    pm = PosMapping(gp.transCalib)
    #* get data obj
    gd_obj = DataSet(f"{src_folder}")
    
    #* processing
    for i in range(len(gd_obj.rgb_list)):
        rgb, floodCsv, spotCsv, spotCoor, intensity = gd_obj.get_data()
        if rgb is not None:
            cv.imwrite("%s/%05d_%s" % (dst_folder, i, "s0.png"), rgb)
        if floodCsv is not None:
            #* mapping to intensity shape
            floodMat, floodMask = pm.map_flood_2_intensitySize(floodCsv, intensity_shape) 
            #* undistortion flood
            undist_floodMat, undist_floodMask = gp.tofCalib.undistortion_coord(floodMat, floodMask)
            floodMat = undist_floodMat
            floodMask = undist_floodMask
            #* to pointcloud
            flood_pc = gp.tofCalib.dm2pc(floodMat, floodMask)
            #* point cloud translation (R,T)
            rgb_flood_pc = gp.transCalib.translate_pcs(flood_pc)
            #* point cloud -> depthmap
            flood_dmap, flood_dmapMask = gp.rgbCalib.pc2dm(rgb_flood_pc, rgb.shape[:2])

            #* mapping depthmap with RGB (test)
            norm_flood_dmap = cv.normalize(flood_dmap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            color_flood_dmap = cv.applyColorMap(norm_flood_dmap.astype('uint8'), cv.COLORMAP_JET)
            rgbMapping_flood = pm.map_depthmap_2_rgb(color_flood_dmap, rgb, flood_dmapMask, size=1)
            cv.imshow("flood_rgb_mapping", rgbMapping_flood)
            cv.waitKey(0)
