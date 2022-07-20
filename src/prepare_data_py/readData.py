import cv2 as cv
from cv2 import CV_8UC1
from cv2 import CV_32FC1
from matplotlib.pyplot import flag
from nbformat import read
import numpy as np
import pandas as pd
import os
import re
import open3d as o3d
import h5py
import math


class DataSet:
    """ image data
    """
    # appendix
    rgb = "rgb_img.png"
    spot = "spot_depth.csv"
    intensity = "intensity_rgb.csv"
    flood = "flood_depth.csv"
    spot_coor = "spot_coor.csv"
    # intensity size
    intensity_width = 320
    intensity_height = 240

    def __init__(self, path) -> None:
        """initialization

        Args:
            path (str): folder of data
        """
        self.path =  path
        self.rgb_list = []
        self.flood_list = []
        self.spot_list = []
        self.spotCoor_list = []
        self.intensity_list = []
        for fn in os.listdir(path):
            res =  re.match(f".*{self.rgb}", fn, flags=0)
            if res is not None:
                self.rgb_list.append(fn)
            res = re.match(f".*{self.flood}", fn, flags=0)
            if res is not None:
                self.flood_list.append(fn)
            res = re.match(f".*{self.spot}", fn, flags=0)
            if res is not None:
                self.spot_list.append(fn)
            res = re.match(f".*{self.spot_coor}", fn, flags=0)
            if res is not None:
                self.spotCoor_list.append(fn)
            res = re.match(f".*{self.intensity}", fn, flags=0)
            if res is not None:
                self.intensity_list.append(fn)
        self.rgb_iter = iter(self.rgb_list) 
        self.flood_iter = iter(self.flood_list) 
        self.spot_iter = iter(self.spot_list) 
        self.spotCoor_iter = iter(self.spotCoor_list) 
        self.intensity_iter = iter(self.intensity_list)

    def read_rgb(self) -> np.ndarray:
        """ read rgb image 
        Returns:
            np.ndarray: rgb image 
        """
        if self.rgb_iter.__length_hint__() <= 0:  # no data
            self.curr_rgbImg = None
            return None 
        fn = next(self.rgb_iter)
        self.curr_rgbImg = cv.imread(f"{self.path}/{fn}", -1)
        return self.curr_rgbImg
    
    def read_spot(self) -> np.ndarray:
        """read spot depth

        Returns:
            np.ndarray: spot depth 
        """
        if self.spot_iter.__length_hint__() <= 0:
            self.curr_spotGrid = None
            self.curr_spotCoor = None
            return None, None
        fn = next(self.spot_iter) 
        self.curr_spotGrid = np.loadtxt(f"{self.path}/{fn}", delimiter=',', \
            usecols=(range(12))) # prevent "," at the ends of lines
        fn = next(self.spotCoor_iter) 
        self.curr_spotCoor = np.loadtxt(f"{self.path}/{fn}", delimiter=',', \
            usecols=(range(24))) # prevent "," at the ends of lines
        return self.curr_spotGrid, self.curr_spotCoor
    
    def read_flood(self) -> np.ndarray:
        """read flood depth

        Returns:
            np.ndarray: flood depth 
        """
        if self.flood_iter.__length_hint__() <= 0:
            self.curr_floodGrid = None
            return None
        fn = next(self.flood_iter)
        self.curr_floodGrid = np.loadtxt(f"{self.path}/{fn}", delimiter=',', \
                    usecols=(range(80))) # prevent "," at the ends of lines
        return self.curr_floodGrid
        
    def read_intensity(self) -> np.ndarray:
        """read intensity image

        Returns:
            np.ndarray: intensity image 
        """
        if self.intensity_iter.__length_hint__() <= 0:
            self.curr_intentsityImg = None
            return None
        fn = next(self.intensity_iter)
        self.curr_intentsityImg = np.loadtxt(f"{self.path}/{fn}", delimiter=',', \
            usecols=((range(320)))) # prevent "," at then ends of lines
        return self.curr_intentsityImg
    
    def get_data(self) -> list:
        """return data frame

        Returns:
            list: [rgb, flood, spot, spotCoor, intensity]
        """
        self.read_rgb(), self.read_flood(), self.read_spot(), self.read_intensity()
        return self.curr_rgbImg, self.curr_floodGrid, self.curr_spotGrid, self.curr_spotCoor, self.curr_intentsityImg 

class PosMapping:
    """mapping depth to rgb
    """
    def __init__(self, trans) -> None:
        self.trans = trans
        pass

    def R2Z(self, src_pc) -> np.ndarray:
        """ Range to Z

        Args:
            src_pc (np.ndarray): num x 3 point cloud with (X, Y, R)

        Returns:
            np.ndarray: dst_pc 
        """
        num, dim = src_pc.shape
        dst_pc = np.zeros((num, dim))
        for idx in range(num):
            X, Y, R = src_pc[idx, :]
            Z = R 
            if R > 0.000001 and abs(X/R) < 0.999:
                theta = math.asin(X/R)
                Z = R/math.cos(theta)
            dst_pc[idx,:] = (X, Y, Z)
        return dst_pc

    def map_flood_2_intensitySize(self, flood, size) -> np.ndarray:
        """map flood points to intensity size

        Args:
            flood (np.ndarray): flood points array
            size (tuple): depthmap size (h, w)

        Returns:
            np.ndarray: floodmat 
        """
        floodmat = np.zeros(size)
        mask = np.zeros(size)
        h, w = flood.shape
        spanX, spanY = int(size[1]/w), int(size[0]/h) # should be 4
        startX, startY = int(spanX/2), int(spanY/2)
        for r in range(h): # loop flood points
            for c in range(w):
                floodmat[startY + r*spanY, startX + c*spanX] = flood[r, c]
                mask[startY + r*spanY, startX + c*spanX] = 255
        return floodmat, mask

    def map_spot_2_intensitySize(self, spot, coord, size) -> np.ndarray:
        """map spot points to intensity size 

        Args:
            spot (np.ndarray): spot points array
            coord (np.ndarray): coordinate (12, 12x2) # row, cow(x, y)
            size(tuple): depthmap size (h, w)

        Returns:
            np.ndarray: depthmap 
        """
        spotmat = np.zeros(size)
        mask = np.zeros(size)
        # spanX, spanY = 4, 4
        h, w = spot.shape
        for r in range(h): # loop spot
            for c in range(int(w)):
                x = round(coord[r, c*2])
                y = round(coord[r, c*2+1])
                spotmat[y, x] = spot[r, c]
                mask[y, x] = 255
        return spotmat, mask

    def map_depthmap_2_rgb(self, depthmap, rgbImg, mask, size=3):
        """map depthmap with rgb image

        Args:
            depthmap (np.ndarray): depth map after position translation to RGB view pointer
            rgbImg (np.ndarray): RGB image 
            mask (np.ndarray): Positions of validate depth
            size (int): point size for draw
        Returns:
            np.ndarray: mapping image 
        """
        h, w = mask.shape
        matMap = rgbImg.copy()
        for v in range(h):
            for u in range(w):
                if mask[v, u] == 0:
                    continue
                value = depthmap[v, u, :]
                cv.circle(matMap, (u, v), size, value.tolist(), thickness=-1)
        return matMap
        

class CamCalib_k5:
    """Camera calibration
    """
    def __init__(self, paramList) -> None:
        self.fx = paramList[0] 
        self.fy = paramList[1]
        self.cx = paramList[2] 
        self.cy = paramList[3] 
        self.k1 = paramList[4] 
        self.k2 = paramList[5] 
        self.k3 = paramList[6] 
        self.p1 = paramList[7] 
        self.p2 = paramList[8] 
        self.k4 = paramList[9]
        self.k5 = paramList[10]
        self.k6 = paramList[11]
        self.mtx = np.zeros((3,3), dtype=float)
        self.mtx[0, 0] = self.fx
        self.mtx[1, 1] = self.fy
        self.mtx[0, 2] = self.cx
        self.mtx[1, 2] = self.cy
        self.mtx[2, 2] = 1.0
        self.dist = (self.k1, self.k2, self.p1, self.p2, self.k3, self.k4, self.k5, self.k6)
        pass


    def undistortion0(self, img) -> np.ndarray: # for rgb image
        """undistortion(crop)

        Args:
            img (np.ndarry): distorted image

        Returns:
            np.ndarray: undistorted image 
        """
        if len(img.shape) == 3:
            (h, w, _) = img.shape
        elif len(img.shape) == 2:
            (h, w) = img.shape
        else:
            return None
        new_mtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 0, (w,h))
        new_mtx = self.mtx.copy()
        mapx, mapy = cv.initUndistortRectifyMap(self.mtx, self.dist, None, new_mtx, (w,h), CV_32FC1)
        
        # use undistort function
        # undistImg = cv.undistort(img, self.mtx, self.dist, None, new_mtx)
        # use remap function
        undistImg = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        (roix, roiy, roiw, roih) = roi
        print("roi = ", roi)
        return undistImg[roiy:roiy+roih+1, roix:roix+roiw+1]


    def undistortion1(self, img) -> np.ndarray:
        """undistortion(no crop) 

        Args:
            img (np.ndarry): distorted image

        Returns:
            np.ndarray: undistorted image 
        """
        if len(img.shape) == 3:
            (h, w, _) = img.shape
        elif len(img.shape) == 2:
            (h, w) = img.shape
        else:
            return None
        new_mtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
        mapx, mapy = cv.initUndistortRectifyMap(self.mtx, self.dist, None, new_mtx, (w,h), CV_32FC1)
        undistImg = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        return undistImg


    def undistortion_coord(self, src, mask):
        """undistortion (data position only, no interplation) 

        Args:
            img (_type_): _description_

        Returns:
            undistImg: undistortion image
            undistMask: undistortion point position (255 otherwise 0)
        """
        if len(mask.shape) == 3:
            (h, w, _) = mask.shape
        elif len(mask.shape) == 2:
            (h, w) = mask.shape
        else:
            return None
        new_mtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 0, (w,h))
        new_mtx = self.mtx.copy()
        mapx, mapy = cv.initUndistortRectifyMap(self.mtx, self.dist, None, new_mtx, (w,h), CV_32FC1)
        undistSrc = np.zeros((h, w), dtype=np.float32)
        undistMask = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                u = round(mapx[y, x]) # original x
                v = round(mapy[y, x]) # original y
                if u < 0 or u >= w or v < 0 or v >= h:
                    continue 
                if mask[v, u] != 0: # is a check point
                    undistSrc[y, x] = src[v, u]
                    undistMask[y, x] = 255
        return undistSrc, undistMask


    def dm2pc(self, img, mask) -> np.ndarray:
        """convert depthmap to point cloud
        Args:
            img (np.ndarray): depthmap
            mask (np.ndarray): positions
        Returns:
            np.ndarray: point cloud [num, 3] # meter
        """
        (h, w) = img.shape
        length = h*w
        pc = np.zeros((length, 3), dtype=np.float32)
        count = 0
        for v in range(h):
            for u in range(w):
                #? treat value as Z
                if mask[v, u] == 0:
                    continue
                Z = img[v, u]
                Y = (v - self.cy) * Z / self.fy
                X = (u - self.cx) * Z / self.fx
                pc[count, :] = [X, Y, Z]
                count += 1
        return pc[:count, :]
    

    def pc2dm(self, pc, size) -> np.ndarray:
        """convert point cloud to depthmap

        Args:
            pc (np.ndarray): point cloud [num, 3]

        Returns:
            np.ndarray: depthmap 
        """
        h, w = size
        num, dim = pc.shape
        img = np.zeros(size, dtype=np.float32)
        mask = np.zeros(size, dtype=np.uint8)
        for idx in range(num):
            X, Y, Z =  pc[idx, :]
            if Z < 0.00001 or np.isnan(Z):
            # if Z < 0.00001:
                continue
            u = round(X * self.fx / Z + self.cx)
            v = round(Y * self.fy / Z + self.cy)
            if u < 0 or u >= w or v < 0 or v >= h:
                continue
            img[v, u] = Z
            mask[v, u] = 255
        return img, mask

class TransCalib:
    """ Translation 2 cameras
    """
    def __init__(self, paramList) -> None:
        self.R = np.zeros((3, 3), dtype = np.float32)
        self.T = np.zeros((1, 3), dtype = np.float32)
        # rotation
        self.R[0,0] = paramList[0]
        self.R[1,0] = paramList[1]
        self.R[2,0] = paramList[2]
        self.R[0,1] = paramList[3]
        self.R[1,1] = paramList[4]
        self.R[2,1] = paramList[5]
        self.R[0,2] = paramList[6]
        self.R[1,2] = paramList[7]
        self.R[2,2] = paramList[8]
        # shift 
        self.T[0, 0] = paramList[9]
        self.T[0, 1] = paramList[10]
        self.T[0, 2] = paramList[11]
        pass
    
    def translate_pcs(self, src) -> np.ndarray:
        """translate point cloud (ToF->RGB)

        Args:
            src (np.ndarray): point cloud 

        Returns:
            np.ndarray: point cloud 
        """
        #? ANCHOR mm or m?
        return src.dot(self.R) + self.T/1000
        # return self.R.dot(src) + self.T/1000

    def translate_pcs1(self, src) -> np.ndarray:
        """translate point cloud (RGB->TOF)

        Args:
            src (np.ndarray): point cloud 

        Returns:
            np.ndarray: point cloud 
        """
        # return (src-self.T/100).dot(np.linalg.inv(self.R))
        return (src-self.T/1000).dot(self.R)

class GetParam:
    """read params from files
    """
    def __init__(self, folder) -> None:
        self.folder = folder
        self.camera_calib_folder = f"{folder}/camera_calib"
        self.device_calib_folder = f"{folder}/device_calib"
        pass

    def read_spot_init_coord(self)->np.ndarray:
        """get initial spot coordinate

        Returns:
            np.ndarray: 2x12x12 
        """
        with h5py.File(f"{self.device_calib_folder}/calibparams_h2d_3_1_006.h5") as f:
            return np.array(f["Params"]["spot_coor_init"])
    
    def read_camera_params(self):
        """read rgb camera and tof camera parameters, also calib parameters between them
        """
        raw = pd.read_csv(f"{self.camera_calib_folder}/param_ref.txt", delimiter="\t+", comment="/", \
                header=None, skip_blank_lines=True, skipinitialspace=True)
        self.rgbCalib = CamCalib_k5(raw.iloc[0:9, 1].tolist())
        self.tofCalib = CamCalib_k5(raw.iloc[9:18, 1].tolist())
        self.transCalib = TransCalib(raw.iloc[18:, 1].tolist())


class CamCalib_k3:
    """Camera calibration
    """
    def __init__(self, paramList) -> None:
        self.fx = paramList[0] 
        self.fy = paramList[1]
        self.cx = paramList[2] 
        self.cy = paramList[3] 
        self.k1 = paramList[4] 
        self.k2 = paramList[5] 
        self.k3 = paramList[6] 
        self.p1 = paramList[7] 
        self.p2 = paramList[8] 
        self.mtx = np.zeros((3,3), dtype=float)
        self.mtx[0, 0] = self.fx
        self.mtx[1, 1] = self.fy
        self.mtx[0, 2] = self.cx
        self.mtx[1, 2] = self.cy
        self.mtx[2, 2] = 1.0
        self.dist = (self.k1, self.k2, self.p1, self.p2, self.k3)
        pass


    def undistortion0(self, img) -> np.ndarray: # for rgb image
        """undistortion(crop)

        Args:
            img (np.ndarry): distorted image

        Returns:
            np.ndarray: undistorted image 
        """
        if len(img.shape) == 3:
            (h, w, _) = img.shape
        elif len(img.shape) == 2:
            (h, w) = img.shape
        else:
            return None
        new_mtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 0, (w,h))
        mapx, mapy = cv.initUndistortRectifyMap(self.mtx, self.dist, None, new_mtx, (w,h), CV_32FC1)
        # use undistort function
        # undistImg = cv.undistort(img, self.mtx, self.dist, None, new_mtx)
        # use remap function
        undistImg = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        (roix, roiy, roiw, roih) = roi
        print("roi = ", roi)
        return undistImg[roiy:roiy+roih, roix:roix+roiw]


    def undistortion1(self, img) -> np.ndarray:
        """undistortion(no crop) 

        Args:
            img (np.ndarry): distorted image

        Returns:
            np.ndarray: undistorted image 
        """
        if len(img.shape) == 3:
            (h, w, _) = img.shape
        elif len(img.shape) == 2:
            (h, w) = img.shape
        else:
            return None
        new_mtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 0, (w,h))
        mapx, mapy = cv.initUndistortRectifyMap(self.mtx, self.dist, None, new_mtx, (w,h), CV_32FC1)
        undistImg = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        return undistImg


    def undistortion_coord(self, src, mask):
        """undistortion (data position only, no interplation) 

        Args:
            img (_type_): _description_

        Returns:
            undistImg: undistortion image
            undistMask: undistortion point position (255 otherwise 0)
        """
        if len(mask.shape) == 3:
            (h, w, _) = mask.shape
        elif len(mask.shape) == 2:
            (h, w) = mask.shape
        else:
            return None
        new_mtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
        mapx, mapy = cv.initUndistortRectifyMap(self.mtx, self.dist, None, new_mtx, (w,h), CV_32FC1)
        undistSrc = np.zeros((h, w), dtype=np.float32)
        undistMask = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                u = round(mapx[y, x]) # original x
                v = round(mapy[y, x]) # original y
                if u < 0 or u >= w or v < 0 or v >= h:
                    continue 
                if mask[v, u] != 0: # is a check point
                    undistSrc[y, x] = src[v, u]
                    undistMask[y, x] = 255
        return undistSrc, undistMask


    def dm2pc(self, img, mask) -> np.ndarray:
        """convert depthmap to point cloud
        Args:
            img (np.ndarray): depthmap
            mask (np.ndarray): positions
        Returns:
            np.ndarray: point cloud [num, 3] # meter
        """
        (h, w) = img.shape
        length = h*w
        pc = np.zeros((length, 3), dtype=np.float32)
        count = 0
        for v in range(h):
            for u in range(w):
                #? treat value as Z
                if mask[v, u] == 0:
                    continue
                Z = img[v, u]
                Y = (v - self.cy) * Z / self.fy
                X = (u - self.cx) * Z / self.fx
                pc[count, :] = [X, Y, Z]
                count += 1
        return pc[:count, :]
    

    def pc2dm(self, pc, size) -> np.ndarray:
        """convert point cloud to depthmap

        Args:
            pc (np.ndarray): point cloud [num, 3]

        Returns:
            np.ndarray: depthmap 
        """
        h, w = size
        num, dim = pc.shape
        img = np.zeros(size, dtype=np.float32)
        mask = np.zeros(size, dtype=np.uint8)
        for idx in range(num):
            X, Y, Z =  pc[idx, :]
            if Z < 0.00001 or np.isnan(Z):
            # if Z < 0.00001:
                continue
            u = round(X * self.fx / Z + self.cx)
            v = round(Y * self.fy / Z + self.cy)
            if u < 0 or u >= w or v < 0 or v >= h:
                continue
            img[v, u] = Z
            mask[v, u] = 255
        return img, mask

class TransCalib:
    """ Translation 2 cameras
    """
    def __init__(self, paramList) -> None:
        self.R = np.zeros((3, 3), dtype = np.float32)
        self.T = np.zeros((1, 3), dtype = np.float32)
        # rotation
        self.R[0,0] = paramList[0]
        self.R[1,0] = paramList[1]
        self.R[2,0] = paramList[2]
        self.R[0,1] = paramList[3]
        self.R[1,1] = paramList[4]
        self.R[2,1] = paramList[5]
        self.R[0,2] = paramList[6]
        self.R[1,2] = paramList[7]
        self.R[2,2] = paramList[8]
        # shift 
        self.T[0, 0] = paramList[9]
        self.T[0, 1] = paramList[10]
        self.T[0, 2] = paramList[11]
        pass
    
    def translate_pcs(self, src) -> np.ndarray:
        """translate point cloud (ToF->RGB)

        Args:
            src (np.ndarray): point cloud 

        Returns:
            np.ndarray: point cloud 
        """
        #? ANCHOR mm or m?
        return src.dot(self.R) + self.T/1000

    def translate_pcs1(self, src) -> np.ndarray:
        """translate point cloud (RGB->TOF)

        Args:
            src (np.ndarray): point cloud 

        Returns:
            np.ndarray: point cloud 
        """
        # return (src-self.T/100).dot(np.linalg.inv(self.R))
        return (src-self.T/1000).dot(self.R)


class GetParam:
    """read params from files
    """
    def __init__(self, folder) -> None:
        self.folder = folder
        self.camera_calib_folder = f"{folder}/camera_calib"
        self.device_calib_folder = f"{folder}/device_calib"
        pass

    def read_spot_init_coord(self)->np.ndarray:
        """get initial spot coordinate

        Returns:
            np.ndarray: 2x12x12 
        """
        with h5py.File(f"{self.device_calib_folder}/calibparams_h2d_3_1_006.h5") as f:
            return np.array(f["Params"]["spot_coor_init"])
    
    def read_camera_params(self):
        """read rgb camera and tof camera parameters, also calib parameters between them
        """
        raw = pd.read_csv(f"{self.camera_calib_folder}/param.txt", delimiter="\s+", comment="/", \
        # raw = pd.read_csv(f"{self.folder}/param.txt", delimiter="\s+", comment="/", \
                header=None, skip_blank_lines=True, skipinitialspace=True)
        self.rgbCalib = CamCalib_k3(raw.iloc[0:9, 1].tolist())
        self.tofCalib = CamCalib_k3(raw.iloc[9:18, 1].tolist())
        self.transCalib = TransCalib(raw.iloc[18:, 1].tolist())
