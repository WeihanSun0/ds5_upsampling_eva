from re import I
from turtle import back, color
import cv2 as cv
import math
import numpy as np
from readData import GetParam, PosMapping

param_folder = "dat/real/camParam"
data_path = "./dat/real/hand_test"
fx = 374.8871819319296
fy = 374.3636185723908
cx = 484.7018651481662
cy = 268.95819709962416


# thresh (m)
def calcGradient(pntList, dm, mask, radius):
    h, w = dm.shape
    matGrad = np.zeros((len(pntList), 3), dtype=float)
    count = 0
    for x, y, z in pntList:
        maxGrad = 0.0
        oriGrad_x, oriGrad_y = 0, 0
        for r in range(y-radius, y+radius):
            for c in range(x-radius, x+radius):
                if r < 0 or r >= h or c < 0 or c >= w: #* out
                    continue
                if mask[r, c] == 255:
                    grad = np.abs(dm[r, c] - z)
                    if grad > maxGrad:
                        maxGrad = grad
                        oriGrad_x = c - x
                        oriGrad_y = r - y
        matGrad[count, :] = oriGrad_x, oriGrad_y, maxGrad
        count += 1
    return matGrad

def arrow(img, pntStart, ori_x, ori_y, dist):
    c = math.sqrt(ori_x**2 + ori_y**2)
    x_scale = ori_x/c
    y_scale = ori_y/c
    pntEnd_x = pntStart[0] + int(dist * x_scale) 
    pntEnd_y = pntStart[1] + int(dist * y_scale) 
    cv.arrowedLine(img, pntStart, (pntEnd_x, pntEnd_y), color=(0, 255, 0), thickness=1)
    return img

def drawGradient(matGrad, pntList, imgColor, scale=100.0, gradThresh = 0.1):
    img = imgColor.copy()
    for i in range(len(pntList)):
        x, y, z = pntList[i]
        ori_x, ori_y, grad = matGrad[i]
        if grad < gradThresh: #* too small difference
            continue
        arrow(img, (x, y), ori_x, ori_y, grad*scale)
    return img 

#! no use
def drawGrid(dm):
    h, w, _ = dm.shape
    for r in range(2, h, 12):
        for c in range(3, w, 12):
            cv.rectangle(dm, (c, r), (c+12, r+12), thickness=1, color=(255, 255, 255))
    return dm

#! no use
def backToGrid(dm, mask):
    h, w = dm.shape
    num_y, num_x = int(h/12), int(w/12)
    grid = np.zeros((num_y, num_x), dtype=float)
    int_x, int_y = 3, 2
    for j  in range(num_y):
        for i in range(num_x):
            for r in range(12):
                for c in range(12):
                    x = c + i*12 + int_x 
                    y = r + j*12 + int_y
                    if x < w and y < h:
                        if (mask[y, x] == 255):
                            grid[j, i] = dm[y, x]
    return grid

def findBoardPoints(dm, mask, imgBoarder, thresh):
    flood_yx = np.transpose(np.where(mask == 255))
    pntList = []
    for y, x in flood_yx:
        if imgBoarder[y, x] == 255 and dm[y, x] < thresh:
            pntList.append([x, y, dm[y, x]])
    return pntList

def drawBoarderPnts(pntList, color_flood_dmap, colorEdge, size):
    img = colorEdge.copy()
    for x, y, z in pntList:
        # cv.circle(img, center=(x, y), radius=size, color=color_flood_dmap[y, x], thickness=-1)
        cv.circle(img, center=(x, y), radius=size, color=tuple(int(e) for e in color_flood_dmap[y, x]), thickness=-1)
        # cv.circle(img, center=(x, y), radius=size, color=(0, 255, ), thickness=-1)
    return img

if __name__ == '__main__':
    # * read param
    gp = GetParam(param_folder)
    gp.read_camera_params()
    pm = PosMapping(gp.transCalib)
    # * read data
    rgb = cv.imread(f"{data_path}/0001_rgb_img.png")
    rgb_flood_pc = cv.imread(f"{data_path}/0001_flood_pc.tiff", -1)
    # pc -> dm
    flood_dmap, flood_dmapMask = gp.rgbCalib.pc2dm(rgb_flood_pc, rgb.shape[:2])
    # * mapping depthmap with RGB (test)
    norm_flood_dmap = cv.normalize(flood_dmap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    color_flood_dmap = cv.applyColorMap(norm_flood_dmap.astype('uint8'), cv.COLORMAP_JET)
    rgbMapping_flood = pm.map_depthmap_2_rgb(color_flood_dmap, rgb, flood_dmapMask, size=1)
    cv.imwrite(f"{data_path}/rgbMappingFlood.png", rgbMapping_flood)

    print("total points = ", np.sum(flood_dmapMask == 255))
    # * detect edge
    imgEdge = cv.Canny(rgb, 70, 130) #!
    kernelmatrix = np.ones((25, 25), np.uint8) #!
    imgBoarder = cv.dilate(imgEdge, kernel=kernelmatrix) #!
    colorEdge = cv.cvtColor(imgEdge, cv.COLOR_GRAY2BGR)
    colorBoarder = cv.cvtColor(imgBoarder, cv.COLOR_GRAY2BGR)
    edgeMapping_flood = pm.map_depthmap_2_rgb(color_flood_dmap, colorEdge, flood_dmapMask, size=1)
    boarderMapping_flood = pm.map_depthmap_2_rgb(color_flood_dmap, colorBoarder, flood_dmapMask, size=1)
    cv.imwrite(f"{data_path}/edgeMappingFlood.png", edgeMapping_flood)
    cv.imwrite(f"{data_path}/boarderMappingFlood.png", boarderMapping_flood)

    pntList = findBoardPoints(flood_dmap, flood_dmapMask, imgBoarder, 0.9)
    print("boarder points = ", len(pntList))

    imgBoardPnts = drawBoarderPnts(pntList, color_flood_dmap, colorEdge, 2)
    cv.imwrite(f"{data_path}/boarderPnts.png", imgBoardPnts)

    matGrad = calcGradient(pntList, flood_dmap, flood_dmapMask, 15)
    imgGrad = drawGradient(matGrad, pntList, edgeMapping_flood)
    cv.imwrite(f"{data_path}/gradArrow.png", imgGrad)
    # imGrid = backToGrid(flood_dmap, flood_dmapMask)
    # imGrid = drawGrid(edgeMapping_flood)
    # cv.imshow("grid", imGrid)
    # cv.waitKey(0)

    exit(0)