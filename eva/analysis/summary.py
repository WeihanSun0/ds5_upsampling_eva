from cProfile import label
import enum
import numpy as np
import pandas as pd
import matplotlib.pylab as ply
from torch import mm

markers = [["r+", "r*", "r<", "r^"],
            ["g+", "g*", "g<", "g^"],
            ["b+", "b*", "b<", "b^"],
            ["m+", "m*", "m<", "m^"]]
ROOT_FOLDER = "./dat/results"
eva_list = ["WFGS", "FGS", "FBS", "PlanarFilter"]
# eva_list = ["WFGS"]
fovs = [30, 45, 60, 90]
num_frames = 50
# types = ["flood", "spot"]
types = ["flood"]

labels = ["TIME[ms]", "MAE[m]", "iMAE", # 0-2 
        "MAD", # 3:Median Absolut Deviation
        "RMSE[m]", "iRMSE", # 4-5
        "Edge RMSE[m]", "nonEdge RMSE", # 6-7
        "PPE[m]", # 8: Point-to-Plane Error
        "SD",  # 9: Spatial Density
        "VP", # 10: valid pixels ratio
        "MNS", # 11: Mean Normal Similarity
        "Edge MNS", "nonEdge MNS",
        "ssim", 
        "badpix", "badpix(radio)"]


def plotOneFigure(list_x_indicator_index, list_y_indicator_index, resMat):
    for x_indicator_index, y_indicator_index in zip(list_x_indicator_index, list_y_indicator_index):
        fig = ply.figure(0) 
        for m, method in enumerate(eva_list):
            for f, fov in enumerate(fovs):
                x_indicator = resMat[m, f, :, 0, x_indicator_index].mean()
                y_indicator = resMat[m, f, :, 0, y_indicator_index].mean()
                ply.plot(x_indicator, y_indicator, markers[m][f], label=f"{method}-fov:{fov}")
        ply.legend(loc='best')
        ply.xlabel(f'avg. {labels[x_indicator_index]}')
        ply.ylabel(f'avg. {labels[y_indicator_index]}')
        ply.show()

# TIME, MAE, iMAE
if __name__ == '__main__':
    # read data
    resMat = np.zeros((len(eva_list), len(fovs), num_frames, len(types), len(labels)), np.float64)
    for m, method in enumerate(eva_list):
        for f,fov in enumerate(fovs):
            for frame in range(num_frames):
                for t, tp in enumerate(types):
                    folder = f"{ROOT_FOLDER}/{method}/CamWithBump_{fov}/%05d/{tp}" % frame
                    res_file = f"{folder}/result.txt"
                    res = pd.read_csv(res_file, delimiter=":", header=None)
                    res = res.replace('-nan(ind)', 0.0)
                    resMat[m, f, frame, t, :] = res[1].to_numpy()
    plotOneFigure([0, 0, 0, 0], [1, 4, 6, 11], resMat)
    exit(0)
    # avg time-MAE 0, 1
    # avg time-RMSE 0, 4
    # avg time-Edge RMSE 0, 
    # avg time-nonEdge RMSE
    # avg