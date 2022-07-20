import numpy as np
import pandas as pd

ROOT_FOLDER = "./dat/results"
eva_list = ["WFGS", "FGS", "FBS", "PlanarFilter"]
fovs = [30, 45, 60, 90]
num_frames = 50
types = ["flood", "spot"]


# TIME, MAE, iMAE
if __name__ == '__main__':
    for method in eva_list:
        for fov in fovs:
            for frame in range(num_frames):
                for t in types:
                    folder = f"{ROOT_FOLDER}/{method}/CamWithBump_{fov}/%05d/{t}" % frame
                    res_file = f"{folder}/result.txt"
                    res = pd.read_csv(res_file, delimiter=":", header=None)
                    exit(0)
