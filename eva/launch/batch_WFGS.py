import subprocess
import os

# app_type = "Debug"
app_type = "Release"

rootPath = os.getcwd()
binPath = "bin/Windows_64"
app = f"{rootPath}/{binPath}_{app_type}/WFGS_test.exe"

dataPath = "dat\\artifact"
sourcePath = "dat\\source"
outputPath = "dat\\results\\WFGS"

fov_list_test = [30, 45, 60, 90];
frames_test = 50

if not os.path.exists(f"{rootPath}/{outputPath}"):
    os.mkdir(f"{rootPath}/{outputPath}")


if __name__ == '__main__':
    for fov in fov_list_test:
        folderIn = f"{rootPath}\\{dataPath}\\CamWithBump_{fov}"
        folderSrc = f"{rootPath}\\{sourcePath}\\CamWithBump_{fov}"
        folderOut = f"{rootPath}\\{outputPath}\\CamWithBump_{fov}"
        if not os.path.exists(folderOut):
            os.mkdir(folderOut)
        for fn in range(frames_test): 
            flood_file = "%s\\%05d_flood.tiff" % (folderIn, fn)
            spot_file = "%s\\%05d_spot.tiff" % (folderIn, fn)
            guide_file = "%s\\house0_round0\\polar\\%05d_s0_denoised.png" % (folderSrc, fn)
            gt_file = "%s\\house0_round0\\polar\\%05d_gtD.png" % (folderSrc, fn)
            cam_file = "%s\\house0_round0\\polar\\%05d_camK.txt" % (folderSrc, fn)
            normal_file = "%s\\house0_round0\\polar\\%05d_gtN.png" % (folderSrc, fn)
            # process
            print(gt_file)
            # flood
            output_folder = "%s\\%05d\\flood" % (folderOut, fn)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            cmd = "%s %s %s %s %s %s %s" % (app, guide_file, flood_file, gt_file, cam_file, normal_file, output_folder)
            subprocess.run(cmd)
            # spot
            output_folder = "%s\\%05d\\spot" % (folderOut, fn)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            cmd = "%s %s %s %s %s %s %s" % (app, guide_file, spot_file, gt_file, cam_file, normal_file, output_folder)
            subprocess.run(cmd)