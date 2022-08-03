from cProfile import label
from calendar import c
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


ROOT_FOLDER = "./dat/real/hand_test"
frame_num = 0 
fn_gt_mask = f"{ROOT_FOLDER}/%08d_mask.png" % frame_num

# fn_est_depths = [f"{ROOT_FOLDER}/%08d_upsampling_results_%d.tiff" % (frame_num, 0), 
#                 f"{ROOT_FOLDER}/%08d_upsampling_results_%d.tiff" % (frame_num, 1), 
#                 f"{ROOT_FOLDER}/%08d_upsampling_results_%d.tiff" % (frame_num, 2), 
#                 f"{ROOT_FOLDER}/%08d_upsampling_results_%d.tiff" % (frame_num, 3), 
#                 f"{ROOT_FOLDER}/%08d_upsampling_results_%d.tiff" % (frame_num, 4)]
est_folder = f"{ROOT_FOLDER}/tuning_method4"
fn_est_depths = [f for f in os.listdir(est_folder)] 

color_lines = ['r-', 'g-', 'b-', 'y-', 'k-', 'c-', 'm-',
                'r:', 'g:', 'b:', 'y:', 'k:', 'c:', 'm:',
                'r--', 'g--', 'b--', 'y--', 'k--', 'c--', 'm--',]

def get_GtMask_EvaRect(fn_gt_mask):
    """get groundtruth mask and evaluation rect

    Args:
        fn_gt_mask (string): file name of ground truth 

    Returns:
        gt_mask(ndarray): groundtruth mask 
        eva_rect(ndarray): evaluation rect
    """
    # read gt mask
    gt_mask = cv.imread(fn_gt_mask, cv.IMREAD_GRAYSCALE)
    # find contours
    ret, thresh = cv.threshold(gt_mask, 127, 255, cv.THRESH_BINARY)
    contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours[0][1]
    # draw poly rect 
    margin = 20 #* boarder margin for evaluation
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    colorImg = cv.cvtColor(gt_mask, cv.COLOR_GRAY2BGR)
    cv.drawContours(colorImg, [box], 0, (0, 0, 255), 2) # draw contours
    colorRect = np.ones(colorImg.shape, dtype=np.uint8) * 255 # color rect image
    cv.polylines(colorRect, [box], isClosed=True, color=(255, 0, 0), thickness=margin) # draw margin
    cv.fillPoly(colorRect, [box], color=(255,0,0)) # fill poly rect
    cv.polylines(colorRect, [box], isClosed=True, color=(255, 0, 255), thickness=1) # marked bounding box
    # evaluaiton rect  
    colorRect[gt_mask == 0] = (0, 0, 0)
    eva_rect = np.ones(gt_mask.shape, dtype=np.uint8) * 255
    eva_rect[colorRect[:, :, 1] == 0] = 0
    if False: # show 
        colorRect[gt_mask == 0] = (0,0,0)
        cv.imshow("eva rect", eva_rect)
        cv.imshow("boundingBox", colorImg)
        cv.imshow("filled Rect", colorRect)
        cv.waitKey(0)
    return gt_mask, eva_rect


def calc_precision_recall(est_depth, gt_mask, eva_rect):
    """ calculation of precision, recall

    Args:
        est_depth (ndarray): estimated depth map
        gt_mask (ndarray): ground truth mask 
        eva_rect (ndarray): evaluation rect 

    Returns:
        precision(list): 
        recall(list):
        dice(list):
    """
    curr_depth = 0.1
    end_depth = 0.51
    span_depth = 0.005
    eva_depth = est_depth.copy()
    eva_depth[eva_rect != 0] = 100.0 # no recall
    positive_gt = gt_mask.copy()
    negative_gt = eva_rect.copy()
    negative_gt[positive_gt == 0] = 255

    recall_p = []
    precision_p = []
    dice_p = []
    while(curr_depth <= end_depth):
        positive_rect = eva_depth <= curr_depth
        negative_rect = eva_depth >= curr_depth
        tp = np.sum(positive_gt[positive_rect] == 0) # true positive
        tn = np.sum(negative_gt[negative_rect] == 0) # true negative
        fp = np.sum(negative_gt[positive_rect] == 0) # false positive
        fn = np.sum(positive_gt[negative_rect] == 0) # false negative
        if tp+fp == 0:
            tp = 1
        precision = (tp*100)/(tp+fp)
        recall = (tp*100)/(tp+fn)
        dice = tp/(tp + (fp + fn)/2) 
        precision_p.append(precision)
        recall_p.append(recall)
        dice_p.append(dice)
        curr_depth += span_depth
    return precision_p, recall_p, dice_p


def make_roc_graph(precision_ps, recall_ps, method_names):
    # makeup recall and precision
    fontsize = 20
    fig = plt.figure(0)
    for i,mn in enumerate(method_names):
        precisions = precision_ps[i]
        recalls = recall_ps[i]
        length = len(precisions)
        cur_idx = length - 1
        while(cur_idx >= 1):
            prev_idx = cur_idx - 1
            if precisions[prev_idx] == np.nan:
                precisions[prev_idx] = precisions[cur_idx]
            # if recalls[prev_idx] == np.nan:
            #     recalls[prev_idx] = recalls[cur_idx]
            if precisions[prev_idx] < precisions[cur_idx]:
                precisions[prev_idx] = precisions[cur_idx] 
            # if recalls[prev_idx] < recalls[cur_idx]:
            #     recalls[prev_idx] = recalls[cur_idx] 
            cur_idx -= 1
        plt.plot(recalls, precisions, color_lines[i], label=mn)
    ax = plt.gca()
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.legend(loc='best', fontsize=fontsize)
    plt.xlabel('recall[%]', fontsize=fontsize)
    plt.ylabel('precision[%]', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.grid()
    plt.show()

    return

def get_method_name(fn_est_depth):
    return fn_est_depth.split('.')[0]

if __name__ == '__main__':
    #* read gt_mask and evaluation rect
    gt_mask, eva_rect = get_GtMask_EvaRect(fn_gt_mask)
    #* read estimated depth and calculate precision & recall 
    precision_p = []
    recall_p = []
    dice_p = []
    methods = []
    for i, fn_est_depth in enumerate(fn_est_depths):
        est_depth = cv.imread(f"{est_folder}/{fn_est_depth}", -1)
        precisions, recalls, dices = calc_precision_recall(est_depth, gt_mask, eva_rect)
        precision_p.append(precisions)
        recall_p.append(recalls)
        dice_p.append(dices)
        methods.append(get_method_name(fn_est_depth))
    #* draw graph    
    make_roc_graph(precision_p, recall_p, methods)
    
