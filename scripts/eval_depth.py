from __future__ import absolute_import, division, print_function
import cv2
import sys
import numpy as np
from mmcv import Config
import os

import torch
from torch.utils.data import DataLoader

sys.path.append('.')
from mono.model.registry import MONO
from mono.model.mono_baseline.layers import disp_to_depth
from mono.datasets.utils import readlines, compute_errors
from mono.datasets.kitti_dataset import KITTIRAWDataset

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
STEREO_SCALE_FACTOR = 36
MIN_DEPTH=1e-3
MAX_DEPTH=80


def evaluate(MODEL_PATH, CFG_PATH, GT_PATH):
    filenames = readlines("/content/FeatDepth/mono/datasets/splits/exp/val_files.txt")
    cfg = Config.fromfile(CFG_PATH)

    dataset = KITTIRAWDataset(cfg.data['in_path'],
                              filenames,
                              cfg.data['height'],
                              cfg.data['width'],
                              [0],
                              is_train=False,
                              gt_depth_path=GT_PATH,
                              img_ext='.png')

    dataloader = DataLoader(dataset,
                            1,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False)

    cfg.model['imgs_per_gpu'] = 1
    model = MONO.module_dict[cfg.model['name']](cfg.model)
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.cuda()
    model.eval()

    pred_disps = []
    tgt_imgs = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            tgt_img = inputs[('color', 0, 0)].cpu().detach().numpy()
            tgt_img = np.transpose(tgt_img[0], (1, 2, 0))
            tgt_imgs.append(tgt_img)

            outputs = model(inputs)

            disp = outputs[("disp", 0, 0)]

            pred_disp, _ = disp_to_depth(disp, 0.1, 100)
            pred_disp = pred_disp.cpu()[:, 0].numpy()
            pred_disps.append(pred_disp)

    pred_disps = np.concatenate(pred_disps)

    gt_depths = np.load(GT_PATH, allow_pickle=True)

    print("-> Evaluating")
    if cfg.data['stereo_scale']:
        print('using baseline')
    else:
        print('using mean scaling')

    errors = []
    ratios = []
    predictions = np.zeros((697, gt_depths[0].shape[0], gt_depths[0].shape[1]))
    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        predictions[i] = pred_disp

        pred_depth = 1 / pred_disp

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)

        if cfg.data['stereo_scale']:
            ratio = STEREO_SCALE_FACTOR

        pred_depth *= ratio
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        errors.append(compute_errors(gt_depth, pred_depth))

    ratios = np.array(ratios)
    med = np.median(ratios)
    errors = np.array(errors)
    mean_errors = errors.mean(0)
    print("Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
    print("\n" + ("{:>}| " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{:.3f} " * 7).format(*mean_errors.tolist()) + "\\\\")
    
    print("\n-> Saving results!")
    output_dir = '/content/FeatDepth/results/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_dir + 'predictions.npy', predictions)
    np.save(output_dir + 'errors.npy', errors)

    # save 5% best images and 5% worst images
    decisive_errors = errors[:, 0] # abs_rel

    good_value = np.percentile(decisive_errors, 5)
    bad_value = np.percentile(decisive_errors, 95)

    best_indices = np.where(decisive_errors < good_value)[0]
    worst_indices = np.where(decisive_errors > bad_value)[0]

    best_predictions = predictions[best_indices]
    worst_predictions = predictions[worst_indices]
    np.save(output_dir + 'best_predictions.npy', best_predictions)
    np.save(output_dir + 'worst_predictions.npy', worst_predictions)


    tgt_imgs = np.asarray(tgt_imgs)
    tgt_imgs = tgt_imgs * 255.
    tgt_imgs = np.asarray(tgt_imgs, dtype=np.int)
    best_imgs = tgt_imgs[best_indices]
    worst_imgs = tgt_imgs[worst_indices]
    np.save(output_dir + 'best_imgs.npy', best_imgs)
    np.save(output_dir + 'worst_imgs.npy', worst_imgs)
    
    print("\n-> Done!")

if __name__ == "__main__":
    CFG_PATH = '/content/FeatDepth/config/cfg_kitti_fm.py'#path to cfg file
    GT_PATH = '/content/logs/gt.npy'#path to kitti gt depth
    MODEL_PATH = '/content/logs/epoch_1.pth'#path to model weights
    evaluate(MODEL_PATH, CFG_PATH, GT_PATH)
