# -*- coding: utf-8 -*-
from __future__ import print_function, division

from cv2_transform import transforms 
from torch.utils.data import DataLoader
import torch

from network.rpm import RPM
from network.pcb import PCB
from data_read import ImageTxtDataset
from train import str2bool

import argparse, time, os, sys
import numpy as np
from os import path as osp
from matplotlib import cm
import matplotlib.pyplot as plt

def get_data(batch_size, test_set, query_set):
    transform_test = transforms.Compose([
        transforms.Resize(size=(opt.img_height, opt.img_width)),
        transforms.ToTensor(),
    ])

    test_imgs = ImageTxtDataset(test_set, transform=transform_test)
    query_imgs = ImageTxtDataset(query_set, transform=transform_test)

    test_data = DataLoader(test_imgs, batch_size, shuffle=False, num_workers=4)
    query_data = DataLoader(query_imgs, batch_size, shuffle=False, num_workers=4)
    return test_data, query_data


def get_id(img_path):
    cameras = []
    labels = []
    for path in img_path:
        cameras.append(int(path[0].split('/')[-1].split('_')[1][1]))
        labels.append(path[1])
    return np.array(cameras), np.array(labels)


def extract_mask(net, dataloaders):
    count = 0
    output_dir = 'vis/output_{}'.format(opt.weight)
    os.makedirs(output_dir, exist_ok=True)

    for img, _ in dataloaders:
        competitive_masks = None
        cooperative_mask = None

        with torch.no_grad():
            _, masks = net(img.cuda())

        B, _, H, W = img.shape

        for i in range(B):
            img_subdir = os.path.join(output_dir, f'image_{count}')
            os.makedirs(img_subdir, exist_ok=True)

            # Save competitive masks
            for idx in range(opt.num_part):
                mask_resized = torch.nn.functional.interpolate(masks[idx], size=(H, W), mode='bicubic', align_corners=False)
                mask_np = mask_resized.squeeze().cpu().detach().numpy()

                cmap = cm.get_cmap('jet')
                mask_colored = (255 * cmap(mask_np)[:, :, 1:]).clip(0, 255).astype(np.uint8)

                original_img = (255 * img[i]).permute(1,2,0).cpu().numpy().clip(0, 255).astype(np.uint8)
                plt.imsave(os.path.join(img_subdir, f'lsm_{idx}.png'), (0.3 * original_img + 0.7 * mask_colored).clip(0, 255).astype(np.uint8))

            count += 1
        break
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--img-height', type=int, default=384)
    parser.add_argument('--img-width', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dataset-root', type=str, default="../dataset/")
    parser.add_argument('--net', type=str, default="regnet_y_3_2gf", help="regnet_y_3_2gf, regnet_y_8gf")
    parser.add_argument('--decoder', type=str, default="rpm_max")
    parser.add_argument('--gpus', type=str, default="0,1", help='number of gpus to use.')
    parser.add_argument('--num-part', type=int, default=2)
    parser.add_argument('--feat-num', type=int, default=0)
    parser.add_argument('--use-global', type=str2bool, default=True)
    parser.add_argument('--weight', type=str, default="ema")

    opt = parser.parse_args()

    data_dir = osp.join(opt.dataset_root, "Market-1501-v15.09.15")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

    test_set = [(osp.join(data_dir, 'bounding_box_test',line), int(line.split('_')[0])) for line in os.listdir(osp.join(data_dir, 'bounding_box_test')) if "jpg" in line and "-1" not in line]
    query_set = [(osp.join(data_dir, 'query',line), int(line.split('_')[0])) for line in os.listdir(osp.join(data_dir, 'query')) if "jpg" in line]   
 
    test_cam, test_label = get_id(test_set)
    query_cam, query_label = get_id(query_set)

    ######################################################################
    # Load Collected data Trained model
    mod_pth = osp.join('params', '{}.pth'.format(opt.weight))
    if opt.feat_num == 0:
        opt.feat_num = 1024

    if "rpm" in opt.decoder:
        net = RPM(decoder=opt.decoder, num_classes=751, num_part=opt.num_part, feat_num=opt.feat_num, net=opt.net, h=opt.img_height//16, w=opt.img_width//16, use_global=opt.use_global)
    elif "pcb" in opt.decoder:
        net = PCB(decoder=opt.decoder, num_classes=751, num_part=opt.num_part, feat_num=opt.feat_num, net=opt.net, h=opt.img_height//16, w=opt.img_width//16, use_global=opt.use_global)

    net.load_state_dict(torch.load(mod_pth), False)
    net.cuda()
    net.eval()

    # Extract feature
    test_loader, query_loader = get_data(opt.batch_size, test_set, query_set)
    extract_mask(net, test_loader)

