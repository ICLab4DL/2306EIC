#!/usr/bin/python3
# coding=utf-8

import os
import sys

# sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lib import dataset
from net import SODNet
import logging as logger
import copy

TAG = "SODNet"
SAVE_PATH = TAG
GPU_ID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S', \
                   filename="test_%s.log" % (TAG), filemode="w")

ROOT_PATH = "/li_zhengdao/SODNet"
SAVE_PATH = ROOT_PATH + "/save_path"

DATASETS = ['/data/DUTS']

# DATASETS = ['/data/SOD', '/data/PASCAL-S', '/data/ECSSD', '/data/HKU-IS',
#             '/data/DUT-OMRON', '/data/DUTS']


class Test(object):
    def __init__(self, Dataset, datapath, Network, cuda = True):
        ## dataset
        self.datapath = datapath.split("/")[-1]
        print("Testing on %s" % self.datapath)
        self.cfg = Dataset.Config(datapath=datapath, snapshot=SAVE_PATH + '/model-100', mode='test', cuda=cuda)
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=True, num_workers=8)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
        self.net.eval()

    def save(self):

        with torch.no_grad():
            #             i = 100
            for omg, image, mask, _, name in self.loader:
                out2, out3, out4, out5 = self.net(image.cuda().float())
                out = out2
                head = '/pred_maps/{}/'.format(TAG) + self.cfg.datapath.split('/')[-1]
                path = ROOT_PATH + head
                if not os.path.exists(path + "/256/"):
                    os.makedirs(path + "/256/")
                if not os.path.exists(path + "/128/"):
                    os.makedirs(path + "/128/")
                if not os.path.exists(path + "/512/"):
                    os.makedirs(path + "/512/")
                self.write_mask_img(out, omg, path, name[0], 512, 512)

    #                 self.write_mask_img(out, omg, path, name[0], 256, 256)

    #                 i -= 1
    #                 if i < 0:
    #                     break

    def write_mask_img(self, mask, img, path, pngname, W, H):

        if "\\" in pngname:
            pngname = pngname.split('\\')[-1]
        # mask output
        mask = F.interpolate(mask, size=(W, H), mode='bilinear')
        mask = (torch.sigmoid(mask[0, 0]) * 255).cpu().numpy()
        mask_name = "{}/{}/mask-{}".format(path, str(W), pngname)

        re_mask = copy.deepcopy(mask)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                re_mask[i][j] = 255 - mask[i][j]
        print(re_mask.shape)
        print(re_mask)
        print(mask)
        mask_name = "{}/{}/{}".format(path, str(W), pngname.replace('png', 'bmp'))
        cv2.imwrite(mask_name, np.uint8(mask))

        # origin output
        omg = img.squeeze().numpy()
        omg = cv2.resize(omg, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
        ori_name = "{}/origin-{}".format(path, pngname)
        # cv2.imwrite(ori_name, omg)

        # mask + origin outout
        mask = mask.reshape((W, H, 1))
        momg = mask + omg
        momg_name = "{}/{}/{}".format(path, str(W), pngname.replace('png', 'jpg'))
        #         print(momg_name)
        cv2.imwrite(momg_name, momg)


if __name__ == '__main__':
    cuda = True
    for e in DATASETS:
        e = ROOT_PATH + e
        t = Test(dataset, e, SODNet)
        t.save()
