#!/usr/bin/python3
# coding=utf-8

import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import dataset
from net import SODNet
import logging as logger
from lib.dataset import SODDataLoader

TAG = "lizhengdao"
# ROOT_PATH = "/li_zhengdao/SODNet"
ROOT_PATH = "."
SAVE_PATH = ROOT_PATH + "/save_path"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S', \
                   filename="train_%s.log" % (TAG), filemode="w")


def train(Dataset, Network, cuda):
    ## dataset
    cfg = Dataset.Config(datapath=ROOT_PATH + '/data/DUTS', savepath=SAVE_PATH, mode='train', batch=16,
                         lr=0.03, momen=0.9, decay=5e-4, epoch=10, cuda=cuda)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=12)
    ## network
    net = Network(cfg)
    net.train(True)
    if cfg.cuda:
        net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)
    global_step = 0
    # training
    for epoch in range(cfg.epoch):
        loader = SODDataLoader(loader, cfg)
        batch_idx = -1
        image, mask = loader.next()
        while image is not None:
            batch_idx += 1
            global_step += 1
            # outg, out4, out3, out2 = net(image)
            out2, out3, out4, out5 = net(image)

            lossg = F.binary_cross_entropy_with_logits(out5, mask)
            loss4 = F.binary_cross_entropy_with_logits(out4, mask)
            loss3 = F.binary_cross_entropy_with_logits(out3, mask)
            loss2 = F.binary_cross_entropy_with_logits(out2, mask)
            loss = 0.4 * lossg + 0.6 * loss4 + 0.8 * loss3 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print( '%s | step:%d/%d/%d | lr=%.6f | loss=%.6f ' % (
                    datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],
                    loss.item()))
            image, mask = loader.next()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.epoch:
            torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))


if __name__ == '__main__':
    cuda = False
    train(dataset, SODNet, cuda)
