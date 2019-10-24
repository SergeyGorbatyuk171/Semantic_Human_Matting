"""
    end to end network

Author: Zhengwei Li
Date  : 2018/12/24
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.M_Net import M_net
from model.T_Net import T_mv2_unet
from model.UNet import UNet16

T_net = T_mv2_unet


class FullNet(nn.Module):
    def __init__(self, tb_writer=None):
        super(FullNet, self).__init__()

        self.t_net = UNet16(pretrained=True)
        self.m_net = M_net()
        self.tb_writer = tb_writer

    def forward(self, input):
        # trimap
        trimap = self.t_net(input)
        trimap_softmax = F.softmax(trimap, dim=1)

        # paper: bs, fs, us
        # was: bg, fg, unsure
        bg, unsure, fg = torch.split(trimap_softmax, 1, dim=1)
        fg_img = torch.cat((fg[0], fg[0], fg[0]), dim=0)
        # self.tb_writer.add_images(f'temp',
        #                      torch.stack([trimap[0], fg_img], dim=0), dataformats='NCHW')

        # concat input and trimap
        m_net_input = torch.cat((input, trimap_softmax), 1)

        # matting
        alpha_r = self.m_net(m_net_input)
        # fusion module
        # paper : alpha_p = fs + us * alpha_r
        alpha_p = fg + unsure * alpha_r

        return trimap, alpha_p
