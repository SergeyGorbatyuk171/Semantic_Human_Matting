"""
    train

Author: Zhengwei Li
Date  : 2018/12/24
"""

import argparse
import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset import make_loader, transforms_train, transforms_test
from model.network import FullNet
from tqdm import tqdm


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='AIFU')
    parser.add_argument('--dataDir', default='./data/coco_densepose', help='dataset directory')
    parser.add_argument('--saveDir', default='./ckpt', help='model save dir')
    parser.add_argument('--valOutDir', default='output/temp', help='directory to save output while validation')
    # parser.add_argument('--trainData', default='human_matting_data', help='train dataset name')
    # parser.add_argument('--trainList', default='./data/list.txt', help='train img ID')
    parser.add_argument('--model_dir', default='coco_densepose', help='where to save model')

    parser.add_argument('--finetuning', action='store_true', default=False, help='finetuning the training')
    parser.add_argument('--without_gpu', action='store_true', default=False, help='no use gpu')

    parser.add_argument('--nThreads', type=int, default=4, help='number of threads for data loading')
    parser.add_argument('--train_batch', type=int, default=8, help='input batch size for train')
    # parser.add_argument('--patch_size', type=int, default=256, help='patch size for train')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lrDecay', type=int, default=100)
    parser.add_argument('--lrdecayType', default='keep')
    parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train')
    # parser.add_argument('--save_epoch', type=int, default=1, help='number of epochs to save model')

    parser.add_argument('--train_phase', default='end_to_end', choices=['end_to_end', 'pre_train_t_net'],
                        help='train phase')

    args = parser.parse_args()
    return args


def set_lr(args, epoch, optimizer):
    lrDecay = args.lrDecay
    decayType = args.lrdecayType
    if decayType == 'keep':
        lr = args.lr
    elif decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2 ** epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'poly':
        lr = args.lr * math.pow((1 - epoch / args.nEpochs), 0.9)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


class Train_Log():
    def __init__(self, args):
        self.args = args

        self.save_dir = os.path.join(args.saveDir, args.model_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)

        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')

    def save_model(self, model, epoch):

        # epoch_out_path = "{}/ckpt_e{}.pth".format(self.save_dir_model, epoch)
        # print("Checkpoint saved to {}".format(epoch_out_path))

        # torch.save({
        #     'epoch': epoch,
        #     'state_dict': model.state_dict(),
        # }, epoch_out_path)

        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, lastest_out_path)

        model_out_path = "{}/model_obj.pth".format(self.save_dir_model)
        torch.save(
            model,
            model_out_path)

    def load_model(self, model):

        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        ckpt = torch.load(lastest_out_path)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(lastest_out_path, ckpt['epoch']))

        return start_epoch, model

    def save_log(self, log):
        self.logFile.write(log + '\n')


def loss_function(args, img, trimap_pre, trimap_gt, alpha_pre, alpha_gt, weight=None):
    # -------------------------------------
    # classification loss L_t
    # ------------------------
    # Cross Entropy 
    # criterion = nn.BCELoss()
    # trimap_pre = trimap_pre.contiguous().view(-1)
    # trimap_gt = trimap_gt.view(-1)
    # L_t = criterion(trimap_pre, trimap_gt)

    if weight is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=weight)
    L_t = criterion(trimap_pre, trimap_gt[:, 0, :, :].long())

    # -------------------------------------
    # prediction loss L_p
    # ------------------------
    eps = 1e-6
    # l_alpha
    L_alpha = torch.sqrt(torch.pow(alpha_pre - alpha_gt, 2.) + eps).mean()

    # L_composition
    fg = torch.cat((alpha_gt, alpha_gt, alpha_gt), 1) * img
    fg_pre = torch.cat((alpha_pre, alpha_pre, alpha_pre), 1) * img

    L_composition = torch.sqrt(torch.pow(fg - fg_pre, 2.) + eps).mean()

    L_p = 0.5 * L_alpha + 0.5 * L_composition

    # train_phase
    if args.train_phase == 'pre_train_t_net':
        loss = L_t
    if args.train_phase == 'end_to_end':
        loss = L_p + 0.01 * L_t

    return loss, L_alpha, L_composition, L_t


def main():
    print("=============> Loading args")
    args = get_args()

    print("============> Environment init")
    if args.without_gpu:
        print("use CPU !")
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("No GPU is is available !")
            device = torch.device('cpu')
    if not os.path.exists(args.valOutDir):
        os.mkdir(args.valOutDir)

    print("============> Building model ...")
    model = FullNet()
    model.to(device)

    print("============> Loading datasets ...")
    train_loader = make_loader(os.path.join(args.dataDir, 'tinyval'), transform=transforms_test(), batch_size=6)
    val_loader = make_loader(os.path.join(args.dataDir, 'tinyval'), transform=transforms_test(), batch_size=2)

    print("============> Set optimizer ...")
    lr = args.lr
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, betas=(0.9, 0.999),
                           weight_decay=0.0005)

    print("============> Start Train ! ...")
    start_epoch = 1
    trainlog = Train_Log(args)
    if args.finetuning:
        start_epoch, model = trainlog.load_model(model)

    model.train()
    val_loss_best = 1e8
    testimg = None

    for epoch in range(start_epoch, args.nEpochs + 1):
        print(f'Epoch {epoch}/{args.nEpochs}')
        train_losses = []
        loss_train = 0
        L_alpha_ = 0
        L_composition_ = 0
        L_cross_ = 0
        ce_weights = torch.FloatTensor([1., 5., 2.], ).to(device)
        if args.lrdecayType != 'keep':
            lr = set_lr(args, epoch, optimizer)
        model.train(True)
        for i, sample_batched in tqdm(enumerate(train_loader)):
            img, trimap_gt, alpha_gt = sample_batched['image'], sample_batched['trimap'], sample_batched['alpha']
            img, trimap_gt, alpha_gt = img.to(device), trimap_gt.to(device), alpha_gt.to(device)
            if testimg is None:
                testimg = img[0].unsqueeze_(0)
            # print(img.shape, trimap_gt.shape, alpha_gt.shape)

            trimap_pre, alpha_pre = model(img)
            # print(trimap_pre.shape, alpha_pre.shape)
            loss, L_alpha, L_composition, L_cross = loss_function(args,
                                                                  img,
                                                                  trimap_pre,
                                                                  trimap_gt,
                                                                  alpha_pre,
                                                                  alpha_gt,
                                                                  weight=ce_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                t_pre, _ = model(testimg)
                t_pre = t_pre[0].detach().cpu().numpy().transpose((1, 2, 0))
                flat = np.argmax(t_pre, axis=2) *  127
                cv2.imwrite(os.path.join(args.valOutDir, f'000.png'), flat)


            loss_train += loss.item()
            train_losses.append(loss.item())
            L_alpha_ += L_alpha.item()
            L_composition_ += L_composition.item()
            L_cross_ += L_cross.item()
        val_losses = []
        model.train(False)
        for i, sample_batched in tqdm(enumerate(val_loader)):
            img, trimap_gt, alpha_gt = sample_batched['image'], sample_batched['trimap'], sample_batched['alpha']
            img, trimap_gt, alpha_gt = img.to(device), trimap_gt.to(device), alpha_gt.to(device)

            trimap_pre, alpha_pre = model(img)
            loss, L_alpha, L_composition, L_cross = loss_function(args,
                                                                  img,
                                                                  trimap_pre,
                                                                  trimap_gt,
                                                                  alpha_pre,
                                                                  alpha_gt)
            val_losses.append(loss.item())
            if i % 10 == 0:
                tm_pre = trimap_pre.detach().cpu().numpy()
                for k in range(tm_pre.shape[0]):
                    ims = tm_pre[k].transpose((1, 2, 0))
                    flat = np.argmax(ims, axis=2) * 127
                    cv2.imwrite(os.path.join(args.valOutDir, f'{i}_{k}.png'), flat)
        validation_epoch_loss = np.mean(np.array(val_losses))
        training_epoch_loss = np.mean(np.array(train_losses))
        tqdm.write(f'Train loss: {training_epoch_loss}\nVal loss: {validation_epoch_loss}')
        if validation_epoch_loss < val_loss_best:
            val_loss_best = validation_epoch_loss
            trainlog.save_model(model, epoch)

        # if epoch % args.save_epoch == 0:
        #
        #     loss_ = loss_ / (i + 1)
        #     L_alpha_ = L_alpha_ / (i + 1)
        #     L_composition_ = L_composition_ / (i + 1)
        #     L_cross_ = L_cross_ / (i + 1)
        #
        #     log = "[{} / {}] \tLr: {:.5f}\nloss: {:.5f}\tloss_p: {:.5f}\tloss_t: {:.5f}\t" \
        #         .format(epoch, args.nEpochs,
        #                 lr,
        #                 loss_,
        #                 L_alpha_ + L_composition_,
        #                 L_cross_)
        #     print(log)
        #     trainlog.save_log(log)
        #     trainlog.save_model(model, epoch)


if __name__ == "__main__":
    main()
