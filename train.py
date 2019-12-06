import argparse
import math
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.utils import make_grid

from data.dataset import make_loader, transforms_train, transforms_test
from model.network import FullNet
from tqdm import tqdm


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='AIFU')
    parser.add_argument('--dataDir', default='./data/coco_densepose', help='dataset directory')
    parser.add_argument('--images_dir', default='images', help='images dir name in the data root')
    parser.add_argument('--masks_dir', default='masks', help='masks dir name in the data root')
    parser.add_argument('--trimaps_dir', default='trimaps', help='trimaps dir name in the data root')
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

    parser.add_argument('--train_phase', default='end_to_end', choices=['end_to_end', 't_net', 'm_net'],
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

        # for some reason this fails with some threading errors. though, actually it is not used anywhere
        # model_out_path = "{}/model_obj.pth".format(self.save_dir_model)
        # torch.save(
        #     model,
        #     model_out_path)

    def load_model(self, model):

        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        ckpt = torch.load(lastest_out_path)
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(lastest_out_path, ckpt['epoch']))

        return start_epoch, model

    def save_log(self, log):
        self.logFile.write(log + '\n')


def loss_m_net(args, alpha_gt, alpha_pre, img):
    criterion = nn.MSELoss()
    # print(alpha_gt.shape, alpha_pred.shape, unsure_region.shape)
    L_alpha = criterion(alpha_gt, alpha_pre)

    fg = torch.cat((alpha_gt, alpha_gt, alpha_gt), 1) * img
    fg_pre = torch.cat((alpha_pre, alpha_pre, alpha_pre), 1) * img
    L_composition = criterion(fg, fg_pre)
    loss = 0.5 * L_alpha + 0.5 * L_composition

    return loss



def loss_function(args, img, trimap_pre, trimap_gt, alpha_pre, alpha_gt, weight=None):
    # -------------------------------------
    # classification loss L_t
    # ------------------------
    # Cross Entropy 
    # criterion = nn.BCELoss()
    # trimap_pre = trimap_pre.contiguous().view(-1)
    # trimap_gt = trimap_gt.view(-1)
    # L_t = criterion(trimap_pre, trimap_gt)

    # TODO: WTF? CEL considered to take RAW inputs, not Softmaxed

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
    # L_alpha = criterion(alpha_pre, alpha_gt[:, 0, :, :].long())
    # L_composition
    fg = torch.cat((alpha_gt, alpha_gt, alpha_gt), 1) * img
    fg_pre = torch.cat((alpha_pre, alpha_pre, alpha_pre), 1) * img

    L_composition = torch.sqrt(torch.pow(fg - fg_pre, 2.) + eps).mean()

    L_p = 0.5 * L_alpha + 0.5 * L_composition

    # train_phase
    if args.train_phase == 't_net':
        loss = L_t
    else:  # training end to end
        loss = L_p + 0.01 * L_t

    return loss, L_alpha, L_composition, L_t


def display_visuals_tb(tb_writer, visuals, it, metalabel):
    for label, image in visuals.items():
        sum_name = '{}/{}'.format(metalabel, label)
        tb_writer.add_image(sum_name, image, it)


def main():
    print("=============> Loading args")
    args = get_args()
    tb_writer = SummaryWriter()

    print("============> Environment init")
    if args.without_gpu:
        print("use CPU !")
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("No GPU available !")
            device = torch.device('cpu')
    if not os.path.exists(args.valOutDir):
        os.mkdir(args.valOutDir)

    print("============> Building model ...")
    model = FullNet(tb_writer)
    model.to(device)

    print("============> Loading datasets ...")
    train_loader = make_loader(os.path.join(args.dataDir, 'train_cropped'),
                               transform=transforms_test(), batch_size=6, args=args)
    val_loader = make_loader(os.path.join(args.dataDir, 'handcrafted_cropped'), transform=transforms_test(),
                             batch_size=2, args=args)

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
    test_img, test_trimap, test_alpha = None, None, None

    for epoch in range(start_epoch, args.nEpochs + 1):
        print(f'Epoch {epoch}/{args.nEpochs}')
        train_losses = []
        train_losses_alpha = []
        train_losses_compose = []
        train_losses_trimaps = []
        train_losses_mnet = []
        ce_weights = torch.FloatTensor([1., 1., 2.]).to(device)
        if args.lrdecayType != 'keep':
            lr = set_lr(args, epoch, optimizer)
        model.train(True)
        for i, sample_batched in tqdm(enumerate(train_loader)):
            img, trimap_gt, alpha_gt, src_img = sample_batched['image'], sample_batched['trimap'], sample_batched[
                'alpha'], sample_batched['src']
            trimap_ideal = sample_batched['ideal_trimap']
            img, trimap_gt, alpha_gt, trimap_ideal = img.to(device), trimap_gt.to(device), \
                                                     alpha_gt.to(device), trimap_ideal.to(device)

            if args.train_phase == 'm_net':
                trimap_pre, alpha_pre, alpha_r = model(img, trimap_ideal)
                L_m = loss_m_net(args, alpha_gt, alpha_pre, img)
                loss = L_m
            else:
                trimap_pre, alpha_pre, alpha_r = model(img)


                loss, L_alpha, L_composition, L_cross = loss_function(args,
                                                                  img,
                                                                  trimap_pre,
                                                                  trimap_gt,
                                                                  alpha_pre,
                                                                  alpha_gt,
                                                                  weight=None)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_losses_alpha.append(L_alpha.item() if args.train_phase != 'm_net' else 0)
            train_losses_compose.append(L_composition.item() if args.train_phase != 'm_net' else 0)
            train_losses_trimaps.append(L_cross.item() if args.train_phase != 'm_net' else 0)
            train_losses_mnet.append(L_m.item() if args.train_phase == 'm_net' else 0)


            if i % 50 == 0:
                if args.train_phase != 'm_net':
                    trimap_pre = F.softmax(trimap_pre[0], dim=0)
                tm_pre = trimap_pre[0][1]*0.5+trimap_pre[0][2]
                tm_pre = torch.stack([tm_pre, tm_pre, tm_pre], dim=0)
                tm_gt = torch.cat((trimap_gt[0], trimap_gt[0], trimap_gt[0]), dim=0)
                a_pre = torch.cat((alpha_pre[0], alpha_pre[0], alpha_pre[0]), dim=0)
                a_r = torch.cat((alpha_r[0], alpha_r[0], alpha_r[0]), dim=0)
                a_gt = torch.cat((alpha_gt[0], alpha_gt[0], alpha_gt[0]), dim=0)
                visuals = OrderedDict()
                visuals['1_source_img'] = src_img[0].to(device)
                visuals['2_trimap_predicted'] = tm_pre
                visuals['3_trimap_gt'] = tm_gt / 2
                visuals['4_alpha_M_net'] = a_r
                visuals['5_alpha_predicted'] = a_pre
                visuals['6_alpha_gt'] = a_gt
                display_visuals_tb(tb_writer=tb_writer,
                                   visuals=visuals,
                                   it=epoch,
                                   metalabel=f'train_{i}')

        val_losses = []
        val_losses_alpha = []
        val_losses_compose = []
        val_losses_trimaps = []
        val_losses_mnet = []
        model.train(False)
        if not os.path.exists(os.path.join(args.valOutDir, f'epoch_{epoch}')):
            os.makedirs(os.path.join(args.valOutDir, f'epoch_{epoch}'))
        for i, sample_batched in tqdm(enumerate(val_loader)):
            img, trimap_gt, alpha_gt, src_img = sample_batched['image'], sample_batched['trimap'], sample_batched[
                'alpha'], sample_batched['src']
            trimap_ideal = sample_batched['ideal_trimap']
            img, trimap_gt, alpha_gt, trimap_ideal = img.to(device), trimap_gt.to(device), \
                                                     alpha_gt.to(device), trimap_ideal.to(device)

            if args.train_phase == 'm_net':
                trimap_pre, alpha_pre, alpha_r = model(img, trimap_ideal)
                L_m = loss_m_net(args, alpha_gt, alpha_pre, img)
                loss = L_m
            else:
                trimap_pre, alpha_pre, alpha_r = model(img)
                loss, L_alpha, L_composition, L_cross = loss_function(args,
                                                                  img,
                                                                  trimap_pre,
                                                                  trimap_gt,
                                                                  alpha_pre,
                                                                  alpha_gt)
            val_losses.append(loss.item())
            val_losses_alpha.append(L_alpha.item() if args.train_phase != 'm_net' else 0)
            val_losses_compose.append(L_composition.item() if args.train_phase != 'm_net' else 0)
            val_losses_trimaps.append(L_cross.item() if args.train_phase != 'm_net' else 0)
            val_losses_mnet.append(L_m.item() if args.train_phase == 'm_net' else 0)

            if i % 15 == 0:
                if args.train_phase != 'm_net':
                    trimap_pre = F.softmax(trimap_pre[0], dim=0)
                tm_pre = trimap_pre[0][1] * 0.5 + trimap_pre[0][2]
                tm_pre = torch.stack([tm_pre, tm_pre, tm_pre], dim=0)
                tm_gt = torch.cat((trimap_gt[0], trimap_gt[0], trimap_gt[0]), dim=0)
                a_pre = torch.cat((alpha_pre[0], alpha_pre[0], alpha_pre[0]), dim=0)
                a_r = torch.cat((alpha_r[0], alpha_r[0], alpha_r[0]), dim=0)
                a_gt = torch.cat((alpha_gt[0], alpha_gt[0], alpha_gt[0]), dim=0)
                visuals = OrderedDict()
                visuals['1_source_img'] = src_img[0].to(device)
                visuals['2_trimap_predicted'] = tm_pre
                visuals['3_trimap_gt'] = tm_gt / 2
                visuals['4_alpha_M_net'] = a_r
                visuals['5_alpha_predicted'] = a_pre
                visuals['6_alpha_gt'] = a_gt
                display_visuals_tb(tb_writer=tb_writer,
                                   visuals=visuals,
                                   it=epoch,
                                   metalabel=f'val_{i}')

        validation_epoch_loss = np.mean(np.array(val_losses))
        training_epoch_loss = np.mean(np.array(train_losses))
        tqdm.write(f'Train loss: {training_epoch_loss}\nVal loss: {validation_epoch_loss}')
        tb_writer.add_scalar('Loss/Train', training_epoch_loss, epoch)
        tb_writer.add_scalar('LossAlpha/Train', np.mean(np.array(train_losses_alpha)), epoch)
        tb_writer.add_scalar('LossCompose/Train', np.mean(np.array(train_losses_compose)), epoch)
        tb_writer.add_scalar('LossTrimap/Train', np.mean(np.array(train_losses_trimaps)), epoch)
        tb_writer.add_scalar('Loss_Mnet/Train', np.mean(np.array(train_losses_mnet)), epoch)

        tb_writer.add_scalar('Loss/Val', validation_epoch_loss, epoch)
        tb_writer.add_scalar('LossAlpha/Val', np.mean(np.array(val_losses_alpha)), epoch)
        tb_writer.add_scalar('LossCompose/Val', np.mean(np.array(val_losses_compose)), epoch)
        tb_writer.add_scalar('LossTrimap/Val', np.mean(np.array(val_losses_trimaps)), epoch)
        tb_writer.add_scalar('Loss_Mnet/Val', np.mean(np.array(val_losses_mnet)), epoch)

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
