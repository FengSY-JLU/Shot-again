import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import DataLoader
from net.net import net
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from data import get_training_set, get_eval_set
from torchvision.transforms.functional import to_tensor
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from measurement.uciqe import torch_uciqe
from measurement.uiqm import torch_uiqm
from utils import *
import random
import time
from net.losses import ColorLoss
from net.losses import ColorLossWithRegularization
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import math
import torch.nn as nn
from evaluate import nmetrics
import json

# Training settings
parser = argparse.ArgumentParser(description='PyTorch UIE')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='10000', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--data_train', type=str, default='../Dataset/UIE/UIEBD/train/image')
parser.add_argument('--label_train', type=str, default='../Dataset/UIE/UIEBD/train/label')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--data_test', type=str, default='../Dataset/UIE/UIEBD/test/image')
parser.add_argument('--label_test', type=str, default='../Dataset/UIE/UIEBD/test/label')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped HR image')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--output_folder', default='Results/', help='Location to save images')
parser.add_argument('--Margin', type=float, default=0.2, metavar='M', help='margin for triplet loss (default: 0.2)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--IndicatorPath', type=str, default='UIEBD/', help='IndicatorPath Name')
parser.add_argument('--Indicator', type=str, default='UIEBD', help='IndicatorPath Name')
parser.add_argument('--model', default='final_weight/UIEBD_final.pth', help='Pretrained base model')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')


opt = parser.parse_args()

opt.cuda = not opt.no_cuda and torch.cuda.is_available()

def seed_torch(seed=opt.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_torch()
cudnn.benchmark = True

mse_loss = torch.nn.MSELoss().cuda()
color_loss = ColorLoss()

def train():
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, label = batch[0], batch[1]

        input = input.cuda()
        label = label.cuda()

        t0 = time.time()
        j_out, t_out = model(input)

        a_out = get_A(input).cuda()
        I_rec = j_out * t_out + (1 - t_out) * a_out
        loss_R = mse_loss(I_rec, input)

        lam = np.random.beta(1, 1)
        input_mix = lam * input + (1 - lam) * j_out

        j_out_mix, t_out_mix = model(input_mix)
        loss_H = mse_loss(j_out_mix, j_out.detach())

        t_max = torch.ones(t_out.shape).cuda()
        dist_a = F.pairwise_distance(t_out, t_max, 2)
        dist_b = F.pairwise_distance(t_out, t_out_mix, 2)

        target = torch.FloatTensor(dist_a.size()).fill_(1)
        if opt.cuda:
            target = target.cuda()
        target = Variable(target)


        loss_triplet = criterion(dist_a, dist_b, target)
        loss_embedd = input.norm(2) + j_out.norm(2) + input_mix.norm(2)
        loss_T = loss_triplet + 0.001 * loss_embedd


        loss_C = color_loss(j_out)

        total_loss = 1*loss_H+1*loss_R+0.01*loss_C+0.01*loss_T


        optimizer.zero_grad()
        total_loss.backward()
        epoch_loss += total_loss.item()
        optimizer.step()
        t1 = time.time()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={} || Timer: {:.4f} sec.".format(epoch,
                iteration, len(training_data_loader), total_loss.item(), optimizer.param_groups[0]['lr'], (t1 - t0)))



def checkpoint(epoch):

    model_out = opt.save_folder+opt.IndicatorPath+opt.Indicator+'/'
    model_out_path = opt.save_folder+opt.IndicatorPath+opt.Indicator+'/'+"epoch_{}.pth".format(epoch)
    if not os.path.exists(model_out):
                os.makedirs(model_out)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")


print('===> Loading datasets')

test_set = get_eval_set(opt.data_test, opt.label_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

train_set = get_training_set(opt.data_train, opt.label_train, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ')

model = net()
model.cuda()

criterion = torch.nn.MarginRankingLoss(margin = opt.Margin)

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

milestones = []
for i in range(1, opt.nEpochs+1):
    if i % opt.decay == 0:
        milestones.append(i)


scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)


if __name__ == '__main__':

    for epoch in range(opt.start_iter, opt.nEpochs + 1):

        train()
        scheduler.step()

        if (epoch) % opt.snapshots == 0:
            checkpoint(epoch)


