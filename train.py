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
from utils import *
import random
import time
from net.losses import ColorLoss
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import math
import torch.nn as nn
import json
import torch
import numpy as np
from evaluation import evaluate
# ----------------------------
# Training settings & switches
# ----------------------------
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
parser.add_argument('--Margin', type=float, default=0.2, metavar='M', help='(kept for compatibility)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--IndicatorPath', type=str, default='UIEBD/', help='IndicatorPath Name')
parser.add_argument('--Indicator', type=str, default='UIEBD', help='Indicator Name')
parser.add_argument('--model', default='final_weight/UIEBD_final.pth', help='Pretrained base model')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--eval_every', type=int, default=None,
                    help='Every N epochs evaluate once; None to disable.')
parser.add_argument('--eval_epochs', type=str, default=None,
                    help='Comma-separated epoch list, e.g., "10,20,30".')

parser.add_argument('--use_spb', dest='use_spb', action='store_true', help='enable SPB module')
parser.add_argument('--no-use_spb', dest='use_spb', action='store_false', help='disable SPB module')
parser.set_defaults(use_spb=False)

parser.add_argument('--use_sgca', dest='use_sgca', action='store_true', help='enable SGCA module')
parser.add_argument('--no-use_sgca', dest='use_sgca', action='store_false', help='disable SGCA module')
parser.set_defaults(use_sgca=False)


parser.add_argument('--use_Lphys', dest='use_Lphys', action='store_true', help='enable L_phys (ratio+floor+ref)')
parser.add_argument('--no-use_Lphys', dest='use_Lphys', action='store_false', help='disable L_phys')
parser.set_defaults(use_Lphys=False)

parser.add_argument('--alpha', type=float, default=1.0, help='weight for L_ratio term in L_phys')
parser.add_argument('--beta', type=float, default=0.2, help='weight for L_floor term in L_phys')

parser.add_argument('--gamma_ref', type=float, default=0.1, help='weight for L_ref term in L_phys (renamed from --gamma)')
parser.add_argument('--use_Lref', dest='use_Lref', action='store_true', help='enable L_ref (requires label presence)')
parser.add_argument('--no-use_Lref', dest='use_Lref', action='store_false', help='disable L_ref')
parser.set_defaults(use_Lref=False)

parser.add_argument('--eps', type=float, default=1e-6, help='epsilon to avoid div-by-zero in ratio')
parser.add_argument('--delta', type=float, default=0.1, help='margin delta for L_ratio (r* + delta)')
parser.add_argument('--tau', type=float, default=0.1, help='floor threshold tau for L_floor')
parser.add_argument('--p_norm', type=int, default=2, choices=[1,2], help='p norm for distances (1 or 2)')

opt = parser.parse_args()
opt.cuda = not opt.no_cuda and torch.cuda.is_available()

if hasattr(opt, "eval_epochs") and opt.eval_epochs:
    try:
        eval_epochs_set = set(int(x) for x in opt.eval_epochs.split(",") if x.strip())
    except:
        eval_epochs_set = set()
else:
    eval_epochs_set = set()

def seed_torch(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch(opt.seed)
cudnn.benchmark = True

mse_loss = torch.nn.MSELoss().cuda()
color_loss = ColorLoss()

def _per_sample_norm(x, p=2):
    # x: tensor shape (B, C, H, W) -> compute per-sample norm over all non-batch dims
    B = x.size(0)
    view = x.view(B, -1)
    if p == 1:
        return torch.norm(view, p=1, dim=1)   # (B,)
    else:
        return torch.norm(view, p=2, dim=1)   # (B,)

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

        with torch.no_grad():
            ones = torch.ones_like(t_out)

        d_pos = _per_sample_norm(t_out - t_out_mix, p=opt.p_norm)         
        d_neg = _per_sample_norm(t_out - ones, p=opt.p_norm)             

        ratio = d_pos / (d_neg + opt.eps)

        r_star = 1.0 - float(lam)
        L_ratio_per = F.relu(ratio - (r_star + opt.delta))
        L_ratio = L_ratio_per.mean()

        L_floor_per = F.relu(opt.tau - d_neg)
        L_floor = L_floor_per.mean()

        L_ref = torch.tensor(0.0, device=input.device)
        if opt.use_Lref:
            try:
                has_label = (os.path.abspath(opt.label_test) != os.path.abspath(opt.data_test))
            except:
                has_label = False
            if has_label:
                model.eval()
                with torch.no_grad():
                    _, t_ref = model(label)
                model.train()
                L_ref = mse_loss(t_out, t_ref.detach())
            else:
                L_ref = torch.tensor(0.0, device=input.device)

        L_phys = opt.alpha * L_ratio + opt.beta * L_floor + opt.gamma_ref * L_ref

        loss_C = color_loss(j_out)

        if opt.use_Lphys:
            total_loss = 1.0 * loss_H + 1.0 * loss_R + 0.01 * loss_C + 0.01 * L_phys
        else:
            total_loss = 1.0 * loss_H + 1.0 * loss_R + 0.01 * loss_C

        optimizer.zero_grad()
        total_loss.backward()
        epoch_loss += total_loss.item()
        optimizer.step()

        t1 = time.time()
        print("===> Epoch[{}]({}/{}): Loss: {:.6f} | L_phys:{:.6f} (ratio:{:.6f}, floor:{:.6f}, ref:{:.6f}) | SPB:{} | SGCA:{} | Lphys:{} | GT:{} | lr={} | {:.4f}s".format(
            epoch, iteration, len(training_data_loader), total_loss.item(),
            L_phys.item(), L_ratio.item(), L_floor.item(), (L_ref.item() if isinstance(L_ref, torch.Tensor) else 0.0),
            'ON' if opt.use_spb else 'OFF',
            'ON' if opt.use_sgca else 'OFF',
            'ON' if opt.use_Lphys else 'OFF',
            'Yes' if (os.path.abspath(opt.label_test) != os.path.abspath(opt.data_test)) else 'No',
            optimizer.param_groups[0]['lr'], (t1 - t0)))

def checkpoint(ep):
    model_out = opt.save_folder + (opt.IndicatorPath or '') + opt.Indicator + '/'
    model_out_path = model_out + f"epoch_{ep}.pth"
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
model = net(use_spb=opt.use_spb, use_sgca=opt.use_sgca)
model.cuda()

criterion = torch.nn.MarginRankingLoss(margin=opt.Margin)

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

milestones = []
for i in range(1, opt.nEpochs + 1):
    try:
        if i % opt.decay == 0:
            milestones.append(i)
    except Exception:
        pass
scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)

if __name__ == '__main__':
    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        train()
        scheduler.step()

        do_eval = False
        if opt.eval_every is not None and opt.eval_every > 0 and (epoch % opt.eval_every == 0):
            do_eval = True
        if eval_epochs_set and (epoch in eval_epochs_set):
            do_eval = True

        if do_eval:
            evaluate(model, opt, testing_data_loader, epoch)

        try:
            snap = int(opt.snapshots)
        except:
            snap = 0
        if snap > 0:
            if epoch % snap == 0:
                checkpoint(epoch)
        if epoch == opt.nEpochs:
            checkpoint(epoch)