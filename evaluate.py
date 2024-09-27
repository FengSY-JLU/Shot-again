from __future__ import print_function
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import DataLoader
from net.net import net
from data import get_eval_set
from utils import *
import cv2

from torchvision.transforms.functional import to_tensor
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
import numpy as np

from measurement.AGandIE import *



print(torch.cuda.is_available())
parser = argparse.ArgumentParser(description='Shot-Again')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--data_test', type=str, default='../Dataset/RUIE/test')
parser.add_argument('--label_test', type=str, default='../Dataset/RUIE/test')
parser.add_argument('--model', default='final_weight/RUIE_300.pth', help='Pretrained base model')
parser.add_argument('--output_folder', type=str, default='Results/')
parser.add_argument('--IndicatorPath', type=str, default='RUIE/', help='IndicatorPath Name')
parser.add_argument('--Indicator', type=str, default='RUIE', help='IndicatorPath Name')


opt = parser.parse_args()
opt.cuda = not opt.no_cuda and torch.cuda.is_available()

print('===> Loading datasets')
test_set = get_eval_set(opt.data_test, opt.label_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print('===> Building model')

model = net().cuda()
model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained model is loaded.')


def eval():
    torch.set_grad_enabled(False)
    model.eval()
    print('\nEvaluation:')
    PSNR = []
    SSIM = []
    MSE = []
    IE = []
    AG = []
    for batch in testing_data_loader:
        with torch.no_grad():
            input, label, name = batch[0], batch[1], batch[2]

            print(name)

        input = input.cuda()

        with torch.no_grad():
            j_out, t_out = model(input)
            a_out = get_A(input).cuda()

            print(opt.output_folder)
            if not os.path.exists(opt.output_folder):
                os.mkdir(opt.output_folder)
                os.makedirs(opt.output_folder + 'J/'+ opt.Indicator + '/')
                os.makedirs(opt.output_folder + 'A/'+ opt.Indicator + '/')
                os.makedirs(opt.output_folder + 'T/'+ opt.Indicator + '/')
                os.makedirs(opt.output_folder + 'test/'+ opt.Indicator + '/')
            j_out_np = np.clip(torch_to_np(j_out), 0, 1)
            t_out_np = np.clip(torch_to_np(t_out), 0, 1)
            a_out_np = np.clip(torch_to_np(a_out), 0, 1)
            input_np = np.clip(torch_to_np(input), 0, 1)
            label_np = np.clip(torch_to_np(label), 0, 1)

            my_save_image(name[0], input_np, opt.output_folder + 'test/'+ opt.Indicator + '/')
            my_save_image(name[0], j_out_np, opt.output_folder + 'J/'+ opt.Indicator + '/')
            my_save_image(name[0], t_out_np, opt.output_folder + 'T/'+ opt.Indicator + '/')
            my_save_image(name[0], a_out_np, opt.output_folder + 'A/'+ opt.Indicator + '/')
            my_save_image(name[0], label_np, opt.output_folder + 'label/'+ opt.Indicator + '/')



        save_path = opt.output_folder+'J/'+opt.Indicator+'/'+'{}'.format(name[0])
        # source_path = opt.output_folder+'test/'+opt.Indicator+'/''{}'.format(name[0])
        source_path = opt.output_folder+'label/'+opt.Indicator+'/''{}'.format(name[0])
    ############################ Measurement ###############################
        image =cv2.imread(save_path)
        img1 = image
        img2 = cv2.imread(source_path)


        img3 = Image.open(save_path).convert('RGB')
        img3 = to_tensor(img3).cuda().unsqueeze(0)
        img3 = torch.cat((img3, img3), 0)


        psnr = compare_psnr(img1, img2)
        ssim = compare_ssim(img1, img2, multichannel=True)
        mse = compare_mse(img1, img2)
        ie = calculate_entropy(img1)
        ag = calculate_average_gradient(img1)

        ######################## SSIM,MSE ########################

        print('PSNR:{},SSIM:{},MSE:{},UCIQE:{},UIQM:{}'.format(psnr, ssim, mse))

        ######################## Create File ########################
        keys = ['PSNR', 'SSIM', 'MSE', 'UCIQE', 'UIQM']
        values = [psnr, ssim, mse]
        indicator = dict(zip(keys, values))
        PSNR.append(psnr)
        SSIM.append(ssim)
        MSE.append(mse)
        IE.append(ie)
        AG.append(ag)
        if not os.path.exists(opt.IndicatorPath):
                os.makedirs(opt.IndicatorPath)
        with open(opt.IndicatorPath + opt.Indicator + '.txt', 'a+') as file:
            file.write(str('PIC:{}__PSNR:{},SSIM:{},MSE:{},UCIQE:{},UIQM:{}'.format(name[0], psnr, ssim, mse)) + '\n')
    psnr_mean = sum(PSNR)/len(PSNR)
    ssim_mean = sum(SSIM)/len(SSIM)
    mse_mean = sum(MSE)/len(MSE)
    print (f'psnr_mean:{psnr_mean}, ssim_mean:{ssim_mean}, mse_mean:{mse_mean}')
        ############################ Measurement ###############################

if __name__ == '__main__':
    eval()

