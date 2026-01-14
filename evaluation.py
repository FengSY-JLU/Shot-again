import os
import cv2
import torch
import numpy as np
from utils import *
from torchvision.transforms.functional import to_tensor
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from PIL import Image

from measurement.metrics_utils import (
    calculate_uciqe,
    calculate_uiqm,
)


@torch.no_grad()
def evaluate(model, opt, testing_data_loader, epoch=None):

    model.eval()
    ep_str = f"{epoch}" if epoch is not None else "N/A"
    print(f'\n[Evaluation Start] Epoch {ep_str}')


    base_dir = f"{opt.output_folder.rstrip('/')}_{opt.Indicator}"
    epoch_dir = os.path.join(base_dir, f"epoch_{ep_str}")
    for sub in ['J', 'A', 'T', 'test', 'label', 'metrics']:
        os.makedirs(os.path.join(epoch_dir, sub), exist_ok=True)

    PSNR, SSIM = [], []
    UCIQE, UIQM = [], []


    metrics_path = os.path.join(epoch_dir, 'metrics', 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Evaluation Results for Epoch {ep_str} ===\n")
        f.write("Image, PSNR, SSIM, UCIQE, UIQM\n")


    for batch in testing_data_loader:
        input, label, name = batch[0].cuda(), batch[1].cuda(), batch[2]
        img_name = name[0]

        j_out, t_out = model(input)
        a_out = get_A(input).cuda()

        j_out_np = np.clip(torch_to_np(j_out), 0, 1)
        t_out_np = np.clip(torch_to_np(t_out), 0, 1)
        a_out_np = np.clip(torch_to_np(a_out), 0, 1)
        input_np = np.clip(torch_to_np(input), 0, 1)
        label_np = np.clip(torch_to_np(label), 0, 1)

        my_save_image(img_name, input_np, os.path.join(epoch_dir, 'test') + '/')
        my_save_image(img_name, j_out_np,  os.path.join(epoch_dir, 'J') + '/')
        my_save_image(img_name, t_out_np,  os.path.join(epoch_dir, 'T') + '/')
        my_save_image(img_name, a_out_np,  os.path.join(epoch_dir, 'A') + '/')
        my_save_image(img_name, label_np,  os.path.join(epoch_dir, 'label') + '/')

        save_path   = os.path.join(epoch_dir, 'J',     img_name)
        source_path = os.path.join(epoch_dir, 'label', img_name)

        img_pred = cv2.imread(save_path)
        img_gt   = cv2.imread(source_path)

        psnr = compare_psnr(img_pred, img_gt)
        ssim = compare_ssim(img_pred, img_gt, channel_axis=2)

        img_tensor = to_tensor(Image.open(save_path).convert('RGB')).cpu()
        uciqe_val = calculate_uciqe(img_tensor)
        uiqm_val  = calculate_uiqm(img_tensor)

        print(f"{img_name} -> PSNR:{psnr:.4f}, SSIM:{ssim:.4f}, "
              f"UCIQE:{uciqe_val:.4f}, UIQM:{uiqm_val:.4f}")

        PSNR.append(psnr)
        SSIM.append(ssim)
        UCIQE.append(uciqe_val)
        UIQM.append(uiqm_val)


        with open(metrics_path, 'a', encoding='utf-8') as f:
            f.write(f"{img_name}, {psnr:.4f}, {ssim:.4f}, "
                    f"{uciqe_val:.4f}, {uiqm_val:.4f}\n")

    def _mean(x): return sum(x) / len(x) if len(x) > 0 else 0
    summary = (
        f"\n=== Summary for Epoch {ep_str} ===\n"
        f"PSNR_mean: {_mean(PSNR):.4f}\n"
        f"SSIM_mean: {_mean(SSIM):.4f}\n"
        f"UCIQE_mean: {_mean(UCIQE):.4f}\n"
        f"UIQM_mean: {_mean(UIQM):.4f}\n"
    )
    print(summary)
    with open(metrics_path, 'a', encoding='utf-8') as f:
        f.write(summary)

    print(f"[Evaluation Finished] Results saved to {metrics_path}\n")


if __name__ == '__main__':
    from data import get_eval_set
    from net.net import net
    import torch
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_test', type=str, default='../Dataset/RUIE/test')
    parser.add_argument('--label_test', type=str, default='../Dataset/RUIE/test')
    parser.add_argument('--output_folder', type=str, default='Results/')
    parser.add_argument('--Indicator', type=str, default='RUIE')
    parser.add_argument('--model', type=str, default='final_weight/RUIE_300.pth')
    opt = parser.parse_args()

    model = net().cuda()
    model.load_state_dict(torch.load(opt.model, map_location=lambda s, l: s))
    test_set = get_eval_set(opt.data_test, opt.label_test)
    loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    evaluate(model, opt, loader, epoch=300)

