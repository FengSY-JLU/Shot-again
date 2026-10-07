import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from data import get_eval_set
from measurement.metrics_utils import calculate_uciqe, calculate_uiqm
from net.net import net
from utils import get_A, my_save_image, torch_to_np


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


class UnpairedEvalDataset(Dataset):
    """Minimal loader for no-reference evaluation."""

    def __init__(self, data_dir):
        self.files = sorted(
            os.path.join(data_dir, name)
            for name in os.listdir(data_dir)
            if name.lower().endswith(IMAGE_EXTENSIONS)
        )
        if not self.files:
            raise RuntimeError(f"No supported images found in: {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(path).convert("RGB")
        return to_tensor(image), os.path.basename(path)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


@torch.no_grad()
def evaluate(model, opt, testing_data_loader, epoch=None):
    model.eval()
    device = next(model.parameters()).device
    no_reference = bool(getattr(opt, "no_reference", False))

    ep_str = str(epoch) if epoch is not None else "final"
    print(f"\n[Evaluation Start] Epoch/Tag {ep_str}")

    base_dir = f"{opt.output_folder.rstrip('/')}_{opt.Indicator}"
    epoch_dir = os.path.join(base_dir, f"epoch_{ep_str}")

    subdirs = ["J", "A", "T", "test", "metrics"]
    if not no_reference:
        subdirs.append("label")

    for sub in subdirs:
        os.makedirs(os.path.join(epoch_dir, sub), exist_ok=True)

    psnr_values, ssim_values = [], []
    uciqe_values, uiqm_values = [], []

    metrics_path = os.path.join(epoch_dir, "metrics", "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"=== Evaluation Results for Epoch/Tag {ep_str} ===\n")
        if no_reference:
            f.write("Image, UCIQE, UIQM\n")
        else:
            f.write("Image, PSNR, SSIM, UCIQE, UIQM\n")

    for batch in testing_data_loader:
        if no_reference:
            input_tensor, name = batch
            input_tensor = input_tensor.to(device)
            img_name = name[0]
            label_tensor = None
        else:
            input_tensor, label_tensor, name = batch
            input_tensor = input_tensor.to(device)
            label_tensor = label_tensor.to(device)
            img_name = name[0]

        j_out, t_out = model(input_tensor)
        a_out = get_A(input_tensor).to(device)

        j_out_np = np.clip(torch_to_np(j_out), 0, 1)
        t_out_np = np.clip(torch_to_np(t_out), 0, 1)
        a_out_np = np.clip(torch_to_np(a_out), 0, 1)
        input_np = np.clip(torch_to_np(input_tensor), 0, 1)

        my_save_image(img_name, input_np, os.path.join(epoch_dir, "test") + "/")
        my_save_image(img_name, j_out_np, os.path.join(epoch_dir, "J") + "/")
        my_save_image(img_name, t_out_np, os.path.join(epoch_dir, "T") + "/")
        my_save_image(img_name, a_out_np, os.path.join(epoch_dir, "A") + "/")

        save_path = os.path.join(epoch_dir, "J", img_name)

        if not no_reference:
            label_np = np.clip(torch_to_np(label_tensor), 0, 1)
            my_save_image(img_name, label_np, os.path.join(epoch_dir, "label") + "/")
            source_path = os.path.join(epoch_dir, "label", img_name)

            img_pred = cv2.imread(save_path)
            img_gt = cv2.imread(source_path)

            psnr = compare_psnr(img_pred, img_gt)
            ssim = compare_ssim(img_pred, img_gt, channel_axis=2)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

        img_tensor = to_tensor(Image.open(save_path).convert("RGB")).cpu()
        uciqe_val = calculate_uciqe(img_tensor)
        uiqm_val = calculate_uiqm(img_tensor)
        uciqe_values.append(uciqe_val)
        uiqm_values.append(uiqm_val)

        if no_reference:
            print(
                f"{img_name} -> UCIQE:{uciqe_val:.4f}, "
                f"UIQM:{uiqm_val:.4f}"
            )
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(f"{img_name}, {uciqe_val:.4f}, {uiqm_val:.4f}\n")
        else:
            print(
                f"{img_name} -> PSNR:{psnr:.4f}, SSIM:{ssim:.4f}, "
                f"UCIQE:{uciqe_val:.4f}, UIQM:{uiqm_val:.4f}"
            )
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{img_name}, {psnr:.4f}, {ssim:.4f}, "
                    f"{uciqe_val:.4f}, {uiqm_val:.4f}\n"
                )

    if no_reference:
        summary = (
            f"\n=== Summary for Epoch/Tag {ep_str} ===\n"
            f"UCIQE_mean: {_mean(uciqe_values):.4f}\n"
            f"UIQM_mean: {_mean(uiqm_values):.4f}\n"
        )
    else:
        summary = (
            f"\n=== Summary for Epoch/Tag {ep_str} ===\n"
            f"PSNR_mean: {_mean(psnr_values):.4f}\n"
            f"SSIM_mean: {_mean(ssim_values):.4f}\n"
            f"UCIQE_mean: {_mean(uciqe_values):.4f}\n"
            f"UIQM_mean: {_mean(uiqm_values):.4f}\n"
        )

    print(summary)
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(summary)

    print(f"[Evaluation Finished] Results saved to {metrics_path}\n")


def _load_state_dict(path, device):
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unsupported checkpoint format: expected a state_dict or a "
            "dictionary containing the key 'state_dict'."
        )

    # Also accept checkpoints saved from DataParallel.
    if checkpoint and all(key.startswith("module.") for key in checkpoint.keys()):
        checkpoint = {
            key[len("module."):]: value for key, value in checkpoint.items()
        }

    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Shot-Again")
    parser.add_argument("--data_test", type=str, required=True)
    parser.add_argument(
        "--label_test",
        type=str,
        default=None,
        help="Ground-truth directory for paired evaluation.",
    )
    parser.add_argument("--output_folder", type=str, default="Results/")
    parser.add_argument("--Indicator", type=str, default="UIEBD")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--epoch-tag",
        type=str,
        default="final",
        help="Tag used in the output directory name.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Evaluate an unpaired dataset and skip PSNR/SSIM.",
    )

    parser.add_argument(
        "--use_spb",
        dest="use_spb",
        action="store_true",
        help="Enable SPB.",
    )
    parser.add_argument(
        "--no-use_spb",
        dest="use_spb",
        action="store_false",
        help="Disable SPB (for ablations).",
    )
    parser.set_defaults(use_spb=True)

    parser.add_argument(
        "--use_sgca",
        dest="use_sgca",
        action="store_true",
        help="Enable SGCA.",
    )
    parser.add_argument(
        "--no-use_sgca",
        dest="use_sgca",
        action="store_false",
        help="Disable SGCA (for ablations).",
    )
    parser.set_defaults(use_sgca=True)

    opt = parser.parse_args()

    if not opt.no_reference and not opt.label_test:
        parser.error("--label_test is required unless --no-reference is used.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = net(use_spb=opt.use_spb, use_sgca=opt.use_sgca).to(device)
    model.load_state_dict(_load_state_dict(opt.model, device), strict=True)

    if opt.no_reference:
        test_set = UnpairedEvalDataset(opt.data_test)
    else:
        test_set = get_eval_set(opt.data_test, opt.label_test)

    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    evaluate(model, opt, loader, epoch=opt.epoch_tag)
