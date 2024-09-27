# Shot-again
An underwater image enhancement method
# Introduction
Underwater images suffer from great degradation due to the complexity of underwater scene, such as light scatter and absorption. Due to the lack of corresponding reference objects in underwater images, self-supervised learning methods have broad application prospects in underwater image restoration. But underwater image enhancement is an ill-posed problem, and unsupervised learning lacks constraints and considerations on the effectiveness of image enhancement. Here, we proposed a novel framework to adjust the adaptability of unsupervised-learning model during enhancement process of underwater image. We introduced a pseudo triple network structure integrated comparative triplet loss, named Shot-Again, which conclude the importance of prior knowledge within data-driven methods in underwater images enhancement and readjusts the effectiveness of image enhancement.
# Configuration
We use Windows 10 system, Python 3.8, Pytorch 1.13.0 and one NVIDIA RTX 3060 GPU.
# Dataset
The datasets are availabel in https://drive.google.com/file/d/10VGKwG_4YqLclQ2qA78L2qkEImScUOcz/view?usp=sharing.
# For use
The pretrained models are in ./final_weight. You can use it with main.py and evaluate it by evaluate.py.
