# Shot-again
An underwater image enhancement method
# Abstract
Underwater images often suffer from severe quality degradation due to light scattering and absorption. Since it is highly impractical to synthesize real-world underwater scenes paired with their ground-truth clean counterparts, unsupervised and self-supervised learning have become mainstream paradigm for underwater image enhancement. However, existing self-supervised approaches typically rely on loosely constrained proxy objectives, which results in suboptimal convergence and leads to either under-enhancement or over-exposure. In addition, the effectiveness of feature learning in current self-supervised frameworks is hindered by accumulated noise within intermediate representations of conventional CNN architectures.
To address these issues, we propose \textit{Shot-Again}, an enhanced self-supervised learning framework that introduces a physical-consistency loss to better regulate the proxy task and guide the model toward stable and reliable optimization. Furthermore, a spatial group cross-channel attention module is designed to suppress noise-correlated local features, and a structure-preserving block is incorporated to reinforce structural fidelity during the restoration process. Comprehensive experiments on widely-used benchmark datasets demonstrate that \textit{Shot-Again} consistently improves visual quality and metric performance within the self-supervised learning paradigm for underwater image enhancement.
# Configuration
We use Python==3.13.2, PyTorch==2.7.1. training on 3060 GPU.
# Datasets
Datasets are avaliable in https://drive.google.com/file/d/1wt-nO6-HIT72p70SIAQtUzyOYYywEImM/view?usp=drive_link
