# Introduction
This code repo contains the implmentation for Contrimix from the paper [ContriMix: Unsupervised disentanglement of content and attribute for domain generalization in microscopy image analysis](https://arxiv.org/abs/2306.04527).


## Installation on a local desktop.
1. Install Python 3.8.8 (see `pyproject.toml)
2. Install [poetry v1.2.2](https://python-poetry.org/docs/#installing-with-the-official-installer) for package management and development.
3. Install [CUDA 11.5](https://developer.nvidia.com/cuda-11-5-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)
This is required to be compatible with the `torch` and `torchvision` packages.
4. Clone the code repo.
5. Change directory to the folder of the code repo.
6. Create your virtual environment with poetry: `poetry install`. You can use `Anaconda` for this.
Set the `PYTHONPATH` to the folder of the `ip_drit` with `export PYTHONPATH=’[path-to-directory]’`. For example,
`export PYTHONPATH=/jupyter-users-home/tan-2enguyen/intraminibatch_permutation_drit/ip_drit`. If you are using VS Code as an IDE,
press `Command + Shift + P`  ->  `Developer: Reload Window` so that VSCode can see the `ip_drit` package.
7. Activate the virtual environment that you created.
8. Create a folder name `datasets` on your computer, update it to `camelyon_17_do_all.py`.
9. Run `camelyon_17_do_all.py` with `--run_on_cluster False`.

## Contrimix application
1If you simply want to try the pre-trained ContriMix encoder from the TCGA dataset, run [contrimix_demo.ipynb](./contrimix_demo.ipynb). If you have more images to test, add them to the [sample_data](/sample_data/) folder.

## Contrimix training.
If you want to run the training scripts for different datasets, look at the followings script:
- TCGA: ContriMix with Data Parallel(DP) ([code](tcga_do_all_unlabeled_contrimix_dp.py)), ContriMix with Distributed Data Parallel (DDP) ([code](/tcga_do_all_unlabeled_contrimix_ddp.py)).
- Camelyon: ERM with DP ([code](/camelyon_17_do_all_erm.py)), ContriMix (jointly trained) with DP ([code](./camelyon_17_do_all_contrimix.py)), ContriMix (jointly trained) with DDP ([code](./camelyon_17_do_all_contrimix_ddp.py)).
- RxRx1: Contrimix encoder training with DP ([code](/rxrx1_do_all_contrimix_encoder_training_dp.py)), Contrimix encoder training with DDP ([code](/rxrx1_do_all_contrimix_encoder_training_ddp.py)), Backbone training using a trained Contrimix encoder ([code](/rxrx1_do_all_contrimix_backbone_training_dp.py)).

## Model zoo
We provide model zoo storing the training checkpoints for Contrimix and HistauGAN with different seeds. The mdoel zoo for the HistauGAN can be found [here](./MODELZOO.md).
## References
[1]. [HistauGAN implementation](https://github.com/sophiajw/HistAuGAN) and the [HistauGAN paper](https://arxiv.org/abs/2107.12357), Wagner, S. J., Khalili, N., Sharma, R., Boxberg, M., Marr, C., de Back, W., Peng, GAN-based augmentation technique for histopathological images presented in the paper "Structure-Preserving Multi-Domain Stain Color Augmentation using Style-Transfer with Disentangled Representations" (2021).

[2]. [MDMM Implementation](https://github.com/HsinYingLee/MDMM) and [DRIT++ paper](/https://arxiv.org/abs/1905.01270). Lee, H.Y., Tseng, H.Y., Mao, Q., Huang, J.B., Lu, Y.D., Singh, M., Yang, M.H.: DRIT : Diverse Image-to-Image translation via disentangled representations (2020).

[3] Tellez, D., Litjens, G., Ba ́ndi, P., Bulten, W., Bokhorst, J.M., Ciompi, F., van der Laak, J.: Quantifying the effects of data augmentation and stain color normaliza- tion in convolutional neural networks for computational pathology. Med. Image Anal. 58, 101544 (Dec 2019)


