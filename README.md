# Introduction
Many thanks to your interest in ContriMix! We glad you are here :)!
This code repo [ContriMix: Unsupervised disentanglement of content and attribute for domain generalization in microscopy image analysis](https://arxiv.org/abs/2306.04527). ContriMix is an extension of the DRIT++ or HistAuGan, which is an application of DRIT++ on digital pathology data. Different from DRIT++ and HistAuGAN that leverage domain adversarial to learn information about content and attribute representations, ContriMix leverages the difference between training samples in the training mini-batches. This allows ContriMix to be useful with data that does not have domain index. Because ContriMix does not use any adversarial training, there should be no hallucination in synthetic images generated by ContriMix.

## Getting started
- The quickest way to evaluate ContriMix is using our Google [colab notebook](https://colab.research.google.com/drive/1ncXRMgHOijT9Uqr1iy0jqNqo1lnBeP-G?usp=sharing) from a pre-trained ContriMix model.
- If you have an NVIDIA GPU-enabled system, you can also use the following notebook [contrimix_demo.ipynb](./contrimix_demo.ipynb). Add more images that you want to try to the [sample_data](/sample_data/) folder.

## Local installations
1.  Install the NVDIA GPU driver.
2.  Install CUDA. CUDA11.0 and above should work. We tested our code with [CUDA 11.5](https://developer.nvidia.com/cuda-11-5-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)
3.  Install [Anaconda](https://www.anaconda.com/)
4.  Clone the ContriMix code repo `git clone https://gitlab.com/huutan86/intraminibatch_permutation_drit`
5.  Change the directory `cd intraminibatch_permutation_drit`.
6.  Run `pip install -r requirements.txt` to install all dependencies.
7. (Optional) If you want to run the backbone training with the HistAuGan algorithm for augmentation, download the HistAuGan checkpoint from the HistAuGan code repo [link](https://github.com/sophiajw/HistAuGAN) and place it in the `checkpoints/` folder.

## How to run the script.
If you want to run the training scripts for different datasets, look at the followings script:
- TCGA: ContriMix with Data Parallel(DP) ([code](tcga_do_all_unlabeled_contrimix_dp.py)), ContriMix with Distributed Data Parallel (DDP) ([code](/tcga_do_all_unlabeled_contrimix_ddp.py)).
- Camelyon: ERM with DP ([code](/camelyon_17_do_all_erm.py)), ContriMix (jointly trained) with DP ([code](./camelyon_17_do_all_contrimix.py)), ContriMix (jointly trained) with DDP ([code](./camelyon_17_do_all_contrimix_ddp.py)).
- RxRx1: Contrimix encoder training with DP ([code](/rxrx1_do_all_contrimix_encoder_training_dp.py)), Contrimix encoder training with DDP ([code](/rxrx1_do_all_contrimix_encoder_training_ddp.py)), Backbone training using a trained Contrimix encoder ([code](/rxrx1_do_all_contrimix_backbone_training_dp.py)).

## Model zoo
We provide a [model zoo](./MODELZOO.md) containing the trained checkpoints for Contrimix and HistauGAN with different seeds. The model zoo also contains a link to a trained ContriMix model trained on 2.5 millions images that you can use for your project.

## References
[1]. [HistauGAN implementation](https://github.com/sophiajw/HistAuGAN) and the [HistauGAN paper](https://arxiv.org/abs/2107.12357), Wagner, S. J., Khalili, N., Sharma, R., Boxberg, M., Marr, C., de Back, W., Peng, GAN-based augmentation technique for histopathological images presented in the paper "Structure-Preserving Multi-Domain Stain Color Augmentation using Style-Transfer with Disentangled Representations" (2021).

[2]. [MDMM Implementation](https://github.com/HsinYingLee/MDMM) and [DRIT++ paper](/https://arxiv.org/abs/1905.01270). Lee, H.Y., Tseng, H.Y., Mao, Q., Huang, J.B., Lu, Y.D., Singh, M., Yang, M.H.: DRIT : Diverse Image-to-Image translation via disentangled representations (2020).

[3] Tellez, D., Litjens, G., Ba ́ndi, P., Bulten, W., Bokhorst, J.M., Ciompi, F., van der Laak, J.: Quantifying the effects of data augmentation and stain color normaliza- tion in convolutional neural networks for computational pathology. Med. Image Anal. 58, 101544 (Dec 2019)

## Questions and suggestions
Please contact us at <a href="mailto:huutan86@gmail.com">huutan86@gmail.com</a> if you have any questions.


