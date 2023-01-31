# Content Attribute Mixing (ConTriMix) Augmentation.
This is the code repo for intrabatch permutation for disentangling the content with the attribute.


## Installation
### System requirement
The system requirements for the code repo. include
- Python 3.8.8 (see `pyproject.toml`)
- [poetry v1.2.2](https://python-poetry.org/docs/#installing-with-the-official-installer) for package management and development.
- [CUDA 11.5](https://developer.nvidia.com/cuda-11-5-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)
This is required to be compatible with the `torch` and `torchvision` packages.

### General setup instructions.
1.  Ensure that all system requirements above are satisfied.
2.  Clone the code repo.
3.  Change into the folder of the code repo.
4.  Create your virtual environment with poetry: `poetry install`.
5.  Set the `PYTHONPATH` to the folder of the `ip_drit` with `export PYTHONPATH=’path/to/directory’`. For example,
`export PYTHONPATH=/jupyter-users-home/tan-2enguyen/intraminibatch_permutation_drit/ip_drit`. If you are using VS Code as an IDE,
press `Command + Shift + P`  ->  `Developer: Reload Window` so that VS Code can see the `ip_drit` folder for correct import.


## How to run the code.
### Run it locally on your computer
1. Checkout the branch `breast_cancer_2` from `origin`.
2. (If you haven't done so) Build a local docker image with `docker-compose build local`.
3. Configure your Pycharm to use the local docker image following the guide [here](https://confluence.services.pathai.com/display/MLPLATFORM/Tutorial%3A+Local+Development).
4. Create a folder name `datasets` on your computer. Update the path to this folder to `camelyon_17_do_all.py`.
5. Run `camelyon_17_do_all.py`.

### Run it from the cluster
1. Configure your virtual environment to use a Jabba image of the `breast_cancer_2` branch. This image has all Python packages that you need. You can do this with `mle use-image [env-name] breast_cancer_2`. Also, change your node to 2 at least (4 is better if you can get it).
2. Connect with your environment with `mle connect [env-name]`.
3. Clone the code repo to the virtual environment, suggesting your folder in `/jupyter-users-home/[your-name]`. This is because it can be maintained across
different environments.
4. Create a location for the dataset, suggesting `/jupyter-users-home/[your-name]/datasets`, update the path to `camelyon_17_do_all.py`.
4. Run `camelyon_17_do_all.py`.





