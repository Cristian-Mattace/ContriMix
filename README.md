# Content Attribute Mixing (ConTriMix) Augmentation.
This is the code repo for content-attribute mixing (ContriMix) augmentation

## How to run the code.
### Using the CPU on your MAC
1. Checkout the branch `breast_cancer_2` from `origin`.
2. (If you haven't done so) Build a local docker image with `docker-compose build local`, see the following guide [here](https://confluence.services.pathai.com/display/MLPLATFORM/Tutorial%3A+Local+Development).
3. Configure your Pycharm to use the local docker image.
4. Change your local virtual environment to `py310`. It is the ml-platform environment that you normally used for ml-platform code development.
5. Change the directory to the folder for this code. Then, install pre-commit with `pre-commit install`.
6. Create a folder name `datasets` on your computer. Update the path to this folder to `camelyon_17_do_all.py`.
7. Run `camelyon_17_do_all.py` with `--run_on_cluster False`.

### Using PathAI's computing cluster
1. Configure your virtual environment to use a Jabba image of the `breast_cancer_2` branch. This image has all Python packages that you need. You can do this with `mle use-image [env-name] breast_cancer_2`. Also, change your node to 2 at least (4 is better if you can get it).
2. Connect with your environment with `mle connect [env-name]`.
3. Clone the code repo to the virtual environment, suggesting your folder in `/jupyter-users-home/[your-name]`. This is because it can be maintained across
different environments.
4. Create a location for the dataset, suggesting `/jupyter-users-home/[your-name]/datasets`, update the path to `camelyon_17_do_all.py`.
5. For using <mle run> command, append the path to the repo in your env in the beginning of camelyon_17_do_all.py
6. Run `camelyon_17_do_all.py` with `--run_on_cluster True`. Alternatively, use mle run <env name> @<local path to camelyon_17_do_all.py> if you want to run this as a mle job in your env. Make sure to associate appropriate number of GPUs in your env before doing that.
Example job - https://app.mle.pathai.com/jobs/view/1ee3c5fa-bbeb-4eda-a2b2-5ebb87f7cc44/status
7. The script above can be run in evaluation mode as well. Example command -  ```python /jupyter-users-home/dinkar-2ejuyal/intraminibatch_permutation_drit/camelyon_17_do_all.py
--dataset_dir_cluster /jupyter-users-home/dinkar-2ejuyal/datasets/ --log_dir_cluster /jupyter-users-home/dinkar-2ejuyal/all_log_dir --use_full_dataset True --eval_only True --model_prefix /jupyter-users-home/dinkar-2ejuyal/all_log_dir/erm_camelyon```
8. Once evaluation has been run on multiple seeds, you can use this script to aggregate the results of these evaluation runs on the seeds and splits of your choice-
``` python evaluate.py --predictions_dir /jupyter-users-home/dinkar-2ejuyal/all_log_dir/erm_camelyon --output_dir /jupyter-users-home/dinkar-2ejuyal/all_log_dir/erm_camelyon/test_results --root_dir /jupyter-users-home/dinkar-2ejuyal/datasets/camelyon17 --run_on_splits val --run_on_seeds 0```


### Using a Ubuntu desktop with GPU (not recommended)
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



