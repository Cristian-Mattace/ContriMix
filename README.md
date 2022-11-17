# Intraminibatch_permutation_DRIT
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





