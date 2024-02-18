"""Installing the Contrimix to the site-package path.

In general, it is best not to install `Contrimix` if you are developing it. This is because it is necessary to reinstall the
package whenever a change is made, which can affect the development speed.

Reference: https://packaging.python.org/en/latest/tutorials/packaging-projects/
"""
from setuptools import setup

_PACKAGE_NAME = "contrimix"
_VERSION = "1.0.0"

if __name__ == "__main__":
    print(f"Building the {_PACKAGE_NAME} package, version {_VERSION}")
    setup(
        name=_PACKAGE_NAME,
        version=_VERSION,
        description="ContriMix package",
        license="MIT",
        author="Tan H. Nguyen, PhD and the ContriMix team",
        author_email="huutan86@gmail.com",
        packages=["ip_drit"],
        install_requires=["wheel"],
    )
