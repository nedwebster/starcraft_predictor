import setuptools

import version


with open("README.md", "r") as read_me:
    long_description = read_me.read()

setuptools.setup(
    name="starcraft_predictor",
    version=version.__version__,
    author="Ned Webster & Roland Webster",
    author_email="edwardpwebster@gmail.com",
    description="Model to predict starcraft 2 win probability from a replay",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nedwebster/starcraft_predictor",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
