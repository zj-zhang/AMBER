[![Documentation Status](https://readthedocs.org/projects/amber-automl/badge/?version=latest)](https://amber-automl.readthedocs.io/en/latest/?badge=latest)
[![Coverage](https://raw.githubusercontent.com/zj-zhang/AMBER/master/tests/coverage.svg)](https://github.com/zj-zhang/AMBER/tree/master/tests)
[![Latest Release](https://img.shields.io/github/release/zj-zhang/AMBER.svg?label=Release)](https://github.com/zj-zhang/AMBER/releases/latest)
[![Downloads](https://pepy.tech/badge/amber-automl)](https://pepy.tech/project/amber-automl)
[![DOI](https://zenodo.org/badge/260604309.svg)](https://zenodo.org/badge/latestdoi/260604309)
<!-- 
[![PyPI Install](https://img.shields.io/pypi/dm/amber-automl.svg?label=PyPI%20Installs)](https://pypi.org/project/amber-automl/)
[![Github All Releases](https://img.shields.io/github/downloads/zj-zhang/AMBER/total.svg?label=Download)](https://github.com/zj-zhang/AMBER/releases)
-->

![logo](docs/source/_static/img/amber-logo.png)

---

**Automated Modeling for Biological Evidence-based Research**

<a id='sec1'></a>
AMBER is a toolkit for designing high-performance neural network models automatically in
Genomics and Bioinformatics.

🧐**AMBER can be used to automatically build:**
- 🟢 Convolution neural networks
- 🟢 Sparsified feed-forward neural network
- 🟡 Transfer learning
- 🟡 Kinetics-interpretable neural network
- 🟡 Symbolic explainable AI [WIP]
- 🔴 Graph neural network [WIP]


🤝**Supported backend deep-learning libraries:**
- 🟢 Tensorflow 1.X / Keras
- 🟡 PyTorch / Pytorch-Lightning
- 🟡 Tensorflow 2

*Legend*
🟢: Running & Tested; 🟡: Release soon; 🔴: Work in Progress

---

The overview, tutorials, API documentation can be found at:
https://amber-automl.readthedocs.io/en/latest/

To get quick started, see this [example](https://github.com/zj-zhang/AMBER/blob/master/examples/digits_sklearn/digits_sklearn.py) on handwritten digits classification, or use this [example](https://colab.research.google.com/gist/zj-zhang/48689d8bdc8adf3375719911f7e41989/amber-epigenetics-tutorial-v2.ipynb) on DeepSEA. 
<a href="https://colab.research.google.com/gist/zj-zhang/48689d8bdc8adf3375719911f7e41989/amber-epigenetics-tutorial-v2.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Finally, you can read the AMBER paper for epigenetics regulatory modelling published in [Nature Machine Intelligence](https://www.nature.com/articles/s42256-021-00316-z).


<a id='sec0'></a>
## Table of Contents
1. [Introduction](#sec1)
2. [Installation](#sec2)
3. [Quick Start](#sec3)
4. [Contact & References](#sec4)



<a id='sec2'></a>
## Installation

Currently AMBER is designed to be run in Unix-like environment. As a prerequisite, please make sure
 you have Anaconda/miniconda installed, as we provide the detailed dependencies through a conda 
 environment.
 

Please follow the steps below to install AMBER. To install `AMBER`, 
you can use `conda` and `pypi` to install a versioned release (recommended).

> NOTE:
We strongly encourage you to create a new conda environment, regardless of the backend library you choose.


### Installing with TF 1.X/Keras
In the command-line terminal, type the following commands to get it installed:

```{bash}
conda create -n amber-tf1 -c anaconda tensorflow-gpu=1.15.0 keras scikit-learn numpy~=1.18.5 h5py~=2.10.0 matplotlib seaborn
# if you don't have CUDA-enabled GPU, or on MacOS, replace tensorflow-gpu=1.15.0 with tensorflow=1.15.0
conda activate amber-tf1
pip install amber-automl
# if you plan to run tests
pip install pytest coverage parameterized pydot graphviz
```

### Installing with PyTorch/Lightning
```{bash}
conda create -n amber-torch -c conda-forge pytorch=1.11.0 scikit-learn numpy scipy matplotlib seaborn tqdm h5py
conda activate amber-torch
pip install pytorch-lightning==1.6.5 torchmetrics==0.11.0 amber-automl
# if you plan to run tests
pip install pytest coverage parameterized expecttest hypothesis
```

### Installing with Tensorflow 2
```{bash}
conda create -n amber-tf2 -c conda-forge tensorflow-gpu scikit-learn seaborn
# if you are on MacOS, or don't have CUDA-enabled GPU, replace tensorflow-gpu with tensorflow
conda activate amber-tf2
pip install pytorch-lightning==1.6.5 torchmetrics==0.11.0 amber-automl
# if you plan to run tests
pip install pytest coverage parameterized pydot graphviz
```

### Switching between Backends
```{bash}
amber-cli config --backend pytorch
```

A second approach is to temporarily append an ENV variable, such as 

```{bash}
AMBBACKEND=tensorflow_1 amber-cli run -config config.pkl -data data.h5
```


### Get the latest source code
First, clone the Github Repository; if you have previous versions, make sure you pull the latest commits/changes:

```
git clone https://github.com/zj-zhang/AMBER.git
cd AMBER
git pull
python setup.py develop
```

If you see `Already up to date` in your terminal, that means the code is at the latest change.


### Testing your installation
You can test if AMBER can be imported to your new `conda` environment by:

```bash
conda activate amber
amber-cli --version
```

If the version number is printed out, and no errors pop up, that means you have successfully installed AMBER.

The typical installation process should take less than 10 minutes with regular network 
connection and hardware specifications. 

[Back to Top](#sec0)


<a id='sec3'></a>
## Quick Start

The easist entry point to `AMBER` is by following the tutorial 
in Google colab, where you can run in a real-time, free GPU 
environment.
- Tutorial is here: https://amber-automl.readthedocs.io/en/latest/resource/tutorials.html
- Open google colab notebook here. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/zj-zhang/43235f916303284fdf8c42a6e3d7b8b4)

In a nutshell, to run `Amber` to build a Convolutional neural network, you will only need to provide the file 
paths to compiled training and validation dataset, and specify the input and output shapes. The output of
AMBER is the optimal model architecture for this search run, and a full history of child models during architecture search.

Once you modified the string of file paths, The canonical way of triggering an AMBER 
run is simply:
```python
from amber import Amber
# Here, define the types and specs using plain english in a dictionary
# You can refer to the examples under "template" folder
amb = Amber(types=type_dict, specs=specs)
amb.run()
```
Please refer to the template file for running transcriptional regulation prediction tasks using Convolutional Neural networks: [here](https://github.com/zj-zhang/AMBER/blob/master/templates/AmberDeepSea.py)

Meanwhile, there are more if one would like to dig more. Going further, two levels of
settings are central to an `Amber` instance: a `types` dictionary and a `specs` dictionary. 
- The `types` dictionary will tell `Amber` which types of components (such as controller and
training environment) it should be using.
- The `specs` will further detail every possible settings for the `types` you specified. Only
use this as an expert mode.

[Back to Top](#sec0)


<a id='sec4'></a>
## Contact
If you encounter any issues and/or would like feedbacks, please leave a [GitHub issue](https://github.com/zj-zhang/AMBER/issues).
We will try to get back to you as soon as possible.

If you find AMBER useful in your research, please cite the following paper:

Zhang Z, Park CY, Theesfeld CL, Troyanskaya OG. An automated framework for efficiently designing deep convolutional neural networks in genomics. Nature Machine Intelligence. 2021 Mar 15:1-9. [Paper](https://www.nature.com/articles/s42256-021-00316-z) [Preprint](https://www.biorxiv.org/content/10.1101/2020.08.18.251561v1.full)

[Back to Top](#sec0)







