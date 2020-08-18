# AMBER

Automated Modeling for Biological Evidence-based Research

August, 2020


Documentation is still *Work-In-Progress*. More comprehensive docs to follow. Latest docs can be found at:
https://amber-dl.readthedocs.io/en/latest/


### Installation

Currently AMBER is designed to be run in Linux environment. As a prerequisite, please make sure
 you have Anaconda/miniconda installed.

First clone this GitHub repository:
```
git clone git@github.com:zj-zhang/AMBER.git
```

The dependencies can be installed by
```
cd AMBER
conda env create -f ./templates/conda_amber.linux_env.yml
```

Finally, install `AMBER` by
```
python setup.py develop
```


### Quick Start

Please refer to the template file for running transcriptional regulation prediction tasks using Convolutional Neural networks: [here](https://github.com/zj-zhang/AMBER/blob/master/templates/AmberDeepSea.py)


