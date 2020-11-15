# AMBER

Automated Modeling for Biological Evidence-based Research

August, 2020

Below we provide a short guide to help you get quick started.

The API documentation is still *Work-In-Progress*. More comprehensive docs to follow. Latest docs can be found at:
https://amber-dl.readthedocs.io/en/latest/

### Installation

Currently AMBER is designed to be run in Linux environment. As a prerequisite, please make sure
 you have Anaconda/miniconda installed, as we provide the detailed dependencies through a conda 
 environment.

First clone this GitHub repository:
```
git clone https://github.com/zj-zhang/AMBER.git
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

The typical installation process should take less than 10 minutes with regular network 
connection and hardware specifications. 

### Quick Start

The easist entry point to `AMBER` is by the wrapper class [Amber](#https://github.com/zj-zhang/AMBER/blob/master/amber/_wrapper.py#L12).

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

I am also writing up a set of tips on how to set these arguments. Details to follow soon.



