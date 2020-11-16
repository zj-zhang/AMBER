Installation
============

AMBER is developed under Python 3.7 and Tensorflow 1.15.

Please follow the steps below to install AMBER.


Get the latest source code
--------------------------
First, clone the Github Repository; if you have previous versions, make sure you pull the latest commits/changes:

.. code-block:: bash

    git clone https://github.com/zj-zhang/AMBER.git
    cd AMBER
    git pull

If you see `Already up to date` in your terminal, that means the code is at the latest change.

Installing with Anaconda
-------------------------
The easiest way to install AMBER is by ``Anaconda``. It is recommended to create a new conda
environment for AMBER:

.. code-block:: bash

    conda create --file ./templates/conda_amber.linux_env.yml
    python setup.py develop


Testing your installation
-------------------------
You can test if AMBER can be imported to your new `conda` environment like so:

.. code-block:: bash

    conda activate amber
    python -c "import amber"

If no errors pop up, that means you have successfully installed AMBER.

.. todo::

    Run ``unittest`` once its in place.
