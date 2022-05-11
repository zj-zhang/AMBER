amber.architect
=======================

.. contents:: Contents
    :local:

Rationale
----------

This module :mod:`amber.architect` can be devided into four major components. It's easy to imagine
a model space to sample models from, and a training environment as a placeholder for
different components to interact. Currently the model space is pre-defined, but we can make it
dynamic and evolving in the future.

The remaining two components do the most of the heavy-lifting. Search algorithms, such as a controller
recurrent neural network vs a genetic algorithm, are separated by sub-packages (a folder with ``__init__.py``)
and each variant of a search algorithm is a separate file. This makes it easy to re-use code within
a search algorithm, relatively independent across different searchers, and let them share the same
configurations of model space and training environments (although it's possible only certain combinations
of searcher - model space - train env are viable).

The manager is perhaps the most tedious. The general workflow is:

    1. takes a model architecture :obj:`arch` as input,
    2. passes :obj:`arch` to the modeler :obj:`model=amber.modeler.model_fn(arch)`,
    3. trains the model :obj:`model.fit(train_data)`,
    4. evaluates the model's reward :obj:`reward=reward_fn(model)`,
    5. stores the model in a buffer :obj:`buffer_fn.store(model, reward)`,
    6. returns the reward signal to the search algorithm.

Each of the steps has variantions, but the overall layout should almost always stay as described above.


Model Space
---------------------------------

.. automodule:: amber.architect.modelSpace
   :members:
   :undoc-members:
   :show-inheritance:


Train Environment
-------------------------------

.. automodule:: amber.architect.trainEnv
   :members:
   :undoc-members:
   :show-inheritance:


Controller(s)
---------------------------------

.. automodule:: amber.architect.controller
   :members:
   :undoc-members:
   :show-inheritance:

General Controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: amber.architect.controller.generalController
   :members:
   :undoc-members:
   :show-inheritance:

MultiIO Controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: amber.architect.controller.multiioController
   :members:
   :undoc-members:
   :show-inheritance:

Operation Controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: amber.architect.controller.operationController
   :members:
   :undoc-members:
   :show-inheritance:

Zero-Shot Controller (AMBIENT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: amber.architect.controller.zeroShotController
   :members:
   :undoc-members:
   :show-inheritance:


Manager
------------------------------

.. automodule:: amber.architect.manager
   :members:
   :undoc-members:
   :show-inheritance:

Buffer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: amber.architect.buffer
   :members:
   :undoc-members:
   :show-inheritance:

Reward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: amber.architect.reward
   :members:
   :undoc-members:
   :show-inheritance:

Store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: amber.architect.store
   :members:
   :undoc-members:
   :show-inheritance:

Common Operations
--------------------------------

.. automodule:: amber.architect.commonOps
   :members:
   :undoc-members:
   :show-inheritance:

