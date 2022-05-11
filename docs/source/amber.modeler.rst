amber.modeler
=====================

.. contents:: Contents
    :local:


Rationale
-----------

This module :mod:`amber.modeler` provides class and interfaces to convert an architecture (usually
a list of tokens/strings) to a model. For simple, sequential models, it's sufficient to use the
out-of-box :mod:`tf.keras.Sequential` for implementations. However, more classes are needed for
advanced architectures, such as conversion of enas super-net to sub-nets.

On the high level, we first need an analog of :mod:`tf.keras.Sequential` that returns a model object
when called; in AMBER, this is :class:`amber.modeler.ModelBuilder` and its subclasses.
To wrap around different implementations of neural networks (e.g., a sequential keras model
vs. a sub-net of enas implemented in tensorflow), `ModelBuilder` will take
:mod:`amber.architect.ModelSpace` as the unifying reference of model architectures, so that
different implementations frameworks, like tensorflow vs keras vs pytorch, will look the same to
the search algorithms in :mod:`amber.architect` to ease its burden.

Moving one level further, we need an analog of :class:`tf.keras.Model` to facilitate the training
and evaluation as class methods. This is implemented by :mod:`amber.modeler.child`.

Under the hood of child models, the corresponding tensor operations and computation graphs are constructed
in module :mod:`amber.modeler.dag`. Currently AMBER builds the enas sub-graphs with keras models, and builds
branching keras model and multi-input/output keras model. Next steps include construction of pytorch computation graphs.



Model Builders
--------------------------------

enasModeler
~~~~~~~~~~~

.. automodule:: amber.modeler.enasModeler
   :members:
   :undoc-members:
   :show-inheritance:

kerasModeler
~~~~~~~~~~~~

.. automodule:: amber.modeler.kerasModeler
   :members:
   :undoc-members:
   :show-inheritance:

Child Models: Training Interface
--------------------------------

.. automodule:: amber.modeler.child
   :members:
   :undoc-members:
   :show-inheritance:

DAG: Computation Graph for Child Models
----------------------------------------

.. automodule:: amber.modeler.dag
   :members:
   :undoc-members:
   :show-inheritance:

Architecture Decoder
----------------------------------------

.. automodule:: amber.modeler.architectureDecoder
   :members:
   :undoc-members:
   :show-inheritance:


