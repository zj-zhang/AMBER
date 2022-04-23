API Reference
====================

CommandLine Interface
---------------------
If you get quick started on using AMBER, the fastest way is to
write a configuration file and call AMBER's command-line interface.

See this page for details on `CLI page <amber-cli.html>`_.

Basic API
----------

The AMBER package has the following design layout:

- architect: algorithms to search for neural network architectures

- modeler: translates an architecture to a model in existing machine learning frameworks

- utils: utility functions commonly-used in AMBER are stored here

- plots: includes both plots and visualizations for sanity checks in AMBER


Some of the modules are either kept in for legacy usages, or still under
developments.
These include:



List of Modules
-------------

.. autosummary::
    :nosignatures:

    amber.architect
    amber.modeler
    amber.utils
    amber.plots


