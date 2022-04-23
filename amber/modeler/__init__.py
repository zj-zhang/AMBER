"""takes in an architecture, returns the machine learning model
"""

from .modeler import DAGModelBuilder, EnasAnnModelBuilder, EnasCnnModelBuilder

from .modeler import KerasModelBuilder, KerasMultiIOModelBuilder, KerasResidualCnnBuilder, KerasBranchModelBuilder

from . import child, dag, modeler
