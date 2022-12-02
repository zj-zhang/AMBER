"""The base class for modelers live here
"""
from typing import Tuple, List, Union
from collections import OrderedDict
import os
from argparse import Namespace
from typing import Optional, Union, List, Dict, Any

import pandas as pd

import copy
try:
    from tensorflow.keras.models import Model
    has_tf = True
except ImportError:
    Model = object
    has_tf = False

try:
    import pytorch_lightning as pl
    from pytorch_lightning import LightningModule
    import torch
    import torch.nn.functional as F
    from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
    from pytorch_lightning.utilities.logger import _add_prefix, _convert_params
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
    has_torch = True
except ImportError:
    LightningModule = object
    LightningLoggerBase = object
    rank_zero_only = lambda f: f
    rank_zero_experiment = lambda f: f
    has_torch = False


class ModelBuilder:
    """Scaffold of Model Builder
    """

    def __init__(self, inputs, outputs, *args, **kwargs):
        raise NotImplementedError("Abstract method.")

    def __call__(self, model_states):
        raise NotImplementedError("Abstract method.")



class GeneralChild(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BaseTorchModel(LightningModule):
    """BaseTorchModel is a subclass of pytorch_lightning.LightningModule

    It implements a basic functions of `step`, `configure_optimizers` but provides a similar
    user i/o arguments as tensorflow.keras.Model 

    A module builder will add `torch.nn.Module`s to this instance, and define its forward 
    pass function. Then this instance is responsible for training and evaluations.
    add module: use torch.nn.Module.add_module
    define forward pass: how?
    """
    def __init__(self, layers=None, data_format='NWC', *args, **kwargs):
        if has_torch is False:
            ImportError("To build TorchModel, you need to install pytorch and pytorch_lightning")
        super().__init__()
        self.__forward_pass_tracker = []
        self.layers = torch.nn.ModuleDict()
        self.hs = {}
        self.is_compiled = False
        self.criterion = None
        self.optimizer = None
        self.metrics = {}
        self.trainer = None
        self.data_format = data_format
        layers = layers or []
        for layer in layers:
            layer_id, operation, input_ids = layer[0], layer[1], layer[2] if len(layer)>2 else None
            self.add(layer_id=layer_id, operation=operation, input_ids=input_ids)
        self.save_hyperparameters()
    
    @property
    def forward_tracker(self):
        # return a read-only view
        return copy.copy(self.__forward_pass_tracker)
    
    def add(self, layer_id: str, operation, input_ids: Union[str, List, Tuple] = None):
        self.layers[layer_id] = operation
        self.__forward_pass_tracker.append((layer_id, input_ids))
    
    def compile(self, loss, optimizer, metrics=None, *args, **kwargs):
        if callable(loss):
            self.criterion = loss
        elif type(loss) is str:
            self.criterion = self._get_loss(loss)
        else:
            raise ValueError(f"unknown loss: {loss}")
        self.optimizer = optimizer
        self.metrics = metrics or {}
        self.is_compiled = True
    
    def fit(self, train_data, validation_data=None, epochs=1, callbacks=None, verbose=False):
        assert self.is_compiled, ValueError("this model instance has not been compiled yet")
        logger = InMemoryLogger()
        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=epochs,
            callbacks=callbacks,
            enable_progress_bar=verbose,
            logger=logger,
            # deterministic=True,
        )            
        trainer.fit(self, train_data, validation_data)
        return logger
    
    def evaluate(self, data, verbose=False):
        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=1,
            enable_progress_bar=verbose,
            # deterministic=True,
        )
        return trainer.test(self, data, verbose=verbose)[0]

    @staticmethod
    def _get_loss(loss_str: str):
        if loss_str == 'mse':
            return torch.nn.MSELoss()
        elif loss_str == 'binary_crossentropy':
            return torch.nn.BCELoss()
        elif loss_str == 'mae':
            return torch.nn.L1Loss()
        elif loss_str == 'nll':
            return torch.nn.NLLLoss()
        else:
            raise ValueError("cannot understand str loss: %s" % loss_str)

    def configure_optimizers(self):
        """Set up optimizers and schedulers.
        Uses Adam, learning rate from `self.lr`, and no scheduler by default.
        """
        assert self.is_compiled
        d = {}
        if isinstance(self.optimizer, (tuple, list)):
            opt = self.optimizer[0](self.parameters(), **self.optimizer[1])
            # if has scheduler? not sure if there should be better ways
            if len(self.optimizer) > 2:
                scheduler = self.optimizer[2](
                    opt, **self.optimizer[3]
                )
                d['scheduler'] = scheduler
        elif self.optimizer == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=0.001)
        elif self.optimizer == 'sgd':
            opt =  torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=5e-4)
        elif isinstance(self.optimizer, (torch.optim.Adam, torch.optim.SGD)):
            opt =  self.optimizer
        elif issubclass(self.optimizer, torch.optim.Optimizer):
            opt =  self.optimizer(self.parameters(), lr=0.001)
        else:
            raise ValueError(f"unknown torch optim {self.optimizer}")
        d = d.update({
            "optimizer": opt,
            "monitor": "val_loss",
            "frequency": 1,
        })
        return d
    
    def forward(self, x, verbose=False):
        """Scaffold forward-pass function that follows the operations in 
        the pre-set in self.__forward_pass_tracker
        """
        # permute input, if data_format has channel last
        if self.data_format == 'NWC':
            x = torch.permute(x, (0,2,1))
        # intermediate outputs, for branching models
        self.hs = {}
        # layer_id : current layer name
        # input_ids : if None,       take the output from prev layer as input
        #             if tuple/list, expect a list of layer_ids (str)
        for layer_id, input_ids in self.__forward_pass_tracker:
            assert layer_id in self.layers
            if verbose:
                print(layer_id)
                print([self.hs[layer_id].shape for layer_id in self.hs])
                print(input_ids)
            this_inputs = x if input_ids is None else self.hs[input_ids] if type(input_ids) is str else [self.hs[i] for i in input_ids]
            out = self.layers[layer_id](this_inputs)
            self.hs[layer_id] = out
            x = out
        return out
    
    def step(self, batch, kind: str) -> dict:
        """Generic step function that runs the network on a batch and outputs loss
        and accuracy information that will be aggregated at epoch end.
        This function is used to implement the training, validation, and test steps.
        """
        # run the model and calculate loss; expect a tuple from DataLoader
        if type(batch) in (tuple, list):
            batch_x = batch[0]
            y_true = batch[1]
        elif hasattr(batch, 'x') and hasattr(batch, 'y'):
            batch_x = batch.x
            y_true = batch.y
        else:
            raise TypeError("cannot decipher x and y from batch")
        y_hat = self(batch_x)
        # be explicit about observation and score
        loss = self.criterion(input=y_hat, target=y_true)
        total = len(y_true)
        batch_dict = {
            "loss": loss,
            "total": total,
        }
        for metric in self.metrics:
            batch_dict.update({
                str(metric): metric.step(input=y_hat, target=y_true)
            })
        return batch_dict

    def epoch_end(self, outputs, kind: str):
        """Generic function for summarizing and logging the loss and accuracy over an
        epoch.
        Creates log entries with name `f"{kind}_loss"` and `f"{kind}_accuracy"`.
        This function is used to implement the training, validation, and test epoch-end
        functions.
        """
        with torch.no_grad():
            # calculate average loss and average accuracy
            total_loss = sum(_["loss"] * _["total"] for _ in outputs)
            total = sum(_["total"] for _ in outputs)
            avg_loss = total_loss / total
            metrics_tot = {}
            for metric in self.metrics:
                metrics_tot[str(metric)] = metric.on_epoch_end(
                    [_[str(metric)] for _ in outputs],
                    [_["total"] for _ in outputs]
                )

        # log
        self.log(f"{kind}_loss", avg_loss)
        for metric in metrics_tot:
            self.log(f"{kind}_{metric}", metrics_tot[metric])

    def training_step(self, batch, batch_idx) -> dict:
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx) -> dict:
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx) -> dict:
        return self.step(batch, "test")

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, "test")


class ExperimentWriter:
    """In-memory experiment writer.
    Currently this supports logging hyperparameters and metrics.
    
    Borrowing from @ttesileanu https://github.com/ttesileanu/cancer-net
    """

    def __init__(self):
        self.hparams: Dict[str, Any] = {}
        self.metrics: List[Dict[str, float]] = []

    def log_hparams(self, params: Dict[str, Any]):
        """Record hyperparameters.
        This adds to previously recorded hparams, overwriting exisiting values in case
        of repeated keys.
        """
        self.hparams.update(params)

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None):
        """Record metrics."""

        def _handle_value(value: Union[torch.Tensor, Any]) -> Any:
            if isinstance(value, torch.Tensor):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics["step"] = step
        self.metrics.append(metrics)


class InMemoryLogger(LightningLoggerBase):
    """In-memory logger -- when you want to access your learning trajectory directly after learning. 
    Borrowing from @ttesileanu https://github.com/ttesileanu/cancer-net

    :param name: experiment name
    :param version: experiment version
    :param prefix: string to put at the beginning of metric keys
    :param save_dir: save directory (for checkpoints)
    :param df_aggregate_by_step: if true, the metrics dataframe is aggregated by step,
        with repeated non-NA entries averaged over
    :param hparams: access recorded hyperparameters
    :param metrics: access recorded metrics
    :param metrics_df: access metrics in Pandas dataframe format; this is only available
        after `finalize` or `save`
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        name: str = "lightning_logs",
        version: Union[int, str, None] = None,
        prefix: str = "",
        save_dir: str = "",
        df_aggregate_by_step: bool = True,
    ):
        super().__init__()
        self._name = name
        self._version = version
        self._prefix = prefix
        self._save_dir = save_dir
        self._metrics_df = None

        self.df_aggregate_by_step = df_aggregate_by_step

        self._experiment: Optional[ExperimentWriter] = None

    @property
    @rank_zero_experiment
    def experiment(self) -> ExperimentWriter:
        """Access the actual logger object."""
        if self._experiment is None:
            self._experiment = ExperimentWriter()

        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]):
        """Log hyperparameters."""
        params = _convert_params(params)
        self.experiment.log_hparams(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics, associating with given `step`."""
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        self.experiment.log_metrics(metrics, step)

    def pandas(self):
        """Return recorded metrics in a Pandas dataframe.
        By default the dataframe is aggregated by step, with repeated non-NA entries
        averaged over. This can be disabled using `self.df_aggregate_by_step`.
        """
        df = pd.DataFrame(self.experiment.metrics)
        if len(df) > 0:
            df.set_index("step", inplace=True)
            if self.df_aggregate_by_step:
                df = df.groupby("step").mean()

        return df

    def save(self):
        """Generate the metrics dataframe."""
        self._metrics_df = self.pandas()

    @property
    def name(self) -> str:
        """Experiment name."""
        return self._name

    @property
    def version(self) -> Union[int, str]:
        """Experiment version. Only used for checkpoints."""
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    def root_dir(self) -> str:
        """Parent directory for all checkpoint subdirectories.
        If the experiment name parameter is an empty string, no experiment subdirectory
        is used and the checkpoint will be saved in `save_dir/version`.
        """
        return os.path.join(self.save_dir, self.name)

    @property
    def log_dir(self) -> str:
        """The log directory for this run.
        By default, it is named `'version_${self.version}'` but it can be overridden by
        passing a string value for the constructor's version parameter instead of `None`
        or an int.
        """
        # create a pseudo standard path
        version = (
            self.version if isinstance(self.version, str) else f"version_{self.version}"
        )
        log_dir = os.path.join(self.root_dir, version)
        return log_dir

    @property
    def save_dir(self) -> str:
        """The current directory where checkpoints are saved."""
        return self._save_dir

    @property
    def hparams(self) -> Dict[str, Any]:
        """Access recorded hyperparameters.
        This is equivalent to `self.experiment.hparams`.
        """
        assert self._experiment is not None
        return self._experiment.hparams

    @property
    def metrics(self) -> List[Dict[str, float]]:
        """Access recorded metrics.
        This is equivalent to `self.experiment.metrics`.
        """
        assert self._experiment is not None
        return self._experiment.metrics

    @property
    def metrics_df(self) -> pd.DataFrame:
        """Access recorded metrics in Pandas format.
        This is a cached version of the output from `self.pandas()` generated on
        `self.save()` or `self.finalize()`.
        """
        return self._metrics_df

    def _get_next_version(self) -> int:
        root_dir = self.root_dir

        if not os.path.isdir(root_dir):
            return 0

        existing_versions = []
        for d in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1
