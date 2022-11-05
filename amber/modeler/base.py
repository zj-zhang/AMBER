"""The base class for modelers live here
"""
from typing import Tuple, List, Union
from collections import OrderedDict
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
    has_torch = True
except ImportError:
    LightningModule = object
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
    def __init__(self, *args, **kwargs):
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
        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=epochs,
            callbacks=callbacks,
            enable_progress_bar=verbose,
            # deterministic=True,
        )            
        trainer.fit(self, train_data, validation_data)
        return self
    
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
        if self.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=0.001)
        elif self.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=5e-4)
        elif isinstance(self.optimizer, (torch.optim.Adam, torch.optim.SGD)):
            return self.optimizer    
        elif issubclass(self.optimizer, torch.optim.Optimizer):
            return self.optimizer(self.parameters(), lr=0.001)
        else:
            raise ValueError(f"unknown torch optim {self.optimizer}")
    
    def forward(self, x, verbose=False):
        """Scaffold forward-pass function that follows the operations in 
        the pre-set in self.__forward_pass_tracker
        """
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