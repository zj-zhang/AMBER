import torch
import pytorch_lightning as pl
import os
from . import cache
from .utils import InMemoryLogger
from .tensor import TensorType


class Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.is_compiled = False
        self.criterion = None
        self.optimizer = None
        self.metrics = {}
        self.trainer = None

    def compile(self, loss, optimizer, metrics=None, *args, **kwargs):
        if callable(loss):
            self.criterion = loss
        elif type(loss) is str:
            self.criterion = get_loss(loss)
        else:
            raise ValueError(f"unknown loss: {loss}")
        self.optimizer = optimizer
        self.metrics = metrics or {}
        cache.CURRENT_GRAPH.add_model(self)
        self.is_compiled = True
    
    @staticmethod
    def _make_dataloader(x, y=None, batch_size=32):
        if isinstance(x, torch.utils.data.DataLoader) and y is None:
            data = x
            return data
        if isinstance(x, (tuple, list)) and y is None:
            x, y = x[0], x[1]
        x_ = x if type(x) is TensorType else torch.tensor(x, dtype=torch.float32)
        if y is None:
            dataset = torch.utils.data.TensorDataset(x_)
        else:
            y_ = y if type(y) is TensorType else torch.tensor(y, dtype=torch.float32)
            dataset = torch.utils.data.TensorDataset(x_, y_)
        data = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        return data
    
    def fit(self, x, y=None, validation_data=None, batch_size=32, epochs=1, nsteps=None, callbacks=None, verbose=False):
        assert self.is_compiled, ValueError("this model instance has not been compiled yet")
        self.train()
        train_data = self._make_dataloader(x=x, y=y, batch_size=batch_size)
        if validation_data is not None:
            if isinstance(validation_data, (tuple, list)):
                validation_data = self._make_dataloader(x=validation_data[0], y=validation_data[1], batch_size=batch_size)
            else:
                validation_data = self._make_dataloader(x=validation_data, batch_size=batch_size)
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
    
    def predict(self, x, y=None, batch_size=32, verbose=False):
        self.eval()
        dataloader = self._make_dataloader(x=x, y=None, batch_size=batch_size)
        with torch.no_grad():
            preds = []
            for batch_x in dataloader:
                if isinstance(batch_x, (list, tuple)):
                    batch_x = batch_x[0]
                preds.append(self.forward(batch_x))
            preds = torch.concat(preds).detach().cpu().numpy()
        return preds

    def evaluate(self, x, y=None, batch_size=32, verbose=False):
        self.eval()
        data = self._make_dataloader(x=x, y=y, batch_size=batch_size)
        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=1,
            enable_progress_bar=verbose,
            # deterministic=True,
        )
        res = trainer.test(self, data, verbose=verbose)[0]
        res = {k.replace('test', 'val'):v for k,v in res.items()}
        return res

    def configure_optimizers(self):
        """Set up optimizers and schedulers.
        Uses Adam, learning rate from `self.lr`, and no scheduler by default.
        """
        assert self.is_compiled
        if isinstance(self.optimizer, (tuple, list)):
            return self.optimizer[0](self.parameters(), **self.optimizer[1])
        elif self.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=0.001)
        elif self.optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=0.01, weight_decay=5e-4)
        elif isinstance(self.optimizer, (torch.optim.Adam, torch.optim.SGD)):
            return self.optimizer
        elif issubclass(self.optimizer, torch.optim.Optimizer):
            return self.optimizer(self.parameters(), lr=0.001)
        else:
            raise ValueError(f"unknown torch optim {self.optimizer}")
        
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
        y_hat = self.forward(batch_x)
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

    def save_weights(self, *args):
        pass


class Sequential(torch.nn.Sequential):
    def __init__(self, layers=None):
        layers = layers or []
        super().__init__(*layers)
    
    def add(layer):
        super().append(layer)


def get_metric(m):
    if callable(m):
        return m
    elif m.lower() == 'kl_div':
        return torch.nn.KLDivLoss()
    else:
        raise Exception("cannot understand metric type: %s" % m)

def get_loss(loss, y_true=None, y_pred=None):
    compute_loss =  (y_true is not None) and (y_pred is not None)
    if type(loss) is str:
        loss = loss.lower()
        if loss == 'mse' or loss == 'mean_squared_error':
            loss_ = torch.nn.MSELoss()
            if compute_loss:
                loss_ = loss_(input=y_pred, target=y_true)
        elif loss == 'categorical_crossentropy':
            loss_ = torch.nn.CrossEntropyLoss(reduction='none')
        elif loss == 'binary_crossentropy':
            loss_ = torch.nn.BCELoss(reduction='none')
            if compute_loss:
               loss_ = loss_(input=y_pred, target=y_true.float())
        elif loss == 'nllloss_with_logits':
            # loss computed with NLL and LogSoftmax is equivalent to categorical_crossentropy, but more efficient for sparse classes
            loss_ = torch.nn.NLLLoss(reduction='none')
            if compute_loss:
                loss_ = loss_(input=torch.nn.LogSoftmax(dim=-1)(y_pred), target=y_true.long())
        else:
            raise Exception("cannot understand string loss: %s" % loss)
    elif type(loss) is callable:
        loss_ = loss(y_true, y_pred)
    else:
        raise TypeError("Expect loss argument to be str or callable, got %s" % type(loss))
    return loss_


def get_callback(m):
    if callable(m):
        return m
    elif m == 'EarlyStopping':
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping
        return EarlyStopping
    elif m == 'ModelCheckpoint':
        from pytorch_lightning.callbacks import ModelCheckpoint
        def ModelCheckpoint_(filename, monitor='val_loss', mode='min', save_best_only=True, verbose=False):
            return ModelCheckpoint(
                dirpath=os.path.dirname(filename), 
                filename=os.path.basename(filename),
                save_top_k=1 if save_best_only else None,
                monitor=monitor,
                mode=mode,
                verbose=verbose
                )    
        return ModelCheckpoint_


def trainable_variables(scope=None):
    scope = scope or ''
    if isinstance(scope, torch.nn.Module):
        return scope.parameters()
    elif type(scope) is str:
        return [param for name, param in cache.CURRENT_GRAPH.param_cache.items() if scope in name]


def get_optimizer(opt, parameters, opt_config=None):
    opt_config = opt_config or {'lr':0.01}
    if callable(opt):
        opt_ = opt
    elif type(opt) is str:
        if opt.lower() == 'adam':
            opt_ = torch.optim.Adam
        elif opt.lower() == 'sgd':
            opt_ = torch.optim.SGD
        else:
            raise Exception(f"unknown opt {opt}")
    else:
        raise Exception(f"unknown opt {opt}")
    return opt_(parameters,  **opt_config)


def get_train_op(loss, variables, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
