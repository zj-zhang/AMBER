"""
Test modeler DAGs (underlying modelers)
"""

import sys
import numpy as np
from tqdm import tqdm
from amber.utils import testing_utils
from amber import modeler
from amber import architect
from amber import backend
from amber import backend as F
from amber.modeler.supernet import EnasCnnModelBuilder
import logging, sys
logging.disable(sys.maxsize)
import unittest
from parameterized import parameterized
try:
    import torch
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    torch = object
try:
    from torchviz import make_dot
    has_torchviz = True
except ImportError:
    has_torchviz = False


@unittest.skipIf(backend.mod_name!="pytorch", reason="skipped because non-pytorch backend")
class TestEnasPyTorchConvDAG(testing_utils.TestCase):
    def setUp(self):
        self.session = F.Session()

        input_op = architect.Operation('input', shape=(100, 4), name="input")
        output_op = architect.Operation('dense', units=1, activation='sigmoid', name="output")
        model_space, _ = testing_utils.get_example_conv1d_space(num_layers=12, num_pool=4)
        #self.controller = architect.GeneralController(model_space=model_space, buffer_type='ordinal', with_skip_connection=True, session=self.session)
        self.decoder = modeler.architectureDecoder.ResConvNetArchitecture(model_space=model_space)
        self.dag = EnasCnnModelBuilder(model_space=model_space, input_node=input_op,  output_node=output_op, model_compile_dict={}, reduction_factor=2)

    def test_1forward(self):
        x = torch.randn((10, 100, 4))
        self.dag.eval()
        for _ in range(10):
            #arc, p = self.controller.get_action()
            arc = self.decoder.sample()
            y_pred = self.dag(arc, x)
            self.assertTrue(y_pred.shape == (10,1))
    
    def test_2backward(self):
        x = torch.randn((10, 100, 4))
        y = torch.ones((10, 1))
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.dag.parameters(), lr=0.001)
        #arc, p = self.controller.get_action()
        arc = self.decoder.sample()
        losses = []
        self.dag.train()
        for _ in tqdm(range(3)):
            optimizer.zero_grad()
            y_pred = self.dag(arc, x)
            self.assertTrue(y_pred.shape == (10,1))
            loss = loss_fn(y_pred, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], losses[0])    


@unittest.skipIf(backend.mod_name!="pytorch", reason="skipped because non-pytorch backend")
class TestPyTorchResConvModelBuilder(unittest.TestCase):
    def setUp(self):
        input_op = architect.Operation('input', shape=(1000, 4), name="input")
        output_op = architect.Operation('dense', units=1, activation='sigmoid', name="output")
        model_space, _ = testing_utils.get_example_conv1d_space(num_layers=3, num_pool=3)
        #self.controller = architect.GeneralController(model_space=model_space, with_skip_connection=True, session=self.session)
        self.arc = [0, 1, 1, 2, 1, 1]
        self.mb = modeler.pytorchModeler.PytorchResidualCnnBuilder(
            input_op, output_op, fc_units=32, flatten_mode='flatten', 
            model_compile_dict={'loss':'mse', 'optimizer':'adam'}, 
            model_space=model_space,
            verbose=False
        )

    def test_1forward(self):
        child = self.mb(self.arc)
        child.eval()
        pred = child.forward(torch.randn(3,1000,4), verbose=False)
        pred = pred.detach().cpu().numpy()
        self.assertTrue(pred.shape == (3,1))
    
    @parameterized.expand([
        ('binary_crossentropy', 'adam'),
        ('mse', 'adam'),
        ('mae', 'adam'),
        ('mse', 'sgd'),
        #('mse', torch.optim.Adam)
    ])
    def test_2backward(self, loss, optimizer):
        x = torch.randn((50, 1000, 4))
        y = torch.randint(low=0, high=1, size=(50, 1), dtype=torch.float)
        train_data = DataLoader(TensorDataset(x, y), batch_size=10)
        arc = self.arc
        model = self.mb(arc)
        model.compile(loss=loss, optimizer=optimizer)
        old_loss = model.evaluate(train_data)
        model.fit(train_data, epochs=10)
        new_loss = model.evaluate(train_data)
        self.assertLess(new_loss['test_loss'], old_loss['test_loss'])
        
    def test_3viz(self):
        if has_torchviz:
            arc = self.arc
            child = self.mb(arc)
            child.eval()
            pred = child.forward(torch.randn(3,1000,4), verbose=False)
            dot = make_dot(pred.mean(), params=dict(child.named_parameters()))
        else:
            dot = None
        return dot


if __name__ == '__main__':
    unittest.main()



