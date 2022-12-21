from amber import getter, architect, modeler
from amber import backend as F
from amber.utils import testing_utils
import unittest, pytest, tempfile
import numpy as np, h5py, pickle

def get_data(ms_type, seed=777):
    if ms_type == 'cnn':
        x = np.random.randn(50, 100, 4)
    elif ms_type == 'dnn':
        x = np.random.randn(50, 5)
    else: raise Exception('unknown ms_type')
    y = np.random.randn(50, 1)
    return x,y

def _test_model_api(model, x, y):
    hist = model.fit(x,y, epochs=2, batch_size=10)
    loss = model.evaluate(x,y)
    pred = model.predict(x)

def test_sequential_cnn():
    model_space, _ = testing_utils.get_example_conv1d_space(num_layers=4, num_pool=2)
    model_space.add_layer(4, [F.Operation('Flatten'), F.Operation('GlobalAveragePooling1D')])
    inputs_op = [F.Operation('input', shape=(100,4))]
    outputs_op = [F.Operation('dense', units=1)]
    model_compile_dict = {'loss':'mse', 'optimizer':'adam', 'metrics':['mae']}
    model_fn = modeler.sequential.SequentialModelBuilder(inputs_op=inputs_op, output_op=outputs_op, model_space=model_space, model_compile_dict=model_compile_dict)
    x, y = get_data(ms_type='cnn')
    for _ in range(10):
        arc = [np.random.randint(len(model_space[i])) for i in range(len(model_space))]
        model = model_fn(arc)
        assert callable(model)
        _test_model_api(model, x, y)


def test_sequential_dnn():
    model_space, _ = testing_utils.get_example_sparse_model_space(num_layers=5)
    inputs_op = [F.Operation('input', shape=(5,))]
    outputs_op = [F.Operation('dense', units=1)]
    model_compile_dict = {'loss':'mse', 'optimizer':'adam', 'metrics':['mae']}
    model_fn = modeler.sequential.SequentialModelBuilder(inputs_op=inputs_op, output_op=outputs_op, model_space=model_space, model_compile_dict=model_compile_dict)
    x, y =get_data(ms_type='dnn')
    for _ in range(10):
        arc = [np.random.randint(len(model_space[i])) for i in range(len(model_space))]
        model = model_fn(arc)
        assert callable(model)
        _test_model_api(model, x, y)

if __name__ == '__main__':
    test_sequential_dnn()