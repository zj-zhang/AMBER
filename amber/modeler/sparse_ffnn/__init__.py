from ...backend import mod_name, _gen_missing_api
import sys


multiio_cls = ('SparseFfnnModelBuilder', 'MulInpSparseFfnnModelBuilder', 'MulInpAuxLossModelBuilder')


if mod_name in ('tensorflow_1', 'tensorflow_2'):
    from . import keras_ffnn as mod
elif mod_name == 'pytorch':
    from . import pytorch_ffnn as mod
else:
    raise Exception(f"Unsupported {mod_name} for resnet model builder")

thismod = sys.modules[__name__]
for api in multiio_cls:
    if api in mod.__dict__:
        setattr(thismod, api, mod.__dict__[api])
    else:
        setattr(thismod, api, _gen_missing_api(api, mod_name))