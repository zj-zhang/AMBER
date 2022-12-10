from ...backend import mod_name, _gen_missing_api
import sys

def set_mod_attr(mod, allow_append=False):
    thismod = sys.modules[__name__]
    for api in sparse_ffnn_cls:
        if api in thismod.__dict__:
            continue
        if api in mod.__dict__:
            setattr(thismod, api, mod.__dict__[api])
        else:
            if not allow_append: setattr(thismod, api, _gen_missing_api(api, mod_name))

sparse_ffnn_cls = ('SparseFfnnModelBuilder', 'MulInpSparseFfnnModelBuilder', 'MulInpAuxLossModelBuilder',
    'FeatModelSparseFfnnModelBuilder', 'CnnFeatureModel')


if mod_name in ('tensorflow_1', 'tensorflow_2'):
    from . import keras_ffnn as mod
    set_mod_attr(mod, allow_append=True)
elif mod_name == 'pytorch':
    from . import pytorch_ffnn as mod
else:
    raise Exception(f"Unsupported {mod_name} for sparse_ffnn model builder")

if mod_name == 'tensorflow_1':
    from . import tf1_featmod_ffnn as mod
    set_mod_attr(mod, allow_append=False)
