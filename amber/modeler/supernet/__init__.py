from ...backend import mod_name, _gen_missing_api
import sys


supernet_cls = ('EnasAnnModelBuilder', 'EnasCnnModelBuilder', 'EnasCNNwDataDescriptor')

if mod_name in ('tensorflow_1', 'tensorflow_2'):
    from . import tf1_supernet as mod
elif mod_name == 'pytorch':
    from . import pytorch_supernet as mod
else:
    raise Exception(f"Unsupported {mod_name} for supernet; must be in {list(lib_types.keys())}")

thismod = sys.modules[__name__]
for api in supernet_cls:
    if api in mod.__dict__:
        setattr(thismod, api, mod.__dict__[api])
    else:
        setattr(thismod, api, _gen_missing_api(api, mod_name))