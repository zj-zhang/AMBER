from ...backend import mod_name, _gen_missing_api
import sys

lib_types = {
    'tensorflow_1': 'static',
    'pytorch': 'dynamic',
    'tensorflow_2': 'dynamic'
}

supernet_cls = ('EnasAnnDAG', 'EnasConv1dDAG', 'EnasConv1DwDataDescrption')

if lib_types[mod_name] == 'static':
    from . import static as mod
elif lib_types[mod_name] == 'dynamic':
    from . import dynamic as mod
else:
    raise Exception(f"Unsupported {mod_name} for supernet; must be in {list(lib_types.keys())}")

thismod = sys.modules[__name__]
for api in supernet_cls:
    if api in mod.__dict__:
        setattr(thismod, api, mod.__dict__[api])
    else:
        setattr(thismod, api, _gen_missing_api(api, mod_name))