"""utilities include data processor, transformations
"""

try:
    import tensorflow as static_tf
    if static_tf.__version__.startswith("2"):
        print("detected tf2 - using compatibility mode")
        static_tf.compat.v1.disable_eager_execution()
        import tensorflow.compat.v1 as static_tf
except ImportError:
    static_tf = object

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']





