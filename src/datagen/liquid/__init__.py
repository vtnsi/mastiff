
def is_available():
    try:
        import liquid
        return True
    except ImportError:
        return False

if is_available():
    from . import ble
    lq_proto_list = ['unknown','ble']
    lq_proto_map = {'unknown':None,
                    'ble':ble.ble_tx}

    from . import waterfall
    from . import signal_stream
    from . import spectrum
    from .signal_stream import sig_stream_list,lq_mod_types
    lq_signal_list = lq_mod_types.to_list()
    del lq_mod_types
