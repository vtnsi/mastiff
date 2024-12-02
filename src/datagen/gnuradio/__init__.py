
def is_available():
    try:
        from gnuradio import gr
        return True
    except ImportError:
        return False

if is_available():
    from . import wifi
    from . import zigbee

    gr_signal_list = ['unknown','wifi','zigbee']
    gr_signal_map = {'unknown':None,
                     'wifi':wifi.wifi_tx,
                     'zigbee':zigbee.zigbee_tx}
