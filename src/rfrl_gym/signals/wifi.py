from rfrl_gym.signals.signal import Signal as Signal

# An entity that always chooses the same action.
class WiFi(Signal):
    def __init__(self, signal_label, num_channels, modulation, power, f0, t0, hop, hop_hold, spectrum_label=None):
        super().__init__(signal_label, num_channels, spectrum_label)
        self.modulation = modulation
        self.power = power
        self.f0 = f0
        self.t0 = t0
        self.hop = hop
        self.hop_hold = hop_hold
        self.period = 0.0278
        self.protocol = 'wifi'

    def _validate_self(self):
        pass

    def _get_action(self):
        pass

    def _reset(self):
        pass