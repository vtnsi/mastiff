from rfrl_gym.spectrums.spectrum import RF_Spectrum

class Signal(RF_Spectrum):
    def __init__(self, signal_label, num_channels, spectrum_label=None):
        super().__init__(spectrum_label, num_channels)
        self.signal_label = signal_label

    # Set the index of the signal.
    def set_signal_index(self, signal_idx):
        self.signal_idx = signal_idx

    def _validate_self(self):
        pass

    def _get_action(self):
        pass

    def _reset(self):
        pass
