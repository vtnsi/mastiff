from abc import *
import rfrl_gym.signals
from datagen.liquid.spectrum import Spectrum
from datagen.liquid.spectrum import SignalContainer
from datagen.liquid.signal_stream import AWGNStream,SignalStream
from datagen.liquid.waterfall import waterfall


class RF_Spectrum(ABC):
    def __init__(self, spectrum_label, num_channels, signal_list=None, observation_bandwidth=None, observation_carrier=None, observation_duration=None, noise_power_dB=None, power_lower_bound=None, power_upper_bound=None, seed=None, scenario_metadata=''):
        self.spectrum_label = spectrum_label
        self.num_channels = num_channels
        self.signal_list = signal_list
        self.observation_bandwidth = observation_bandwidth
        self.observation_carrier = observation_carrier
        self.observation_duration = observation_duration
        self.noise_power_dB = noise_power_dB
        self.power_lower_bound = power_lower_bound
        self.power_upper_bound = power_upper_bound
        self.seed = seed
        self.scenario_metadata = scenario_metadata
        self.spectrum = None
        self._reset()

    def set_spectrum_index(self, spectrum_idx):
        self.spectrum_idx = spectrum_idx

    def _add_signals(self):
        self.signal_idx += 1
        for signal in self.signal_list:
            obj_str = 'rfrl_gym.signals.' + self.scenario_metadata['signals'][signal]['protocol'] + '.' + self.scenario_metadata['signals'][signal]['type'] + '(signal_label=\'' + str(signal) + '\', num_channels=' + str(self.num_channels) + ', '
            for param in self.scenario_metadata['signals'][signal]:
                if param == 'protocol':
                    pass
                elif param == 'type':
                    pass
                else:
                    obj_str += (param + '=' + str(self.scenario_metadata['signals'][signal][param]) + ', ')
            obj_str += ')'
            self.signal_object_list.append(eval(obj_str))
            self.signal_object_list[-1].set_signal_index(self.signal_idx)

    def _get_spectrum(self,power,bandwidth):
        for signal in self.signal_object_list:
            sig = SignalStream(protocol=signal.protocol)
            if bandwidth != '':
                sig.bandwidth = bandwidth
            sig.power = power
            sig.sample_rate = 100e6
            self.spectrum.add_signal(sig,signal.period,power,signal.t0)

    def _reset(self):
        self.signal_idx = 0
        self.signal_object_list = []
        if self.spectrum is not None:
            self.spectrum.close()
        self.spectrum = Spectrum(self.observation_bandwidth,self.observation_carrier,self.observation_duration,self.noise_power_dB,self.power_lower_bound,self.power_upper_bound,self.seed)
        self.spectrum._disable_fig_gen = True
        self._add_signals()
    
    def _step(self,t0=0):
        self.fig,self.matrix,self.timesteps,self.freqs,self.samples,self.bounds = self.spectrum.step()
        return self.fig, self.matrix, self.timesteps, self.freqs, self.samples, self.bounds

    def _close(self):
        self.spectrum.close()
