
import numpy as np
from typing import List,Union,Dict
from rfrl_gym.entities.entity import Entity
from datagen.liquid.signal_stream import SignalStream
from datagen.liquid.spectrum import Spectrum
try:
    from .signal_streamer import Signal
except ImportError:
    from rfrl_gym.entities.signal_streamer import Signal

''' Background - the naive signals that don't react

    Args:
    entity_label: unique name for proper entity selection
    signal_list: a list of all signals 'SignalStream' objects to be controlled
    observations_meta: a dictionary of the parameters to define how the background should function
        - sample_rate: The sample rate to observe
        - instantaneous_bandwidth: How much bandwidth is actually usable (assume sample_rate for now)
        - observation_bandwidth: define how much the background actually encompasses, in case of scanning
        - scan_mode: {"stare","sweep","other"} (use stare for now) -- how the background should update
                per step, hold|stare, change|sweep, other|TBD.
'''
_max_steps_ = 9223372036854775807

class Background(Entity):
    def __init__(self, entity_label:str, signal_list:List[Union[SignalStream,Signal]], observation_meta:Dict):
        self.sample_rate = observation_meta['sample_rate']
        self.instant_bw = observation_meta['instantaneous_bandwidth']
        self.observed_bw = observation_meta['observation_bandwidth']
        self.scan_mode = observation_meta['scan_mode']
        self.step_len = observation_meta['time_step']
        self.n_time_segments = len(observation_meta['time_segments'])
        self.time_segments = observation_meta['time_segments']
        self.n_freq_segments = len(observation_meta['freq_segments'])
        self.freq_segments = observation_meta['freq_segments']
        self.stop_at = (observation_meta['max_steps'] if observation_meta['max_steps'] is not None else _max_steps_) if 'max_steps' in observation_meta else _max_steps_
        self.seed = observation_meta['seed'] if 'seed' in observation_meta else None
        self._rng = np.random.default_rng(self.seed)
        super().__init__(entity_label,self.n_freq_segments,self.freq_segments,[1,0,0],0,self.stop_at)

        self.gen = Spectrum(self.sample_rate,self.freq_segments[0],self.step_len,-100,seed=self._rng.integers(0,_max_steps_,(10,)))
        for sig in signal_list:
            sig.sample_rate = self.sample_rate
            if sig.protocol in ['']:
                self.gen.add_signal(sig,self._rng.uniform(5e-5))

    def _validate_self(self):
        pass

    def _reset(self):
        pass

    def _get_action(self):
        return -1
