
import numpy as np
from typing import List,Union,Dict
from rfrl_gym.entities.entity import Entity
from datagen.liquid.signal_stream import SignalStream

''' Signal - naive signal

    Args:
    entity_label: unique name for proper entity selection
    protocol: The string name of the protocol to use, or None {'wifi','zigbee','ble'}
    modulation: The string name of an implemented modulation. See 'datagen.liquid.signal_stream.lq_mod_types'.
    signal_meta: a dictionary of the parameters to define how the signal should be configured
        The assumption is that the following will work:

            self.gen = SignalStream(modulation=modulation,protocol=protocol)
            for k,v in signal_meta.items():
                setattr(self or self.gen,k,v)

        will run without error
'''
_max_steps_ = 9223372036854775807

class SignalAction(object):
    def __init__(self,time,carrier,gain,data,action_index):
        self.time = time
        self.carrier = carrier
        self.gain = gain
        self.data = data
        self.action_index = action_index

class Signal(Entity):
    def __init__(self, entity_label:str, protocol:Union[str,None]=None, modulation:Union[str,None]=None, signal_meta:Dict={}):
        self.seed = signal_meta['seed'] if 'seed' in signal_meta else None
        self._rng = np.random.default_rng(self.seed)

        self.gen = SignalStream(modulation=modulation if modulation is not None else -1,protocol=protocol)
        self._channel_list = None
        self._channel = None
        self._period = None
        self._next_time = None
        self._start_time = None
        self._gain = None
        self._action_idx = 0
        self.gen.rng = self._rng
        super().__init__(entity_label,0,[],[1,0,0],0,_max_steps_,None)
        for k,v in signal_meta.items():
            if hasattr(self,k):
                setattr(self,k,v)
            else:
                if hasattr(self.gen,k):
                    setattr(self.gen,k,v)
                else:
                    try:
                        setattr(self.gen,k,v)
                    except:
                        print('key({}) failed, skipping'.format(repr(k)))

    @property
    def modulation(self):
        return self.gen._mod
    @property
    def protocol(self):
        return self.gen._prot
    @property
    def channel_list(self):
        if self._channel_list is not None:
            return self._channel_list
        if self.gen._prot_gen is not None and hasattr(self.gen._prot_gen,'channel_list'):
            self._channel_list = self.gen._prot_gen.channels_carrier
            return self._channel_list
        return None
    @channel_list.setter
    def channel_list(self,channel_list:List[Union[int,float]]):
        self._channel_list = channel_list
    @property
    def channel_index(self):
        if self._channel is not None:
            return self._channel
        if self.gen._prot_gen is not None and hasattr(self.gen._prot_gen,'channel'):
            self._channel = self.gen._prot_gen.channel
            return self._channel
        return None
    @channel_index.setter
    def channel_index(self,channel_index:int):
        if self._channel is not None and self._channel_list is not None:
            return self._channel
        if self.channel_list is None or channel_index < 0 or channel_index >= len(self.channel_list):
            raise ValueError("Cannot set channel because {}".format("channel_list isn't set" if self.channel_list is None
                             else "channel value is invalid {} [0,{})".format(channel_index,len(self.channel_list))))
        self._channel = channel_index
    @property
    def period(self):
        if self._period is not None:
            return self._period
        if self.gen._prot_gen is not None and hasattr(self.gen._prot_gen,'period'):
            self._period = self.gen._prot_gen.period
            return self._period
        return None
    @period.setter
    def period(self,period:float):
        self._period = period
    @property
    def start_time(self):
        if self._start_time is not None:
            return self._start_time
        if self.gen._prot_gen is not None and hasattr(self.gen._prot_gen,'start_time'):
            self._start_time = self.gen._prot_gen.start_time
            return self._start_time
        return None
    @start_time.setter
    def start_time(self,start_time:float):
        self._start_time = start_time
    @property
    def gain(self):
        if self._gain is not None:
            return self._gain
        if self.gen._prot_gen is not None and hasattr(self.gen._prot_gen,'gain'):
            self._gain = self.gen._prot_gen.gain
            return self._gain
        return None
    @gain.setter
    def gain(self,gain:float):
        self._gain = gain

    @staticmethod
    def from_stream(stream:SignalStream):
        ### make it easier to pass them back and forth
        pass

    def to_stream(self):
        return self.gen

    def _validate_self(self):
        assert(self.gen != None)
        assert(self.gen._made == True)
        assert(self.gen._mod_gen is not None or self.gen._prot_gen is not None)

    def _get_action(self):
        ### FIXME: figure out what to return if not quite clear
        if self._next_time is None and self._start_time is not None:
            self._next_time = self._start_time
        fc,data = self.gen.step()
        action = SignalAction(self._next_time,fc,self.gain,data,self._action_idx)
        self._next_time += self._period
        self._action_idx += 1
        return action

    def _reset(self):
        self.gen.reset()
