
from typing import Union,List,Tuple,Iterable,Dict
import liquid
import numpy as np

try:
    from ..gnuradio import gr_signal_list,gr_signal_map
except ImportError:
    from datagen.gnuradio import gr_signal_list,gr_signal_map

class AWGNStream(object):
    def __init__(self, noise_power_dB:float=-100, rng:np.random.Generator=None,bw=1.0,rate=1.0):
        self._np = noise_power_dB
        self._rng = rng
        self._rate = rate
        self._bw = bw
        self._build_gen()

    def _build_gen(self):
        if self._rng is None:
            self._gen = lambda *x : np.random.randn(*x).astype(np.single)
        else:
            self._gen = lambda *x : self._rng.normal(scale=np.sqrt(0.5),size=x).astype(np.single)
        if self._bw != self._rate:
            self.resamp = liquid.msresamp(self._rate/self._bw)
        else:
            self.resamp = None

    @property
    def delay(self):
        if self.resamp is None:
            return 0
        return self.resamp.delay

    @property
    def bits_per_symbol(self):
        return 1

    @property
    def scheme(self):
        return 'awgn'

    def reset(self):
        if self.resamp is not None:
            self.resamp.reset()

    def demodulate(self,arg0:complex):
        return np.angle(arg0)

    def modulate(self,arg0:Union[None,int,float,List]):
        if arg0 is None:
            return np.power(10,self._np/20)*np.squeeze(self._gen(1,2).view(np.csingle))
        elif isinstance(arg0,(int,float)):
            return np.power(10,self._np/20)*np.squeeze(self._gen(int(arg0),2).view(np.csingle))
        else:
            return np.array([self.modulate(x) for x in arg0],dtype=np.csingle)

    def generate(self,arg0:int=256):
        stream = self.modulate(arg0)
        if self.resamp is not None:
            stream = self.resamp.execute(stream)
        return stream

    def burst(self,arg0:int=256):
        stream = self.modulate(int(arg0-self.delay))
        stream = np.concatenate([stream,np.zeros((arg0-int(arg0-self.delay),),np.csingle)],axis=0)
        if self.resamp is not None:
            stream = self.resamp.execute(stream)
        return stream

class SignalStream(object):
    def __init__(self,*,modulation:Union[str,None]=-1,protocol:Union[str,None]=None):
        ''' SignalStream - produce the waveform samples

        args:
            - modulation: the requested modulation, if defined overrides protocol
            - protocol: the requested protocol -- set characteristics

        attr:
            - modulation: the current modulation -- protocol default if unknown
            - protocol: the current protocol -- unstructured if protocol is unknown
            - sample_rate: the assumed output sample rate of the generated samples (Hz)
            - duration: the expected time duration generated at each step (s)
            - bandwidth: the target bandwidth for the waveform to occupy at sample_rate
                    if using a protocol, this is ignored by default (Hz)
        '''
        self._verbose = 0
        self._rate = -1
        self._bw = -1
        self._fc = None
        self._dur = -1
        self._mod_gen = None
        self._prot_gen = None
        self._update_bw = False
        self._update_fc = False
        self._update_dur = False
        self._update_mod = False
        self._update_prot = False
        self._update_rate = False
        self.inflate_bbox = 0
        if modulation is not None:
            setattr(self,'modulation',modulation if modulation != -1 else None)
        setattr(self,'protocol',protocol)
        setattr(self,'sample_rate',100e6)
        setattr(self,'duration',0.05)
        if self._prot is None:
            setattr(self,'bandwidth',4.1e6)

        self._made = False
        self._make()
        # print(f"---M---{self._mod_gen}")
        # print(f"---P---{self._prot_gen}")

    def __str__(self):
        out = f"SignalStream(mod:{self._mod}, protocol:{self._prot})"
        return out

    def __repr__(self):
        return str(self)

    def start(self):
        if self.need_update():
            self.reset()
        if hasattr(self._mod_gen,'start'):
            self._mod_gen.start()
        if hasattr(self._prot_gen,'start'):
            self._prot_gen.start()

    def stop(self):
        if hasattr(self._mod_gen,'stop'):
            self._mod_gen.stop()
        if hasattr(self._prot_gen,'stop'):
            self._prot_gen.stop()

    def wait(self):
        if hasattr(self._mod_gen,'wait'):
            self._mod_gen.wait()
        if hasattr(self._prot_gen,'wait'):
            self._prot_gen.wait()

    @property
    def modulation(self):
        return self._mod

    @modulation.setter
    def modulation(self,modulation:str):
        if modulation is None:
            setattr(self,'modulation','unknown')
        else:
            if not hasattr(self,'_mod') or modulation.lower() != self._mod:
                if modulation.lower() in sig_stream_list:
                    self._mod = modulation
                else:
                    self._mod = 'unknown'
                self._update_mod = True
            elif self._verbose:
                print("modulation - No action taken")

    @property
    def protocol(self):
        return self._prot

    @protocol.setter
    def protocol(self,protocol:str):
        try:
            from ..liquid import lq_proto_map
        except ImportError:
            from datagen.liquid import lq_proto_map
        if protocol is None:
            setattr(self,'protocol','unknown')
        else:
            if not hasattr(self,'_prot') or protocol.lower() != self._prot:
                if protocol.lower() in gr_signal_map or protocol.lower() in lq_proto_map:
                    self._prot = protocol
                else:
                    self._prot = 'unknown'
                self._update_prot = True
            elif self._verbose:
                print("protocol - No action taken")

    def bbox_adjust(self):
        if self.inflate_bbox:
            return (0,2*self.inflate_bbox[0],-self.inflate_bbox[1],self.inflate_bbox[1])
        return (0,0,0,0)

    @property
    def sample_rate(self):
        return self._rate

    @sample_rate.setter
    def sample_rate(self,sample_rate):
        #### check against a minimum,maximum before setting
        if sample_rate != self._rate:
            self._rate=sample_rate
            self._update_rate = True
        elif self._verbose:
            print("sample_rate - no action taken")

    @property
    def duration(self):
        if self._prot_gen is not None:
            return self._prot_gen.burst_time()
        return self._dur

    @duration.setter
    def duration(self,duration):
        if self._prot_gen is None:
            #### check against a minimum,maximum before setting
            if duration != self._dur:
                self._dur = duration
                self._update_dur = True
            elif self._verbose:
                print("duration - no action taken")
        elif self._verbose:
            print("protocol is set, can't affect duration directly")

    @property
    def bandwidth(self):
        return self._bw

    @bandwidth.setter
    def bandwidth(self,bandwidth):
        #### check against a minimum,maximum before setting
        if bandwidth != self._bw:
            self._bw = bandwidth
            self._update_bw = True
        elif self._verbose:
            print("bandwidth - no action taken")

    @property
    def carrier(self):
        if self._prot_gen is not None:
            return self._prot_gen.frequency
        return self._fc

    @carrier.setter
    def carrier(self,carrier):
        #### check against a minimum,maximum before setting
        if carrier != self._fc:
            if self._prot_gen is not None:
                try:
                    self._prot_gen.frequency = carrier
                except:
                    closest = np.argmin(np.abs(carrier-np.array(self._prot_gen.channels_carrier)))
                    self._prot_gen.frequency = closest
            else:
                self._fc = carrier
            self._update_fc = True
        elif self._verbose:
            print("carrier - no action taken")

    @property
    def _gen_sample_rate(self):
        if hasattr(self,"_gen_sample_rate_override") and self._gen_sample_rate_override is not None:
            return self._gen_sample_rate_override
        if self._prot_gen is not None:
            return self._prot_gen.sample_rate
        return self.sample_rate
    @property
    def _gen_bandwidth(self):
        if hasattr(self,"_gen_bandwidth_override") and self._gen_bandwidth_override is not None:
            return self._gen_bandwidth_override
        if self._prot_gen is not None:
            return self._prot_gen.bw
        return self.bandwidth

    def _make(self):
        if not self._made:
            if (self._prot is not None and self._prot != "unknown") and (self._mod is not None and self._mod != "unknown"):
                self._build_prot()
                setattr(self,'sample_rate',self._gen_sample_rate)
                setattr(self,'bandwidth',self._gen_bandwidth)
                self._nsamps = self._prot_gen.burst_length()
                self._build_mod()
            elif (self._prot is not None and self._prot != "unknown"):
                self._build_prot()
                setattr(self,'sample_rate',self._gen_sample_rate)
                setattr(self,'bandwidth',self._gen_bandwidth)
                self._nsamps = self._prot_gen.burst_length()
            else:
                self._build_mod()
                self._nsamps = int(self._dur*self._rate)
            self._rr = 1.0
            self._fbw = self._bw/self._rate
            self._resamp = None
            self._made = True

    def _build_mod(self):
        if self._mod != 'unknown':
            if self._mod in lq_mod_types.to_list():
                print(self._mod,self._bw,self._rate,self._gen_bandwidth,self._gen_sample_rate)
                self._mod_gen = liquid.symstreamr(ms=self._mod,bw=self._bw/self._rate,ftype='rrcos',m=5,beta=0.2,gain=1.0)
            else:#elif self._mod in sig_stream_list:
                self._mod_gen = AWGNStream(bw=self._bw,rate=self._rate)
        else:
            self._mod_gen = None

    def _build_prot(self):
        try:
            from ..liquid import lq_proto_map
        except ImportError:
            from datagen.liquid import lq_proto_map
        if not hasattr(self,'_prot_gen'):
            self._prot_gen = None
        if self._prot_gen is not None:
            if hasattr(self._prot_gen,'stop'):
                self._prot_gen.stop()
                self._prot_gen.wait()
            self._prot_gen = None
        if self._prot != 'unknown':
            if self._prot.lower() in gr_signal_list:
                self._prot_gen = gr_signal_map[self._prot.lower()]()
            elif self._prot.lower() in lq_proto_map:
                self._prot_gen = lq_proto_map[self._prot.lower()]()
            self._prot_gen.randomize()
            if hasattr(self._prot_gen,'start'):
                self._prot_gen.start()
        else:
            self._prot_gen = None

    def reset_filters(self):
        if self._resamp is not None:
            self._resamp.reset()
        if self._mod_gen is not None:
            self._mod_gen.reset()
        if self._prot_gen is not None and hasattr(self._prot_gen,'reset'):
            self._prot_gen.reset()

    def reset(self):
        self.stop()
        self.wait()
        late_mod = False
        if self._update_prot:
            self._build_prot()
            if self._prot_gen is not None:
                if not self._update_bw:
                    setattr(self,'bandwidth',self._gen_bandwidth)
                if not self._update_rate:
                    setattr(self,'sample_rate',self._gen_sample_rate)
            late_mod = True
        elif self._update_mod:
            self._build_mod()
        if self._update_bw or self._update_rate:
            self._rr = 1.0
            if self._prot_gen is not None:
                gen_bw = self._gen_bandwidth
                gen_rate = self._gen_sample_rate
                gen_ratio = gen_bw/gen_rate
                target_bw = self._bw
                target_rate = self._rate
                target_ratio = target_bw/target_rate
                if target_ratio > 1.0:
                    raise RuntimeError("Bandwidth exceeds samples rate")
                self._rr = gen_ratio/target_ratio
                if self._rr != 1.0:
                    self._resamp = liquid.msresamp(self._rr,60.)
            self._fbw = self._bw/self._rate
        if self._update_dur or self._update_rate or self._update_prot:
            if self._prot_gen is None:
                self._nsamps = int(self._dur*self._rate)
            else:
                self._nsamps = self._prot_gen.burst_length()
        self._update_bw = False
        self._update_dur = False
        self._update_prot = False
        self._update_rate = False
        if late_mod:
            self._build_mod()
        self._update_mod = False
        if self._resamp is not None:
            self._resamp.reset()

    def need_update(self):
        return any([self._update_bw, self._update_dur, self._update_mod, self._update_prot, self._update_rate])

    def get_protocol_payload_bounds(self):
        if self._prot_gen is not None:
            return [x for x in self._prot_gen.pdu_range]

    def set_protocol_payload_length(self,length:int):
        if self._prot_gen is not None:
            bounds = self._prot_gen.pdu_range
            if length >= bounds[0] and length < bounds[1]:
                self._prot_gen.pdu_length = length
            else:
                raise ValueError(f"Trying to set a payload length of {length} when the bounds are [{bounds[0]},{bounds[1]})")

    def set_protocol(self,**kwargs):
        for k,v in kwargs.items():
            if hasattr(self._prot_gen,'_'.join(['set',k])):
                getattr(self._prot_gen,'_'.join(['set',k]))(v)
            elif hasattr(self._prot_gen,k):
                setattr(self._prot_gen,k,v)

    @property
    def _gfbw(self):
        return self._gen_bandwidth/self._gen_sample_rate

    def step(self)-> Tuple[float,np.ndarray]:
        if self.need_update():
            self.reset()
        # symbol_delay = self._mod_gen.delay if self._mod_gen is not None else 0
        # total_delay = int(symbol_delay/self._rr + sample_delay)
        if self._mod_gen is not None and self._prot_gen is not None:
            nsamps = self._prot_gen.burst_length()
            samples = self._mod_gen.burst(nsamps)
            fc = self._prot_gen.frequency
        elif self._prot_gen is not None:
            samples = self._prot_gen.step()
            nsamps = len(samples)
            fc = self._prot_gen.frequency
        elif self._mod_gen is not None:
            nsamps = self._nsamps
            samples = self._mod_gen.burst(nsamps)
            fc = self._fc
        else:
            raise RuntimeError("No modulation nor protocol defined")
        sample_delay = self._resamp.delay if self._resamp is not None else 0
        delay = int(sample_delay*self._rr)
        rough_est = int(nsamps*self._rr)
        if self._resamp is not None:
            samples = self._resamp.execute(np.concatenate([samples,np.zeros((delay,),dtype=np.csingle)]))
        samples = samples[delay//2:delay//2+rough_est].copy()
        self.reset_filters()
        return fc,samples

    def close(self):
        if hasattr(self,'_prot_gen'):
            if self._prot_gen is not None:
                if hasattr(self._prot_gen,'stop'):
                    self._prot_gen.stop()
                    self._prot_gen.wait()

class lq_mod_types(enumerate):
    LIQUID_MODEM_UNKNOWN =     "UNKNOWN".lower()
    LIQUID_MODEM_PSK2 =        "PSK2".lower()
    LIQUID_MODEM_PSK4 =        "PSK4".lower()
    LIQUID_MODEM_PSK8 =        "PSK8".lower()
    LIQUID_MODEM_PSK16 =       "PSK16".lower()
    LIQUID_MODEM_PSK32 =       "PSK32".lower()
    LIQUID_MODEM_PSK64 =       "PSK64".lower()
    LIQUID_MODEM_PSK128 =      "PSK128".lower()
    LIQUID_MODEM_PSK256 =      "PSK256".lower()
    LIQUID_MODEM_DPSK2 =       "DPSK2".lower()
    LIQUID_MODEM_DPSK4 =       "DPSK4".lower()
    LIQUID_MODEM_DPSK8 =       "DPSK8".lower()
    LIQUID_MODEM_DPSK16 =      "DPSK16".lower()
    LIQUID_MODEM_DPSK32 =      "DPSK32".lower()
    LIQUID_MODEM_DPSK64 =      "DPSK64".lower()
    LIQUID_MODEM_DPSK128 =     "DPSK128".lower()
    LIQUID_MODEM_DPSK256 =     "DPSK256".lower()
    LIQUID_MODEM_ASK2 =        "ASK2".lower()
    LIQUID_MODEM_ASK4 =        "ASK4".lower()
    LIQUID_MODEM_ASK8 =        "ASK8".lower()
    LIQUID_MODEM_ASK16 =       "ASK16".lower()
    LIQUID_MODEM_ASK32 =       "ASK32".lower()
    LIQUID_MODEM_ASK64 =       "ASK64".lower()
    LIQUID_MODEM_ASK128 =      "ASK128".lower()
    LIQUID_MODEM_ASK256 =      "ASK256".lower()
    LIQUID_MODEM_QAM4 =        "QAM4".lower()
    LIQUID_MODEM_QAM8 =        "QAM8".lower()
    LIQUID_MODEM_QAM16 =       "QAM16".lower()
    LIQUID_MODEM_QAM32 =       "QAM32".lower()
    LIQUID_MODEM_QAM64 =       "QAM64".lower()
    LIQUID_MODEM_QAM128 =      "QAM128".lower()
    LIQUID_MODEM_QAM256 =      "QAM256".lower()
    LIQUID_MODEM_APSK4 =       "APSK4".lower()
    LIQUID_MODEM_APSK8 =       "APSK8".lower()
    LIQUID_MODEM_APSK16 =      "APSK16".lower()
    LIQUID_MODEM_APSK32 =      "APSK32".lower()
    LIQUID_MODEM_APSK64 =      "APSK64".lower()
    LIQUID_MODEM_APSK128 =     "APSK128".lower()
    LIQUID_MODEM_APSK256 =     "APSK256".lower()
    LIQUID_MODEM_BPSK =        "BPSK".lower()
    LIQUID_MODEM_QPSK =        "QPSK".lower()
    LIQUID_MODEM_OOK =         "OOK".lower()
    LIQUID_MODEM_SQAM32 =      "SQAM32".lower()
    LIQUID_MODEM_SQAM128 =     "SQAM128".lower()
    LIQUID_MODEM_V29 =         "V29".lower()
    LIQUID_MODEM_ARB16OPT =    "ARB16OPT".lower()
    LIQUID_MODEM_ARB32OPT =    "ARB32OPT".lower()
    LIQUID_MODEM_ARB64OPT =    "ARB64OPT".lower()
    LIQUID_MODEM_ARB128OPT =   "ARB128OPT".lower()
    LIQUID_MODEM_ARB256OPT =   "ARB256OPT".lower()
    LIQUID_MODEM_ARB64VT =     "ARB64VT".lower()
    LIQUID_MODEM_PI4DQPSK =    "PI4DQPSK".lower()

    @staticmethod
    def to_list():
        return [
            lq_mod_types.LIQUID_MODEM_UNKNOWN,
            lq_mod_types.LIQUID_MODEM_PSK2,
            lq_mod_types.LIQUID_MODEM_PSK4,
            lq_mod_types.LIQUID_MODEM_PSK8,
            lq_mod_types.LIQUID_MODEM_PSK16,
            lq_mod_types.LIQUID_MODEM_PSK32,
            lq_mod_types.LIQUID_MODEM_PSK64,
            lq_mod_types.LIQUID_MODEM_PSK128,
            lq_mod_types.LIQUID_MODEM_PSK256,
            lq_mod_types.LIQUID_MODEM_DPSK2,
            lq_mod_types.LIQUID_MODEM_DPSK4,
            lq_mod_types.LIQUID_MODEM_DPSK8,
            lq_mod_types.LIQUID_MODEM_DPSK16,
            lq_mod_types.LIQUID_MODEM_DPSK32,
            lq_mod_types.LIQUID_MODEM_DPSK64,
            lq_mod_types.LIQUID_MODEM_DPSK128,
            lq_mod_types.LIQUID_MODEM_DPSK256,
            lq_mod_types.LIQUID_MODEM_ASK2,
            lq_mod_types.LIQUID_MODEM_ASK4,
            lq_mod_types.LIQUID_MODEM_ASK8,
            lq_mod_types.LIQUID_MODEM_ASK16,
            lq_mod_types.LIQUID_MODEM_ASK32,
            lq_mod_types.LIQUID_MODEM_ASK64,
            lq_mod_types.LIQUID_MODEM_ASK128,
            lq_mod_types.LIQUID_MODEM_ASK256,
            lq_mod_types.LIQUID_MODEM_QAM4,
            lq_mod_types.LIQUID_MODEM_QAM8,
            lq_mod_types.LIQUID_MODEM_QAM16,
            lq_mod_types.LIQUID_MODEM_QAM32,
            lq_mod_types.LIQUID_MODEM_QAM64,
            lq_mod_types.LIQUID_MODEM_QAM128,
            lq_mod_types.LIQUID_MODEM_QAM256,
            lq_mod_types.LIQUID_MODEM_APSK4,
            lq_mod_types.LIQUID_MODEM_APSK8,
            lq_mod_types.LIQUID_MODEM_APSK16,
            lq_mod_types.LIQUID_MODEM_APSK32,
            lq_mod_types.LIQUID_MODEM_APSK64,
            lq_mod_types.LIQUID_MODEM_APSK128,
            lq_mod_types.LIQUID_MODEM_APSK256,
            lq_mod_types.LIQUID_MODEM_BPSK,
            lq_mod_types.LIQUID_MODEM_QPSK,
            lq_mod_types.LIQUID_MODEM_OOK,
            lq_mod_types.LIQUID_MODEM_SQAM32,
            lq_mod_types.LIQUID_MODEM_SQAM128,
            lq_mod_types.LIQUID_MODEM_V29,
            lq_mod_types.LIQUID_MODEM_ARB16OPT,
            lq_mod_types.LIQUID_MODEM_ARB32OPT,
            lq_mod_types.LIQUID_MODEM_ARB64OPT,
            lq_mod_types.LIQUID_MODEM_ARB128OPT,
            lq_mod_types.LIQUID_MODEM_ARB256OPT,
            lq_mod_types.LIQUID_MODEM_ARB64VT,
            lq_mod_types.LIQUID_MODEM_PI4DQPSK
        ]

sig_stream_list = lq_mod_types.to_list()[1:] + ['awgn',]

if __name__ == '__main__':
    from datagen.liquid.waterfall import waterfall
    import matplotlib.pyplot as plt
    modem = liquid.modem('qpsk')
    wf = waterfall()
    fig = wf.fig(np.array(
        [modem.modulate(int(x)) for x in np.random.randint(0,4,(5000000,))]
    ).astype(np.csingle))
    plt.show()
