

from typing import Union,List,Tuple,Dict,Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from .signal_stream import AWGNStream,SignalStream
    from .waterfall import waterfall
except ImportError:
    from datagen.liquid.signal_stream import AWGNStream,SignalStream
    from datagen.liquid.waterfall import waterfall

Hertz = (float)
Second = (float)
dB = (float)
Samples = (int)
SampArray = (np.ndarray)
FcBwT0Tn = (Tuple)

class SignalContainer(object):
    '''
    SignalContainer

    Parameters:
        signal (SignalStream): Signal to contain
        period (int, `samples`): time between bursts in samples
        gain (float, `dB`): direct gain in dB to apply to the signal (SNR eventually replacing this)
        t0 (float, `seconds`): initial burst timing delay (seconds)
        fc (float, `hertz`): carrier frequency to initially use (any hops will update this)
    '''
    def __init__(self,signal:SignalStream,period:Samples,
                 gain:dB,t0:Samples,fc:Hertz):
        self.signal = signal    #signal
        self.period = period    #period
        self.gain = gain        #gain
        self.carrier = fc       #current carrier
        self.t0 = t0            #starting sample
        self.counter = 0        #number of steps taken
        self.next = t0          #next sample for the burst
        self.cache = []         #leftover from previos step

    def package(self,
                samples:SampArray,
                s0:Samples,
                s1:Samples,
                t0:Samples,
                t1:Samples,
                time_window:np.ndarray,
                bounds:FcBwT0Tn,
                n:Samples):
        ''' Package up the given samples with the given contraints

        handle whether the signal burst is part of this step, store if not
        
        Parameters:
            samples (np.ndarray): the IQ samples at baseband
            s0 (int): starting sample number of this sample array
            s1 (int): ending sample number of this sample array
            t0 (int): starting sample number of this step
            t1 (int): ending sample number of this step
            time_window (np.ndarray): array of all samples in this step
            bounds (tuple(float, float, int, int)): frequency (min,max) and time (min,max) bounds of the signal
            n (int): adjustment factor based on step duration in samples

        Returns:
            samples (np.ndarray): the array of samples that belong in this step
            in_bound (np.ndarray): logical array of which samples are bounds
            bounds (tuple): the adjusted bounds for this step

        '''
        in_bound = np.logical_and(time_window>=s0,time_window<s1)
        srange = np.arange(s0,s1)
        if np.sum(in_bound) == len(samples):
            return samples,in_bound,bounds
        else:
            push_out = np.logical_and(srange >= t0, srange < t1)
            store = srange >= t1
            if np.sum(store) > 0:
                bounder = (bounds[0],bounds[1],
                        t1-n*(self.counter+1),
                        bounds[3]-n*(self.counter+1))
                bounds = (bounds[0],bounds[1],
                        bounds[2],
                        t1-n*self.counter)
                self.cache.append((samples[store],t1,s1,bounder))
            else:
                print("does this even happen?")
                bounds = (bounds[0],bounds[1],bounds[2],
                        s1-n*self.counter)
        return samples[push_out],in_bound,bounds

    def step(self,t0:Samples,t1:Samples):
        ''' Step

        Return the samples for this signal within the sample bounds given

        Parameters:
            t0 (int): Starting sample for this sample
            t1 (int): Starting sample for the next sample

        Returns:
            bursts(list(tuple(np.ndarray,nd.array,tuple(float,float,int,int)))): List of the bursts samples and bounds
        '''
        n = t1-t0
        bursts = []
        time_window = np.arange(t0,t1)
        gain = np.power(10,self.gain/20)
        while len(self.cache):
            chunk = self.cache[0]
            del self.cache[0]
            samps,s0,s1,bounds = chunk
            # print(bounds[0],bounds[1],bounds[2]/n,bounds[3]/n)
            samps,valid_samps,bounder = self.package(samps,s0,s1,t0,t1,
                                    time_window,bounds,n)
            bursts.append((samps,valid_samps,bounder))

        while self.next < t1:
            fc,samps = self.signal.step()
            #### moot if autohop
            # if self.signal._mod not in ['unknown',None] and self.signal._prot not in ['unknown',None]:
            #     if hasattr(self.signal._prot_gen,'auto_hop') and self.signal._prot_gen.auto_hop:
            #         self.signal._prot_gen.hop()
            df = (fc-self.carrier)/self.signal.sample_rate      # frequency shift to sample rate ratio
            dw = self.signal.bandwidth/self.signal.sample_rate  # bandwidth to sample rate ratio
            s0 = self.next                                      # first sample for this burst
            s1 = len(samps) + s0                                # last sample + 1
            if hasattr(self.signal,'bbox_adjust'):
                adjustments = self.signal.bbox_adjust()
            else:
                adjustments = (0,0,0,0)

            bounder = (
                adjustments[0] + df,                        #### adjustment in percentage 
                adjustments[1] + dw,                        #### adjustment in percentage
                max(0,adjustments[2] + s0-n*self.counter),  #### adjustment in samples
                min(n,adjustments[3] + s1-n*self.counter)   #### adjustment in samples
            )
            srange = np.arange(s0,s1)           ### samples to fill
            shift = np.exp(2j*np.pi*df*srange)  ### frequency shift for samples
            #print('type: ', type(gain), type(samps), type(shift))
            samps = gain*samps*shift            ### gain and shift

            # print(bounder[0],bounder[1],bounder[2]/n,bounder[3]/n)
            samps,valid_samps,bounds = self.package(samps,s0,s1,
                                        t0,t1,time_window,bounder,n)
            bursts.append((samps,valid_samps,bounds))
            self.next += self.period
        self.counter += 1
        return bursts

    def close(self):
        ''' Close

        Cleanup any loose signal junk

        '''
        self.signal.close()

class Spectrum(object):
    ''' Spectrum Generation

    This will produce spectrum in image and numpy array format.

    Parameters:
        observation_bandwidth (float, `Hertz`): The instantaneous bandwidth of this observation point
        observation_carrier (float, `Hertz`): The center frequency of this observation point
        observation_duration (float, `seconds`): The duration that one step should produce
        noise_power_dB (float, `dB`): the gain to apply to `CN(0,1)` noise source that will be injected
        power_lower_bound (float, `dB`): The lower bound for the figure to clip at
        power_upper_bound (float, `dB`): The upper bound for the figure to clip at
        seed (list(int),int,`optional`): Specify a seed for random generation of the noise
    '''
    def __init__(self,
                 observation_bandwidth:Hertz=100e6,
                 observation_carrier:Hertz=2.45e9,
                 observation_duration:Second=0.05,
                 noise_power_dB:dB=-100,
                 power_lower_bound:dB=-110.,
                 power_upper_bound:dB=-30.,
                 seed=None):
        self.sample_rate = observation_bandwidth
        self.step_time = observation_duration
        self.step_size = observation_duration*observation_bandwidth
        self.N0 = noise_power_dB
        self.vmin = power_lower_bound
        self.vmax = power_upper_bound
        self.carrier = observation_carrier
        self.signals = []
        self.time_step = -1
        self.excess = None
        self.simulation_time = None
        self.seed = seed
        self.base_rng = np.random.default_rng(seed)
        self.noise = AWGNStream(self.N0,
                                np.random.default_rng(self.base_rng.integers(
                                    0,
                                    np.iinfo(np.int64).max,
                                    size=(10,))),
                                self.sample_rate,
                                self.sample_rate)
        self.wf = waterfall(sample_rate=self.sample_rate,
                            center_frequency=observation_carrier)
        self.wf.vmin = power_lower_bound
        self.wf.vmax = power_upper_bound
        self._disable_fig_gen = False

    def add_signal(self,signal:SignalStream,period:float,gain:float,t0:float):
        ''' register signal with spectrum

        Make sure the spectrum is aware of signals that might occur within it

        Parameters:
            signal (SignalStream): the signal generator source
            period (float, `seconds`): the period with which this signal will be generated
            gain (float, `dB`): the gain to apply to the samples coming out of this generator
            t0 (float, `seconds`): The initial delay before any bursts of this signal come out.
        '''
        nsamps = int(period*self.sample_rate)
        init_delay = int(t0*self.sample_rate)
        self.signals.append(SignalContainer(signal,nsamps,gain,
                                            init_delay,self.carrier))

    def step(self):
        ''' Generate next spectrogram

        Returns:
            fig (matplotlib.pyplot.figure): a figure handle to the image of this spectrogram
            S (np.ndarray): Spectrogram matrix in dB (freq along x, time along y when plotting)
            t (np.ndarray): time values corresponding to the spectrogram (seconds)
            f (np.ndarray): frequency values corresponding to the spectrogram (hertz)
            frame (np.ndarray): IQ samples used to generate the spectrogram
            bounds (tuple): where the signal _should_ be in the spectrogram time/freq
        '''
        self.time_step += 1
        frame = self.noise.generate(self.step_size)
        bounds = []
        time_window = [self.step_time*self.time_step,
                       self.step_time*(self.time_step+1)]
        samp_window = [round(x*self.sample_rate) for x in time_window]
        for sig_idx,sig in enumerate(self.signals):
            bursts = sig.step(*samp_window)


            for burst in bursts:
                samps,indices,bound = burst
                bound = (sig_idx,
                         bound[0]-bound[1]/2 + 0.5,
                         bound[0]+bound[1]/2 + 0.5,
                         bound[2]/self.step_size,
                         bound[3]/self.step_size)
                bounds.append(bound)
                frame[indices] += samps
        fig = self.wf.fig(frame) if not self._disable_fig_gen else None
        S,t,f = self.wf(frame)
        return fig,S,t,f,frame,bounds
    
    def nsteps(self, n:int):
        ''' Generate next spectrogram n times

        Returns:
            steps (list): n steps in a list
        '''
        steps = [None]*n
        for idx in range(n):
            steps[idx] = self.step()
        return steps

    def close(self):
        ''' Wrap up and dangling objects
        '''
        for sig in self.signals:
            sig.close()



def main():
    import matplotlib.pyplot as plt
    spec = Spectrum()
    wifi = SignalStream(protocol='wifi')
    wifi.sample_rate = 100e6
    zigbee = SignalStream(protocol='zigbee')
    zigbee.sample_rate = 100e6
    ble = SignalStream(protocol='ble')
    ble.sample_rate = 100e6
    try:
        spec.add_signal(wifi,0.0278,-80,0.01)
        spec.add_signal(zigbee,0.0153,-80,0.005)
        spec.add_signal(ble,625e-6,-80,0.034)
        for _ in range(5):
            fig,matrix,timesteps,freqs,samples,bounds = spec.step()
            for b in bounds:
                print(_,b)
            # plt.close(fig)
    finally:
        spec.close()
    plt.show()


def tweak():
    import matplotlib.pyplot as plt
    spec = Spectrum()
    wifi = SignalStream(modulation='qam64',protocol='wifi')
    wifi.sample_rate = 100e6
    zigbee = SignalStream(modulation='qam64',protocol='zigbee')
    zigbee.sample_rate = 100e6
    ble = SignalStream(modulation='qam64',protocol='ble')
    ble.sample_rate = 100e6
    try:
        spec.add_signal(wifi,0.0278,-80,0.01)
        spec.add_signal(zigbee,0.0153,-80,0.005)
        spec.add_signal(ble,625e-6,-80,0.034)
        for _ in range(5):
            fig,matrix,timesteps,freqs,samples,bounds = spec.step()
            for b in bounds:
                print(_,b)
            # plt.close(fig)
    finally:
        spec.close()
    plt.show()

#if __name__ == '__main__':
#    main()
    # tweak()

