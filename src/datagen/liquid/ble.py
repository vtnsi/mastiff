

import os,sys
import liquid

import numpy as np
import time
from typing import Union,List,Tuple,Dict,Any
import multiprocessing as mp
from crccheck.crc import Crc24Ble

_base_period = 625e-6

class ble_tx(object):
    def __init__(self,phy_mode=0,hop_hold=1):
        self.digital_gain = 1.0
        self.channels_carrier = np.arange(2402e6,2481e6,2e6).tolist()
        self.channel = 0
        self.auto_hop = True
        self.period = 625e-6
        self._pdu_length = 60
        self._pdu_lookup=[[1,61],[1,121]]
        self.rng = np.random.default_rng()
        self.access_address = 0x8E89BED6
        self._access_bytes = [0x8E,0x89,0xBE,0xD6]
        self.sps = 4
        self.phy_modes = ['LE1M','LE2M']
        self._phy_mode = phy_mode
        self.data_mode = 0
        self.hop_hold = hop_hold
        self._hop_hold = hop_hold
        self._data_mode_values = [1,2,3]
        self._data_modes = [
            'Data continuation',
            'Data start',
            'Control'
        ]
        self.crc_comp = Crc24Ble

        self.delay=1
        self.filt = liquid.firinterp('gmsktx',k=4,m=self.delay,beta=0.5)
        self.nco = liquid.nco(0.,0.)
        self.sensitivity = 2*np.pi*(0.5/self.sps)
        self.pinit = np.zeros((7,),np.uint8)
        self.pseq = None
        self.whiten=False


    def header_gen(self):
        llid = self._data_mode_values[self.data_mode]
        nesn = 0
        sn = 0
        md = 0
        rfu = 0
        length=self.pdu_length
        header = [(llid<<6)
                  + (nesn<<5)
                  + (sn<<4)
                  + (md<<3)
                  + (rfu),
                  length]
        return header

    def preamble_gen(self):
        if self._access_bytes[0] > 127:
            if self._phy_mode == 1:
                preamble = [170,170]
            else:
                preamble = [170]
        else:
            if self._phy_mode == 1:
                preamble = [85,85]
            else:
                preamble = [85]
        return preamble

    def step(self):
        if not self.running:
            return list()

        def whiten_message(message,chan_idx):
            msg = np.array(message,np.uint8)
            ic = np.unpackbits(np.array([(1<<6)+chan_idx],dtype=np.uint8))[1:]
            mbits = np.unpackbits(msg,bitorder='little')
            offset=0
            if np.any(np.equal(self.pinit,ic)) or self.pseq is None:
                seq = np.zeros((127,),mbits.dtype)
                for k in range(127):
                    ps7 = ic[6]
                    ps4 = ic[3]
                    seq[k] = ic[6]
                    ic = np.roll(ic,1)
                    ic[4] = ps4^ps7
                self.pseq = seq
            seq = np.repeat(self.pseq,1+np.ceil(len(mbits)/127),axis=0)
            y = np.bitwise_xor(mbits,seq[offset:offset+len(mbits)])
            self.p_init = ic
            return np.packbits(y,bitorder='little').tolist()


        data_pdu = self.header_gen()
        data_pdu += self.rng.integers(
                        0,256,
                        (self.pdu_length,),
                        dtype=np.uint8).tolist()
        crc = self.crc_comp(data_pdu)
        frame = crc.value()
        if self.whiten:
            frame = whiten_message(frame,self.channel) ### list of uint8 including data_pdu
        message = np.array(self.preamble_gen() + self._access_bytes + frame,np.uint8)
        bits = np.unpackbits(message,bitorder='little')
        symbs = bits.astype(np.single)*2-1
        d0 = symbs.astype(np.csingle)
        d1 = self.filt.execute(np.concatenate([d0,np.zeros((self.delay,),dtype=d0.dtype)]))
        d2 = d1.real.copy()*np.sqrt(2/5)
        d3 = np.ones(d2.shape,dtype=np.csingle)
        def stepper(x,y):
            s = self.sensitivity
            n = self.nco
            n.phase = n.phase + s*y
            return n.mix_up(x)
        for idx in range(len(d2)):
            d3[idx:idx+1] = stepper(d3[idx:idx+1],d2[idx:idx+1])
        data = d3[self.sps//2:self.sps//2+len(d0)*4]
        # print("ble:",len(data))
        if self.auto_hop:
            self.hop()
        return data

    @property
    def pdu_length(self):
        return self._pdu_length

    @pdu_length.setter
    def pdu_length(self,l:int):
        if l >= self.pdu_range[1]:
            raise ValueError(f"Value provided exceeds limit of {self.pdu_range[1]}")
        if l < 0:
            raise ValueError(f"Value provided is beneath limit of {self.pdu_range[0]}")
        self._pdu_length = l

    @property
    def pdu_range(self):
        return self._pdu_lookup[self._phy_mode]

    @pdu_range.setter
    def pdu_range(self,*args,**kwargs):
        raise ValueError("cannot set pdu_range")

    @property
    def bw(self):
        return self.bandwidth

    @property
    def bandwidth(self):
        return 2e6+self._phy_mode*2e6

    @bandwidth.setter
    def bandwidth(self,*args,**kwargs):
        raise ValueError("Cannot set bandwidth")

    @property
    def sample_rate(self):
        return 4e6+self._phy_mode*4e6

    @sample_rate.setter
    def sample_rate(self,*args,**kwargs):
        raise ValueError("Cannot set sample_rate")

    @property
    def frequency(self):
        return self.channels_carrier[self.channel]

    @frequency.setter
    def frequency(self,freq:Union[int,float]):
        if isinstance(freq,int):
            if freq >= 0 and freq < len(self.channels_carrier):
                self.channel = freq
            else:
                raise ValueError("The frequency value is not in the range [0,len(channels_carrier))")
        if isinstance(freq,float):
            if freq in self.channels_carrier:
                self.channel = self.channels_carrier.index(freq)
            else:
                raise ValueError("The frequency value is not in the range [0,len(channels_carrier))")

    def burst_time(self):
        samps = self.burst_length()
        return samps/self.sample_rate

    def burst_length(self):
        pdu_len = (self.pdu_length + 6 + 3 
                   + len(self._access_bytes)
                   + self._phy_mode + 1)
        samples = 8*pdu_len*self.sps
        # print(samples)
        return samples

    def hop(self):
        if self._hop_hold is not None:
            self.hop_hold = self.hop_hold - 1
            if self.hop_hold == 0:
                self.hop_hold = self._hop_hold
                setattr(self,'frequency',int(self.rng.integers(0,len(self.channels_carrier))))
        else:
            setattr(self,'frequency',int(self.rng.integers(0,len(self.channels_carrier))))

    def start(self):
        self.running=True
    def stop(self):
        self.running=0
    def wait(self):
        self.running = False

    def randomize(self):
        self._phy_mode = self.rng.integers(2)
        self._pdu_length = self.rng.integers(*self.pdu_range)
        self.channel = self.rng.integers(len(self.channels_carrier))


# def expected(pdu_len):
#     return int(2176 + 128*pdu_len)
# def analyitcs():
#     import pandas as pd,time
#     tester = zigbee_tx()
#     tester.start()
#     lls = [None]*117
#     tester.pdu_length = 0
#     for l in range(117):
#         try:
#             tester.pdu_length = l
#             burst = tester.step()
#             pred = tester.burst_length()
#             pred2 = expected(l)
#             valid = pred == len(burst)
#             valid2 = pred2 == len(burst)
#             lls[l] = (l,len(burst),len(burst)/tester.sample_rate,pred,valid,pred2,valid2)
#             time.sleep(0.01)
#         except:
#             lls[l] = (l,-1,-1,-1,False,expected(l),False)
#     l,s,d,p,v,p2,v2 = zip(*lls)
#     info = pd.DataFrame({'pdu_length':l,'samples':s,'duration':d,'prediction':p,'valid':v,'prediction2':p2,'valid2':v2})
#     tester.stop()
#     tester.wait()
#     return info


if __name__ == '__main__':
    tester = ble_tx()
    tester.pdu_length = 60
    tester.start()
    bursty = [np.squeeze(np.random.randn(int(0.002*tester.sample_rate),2).astype(np.single).view(np.csingle))]
    for _ in range(3):
        bursty.extend(
            [
                np.array(tester.step(),np.csingle),
                np.squeeze(np.random.randn(int(0.002*tester.sample_rate),2).astype(np.single).view(np.csingle))
            ]
        )
    data = np.concatenate(bursty,axis=0)
    print(len(data),data.dtype,len(data)/tester.sample_rate)
    tester.stop()
    tester.wait()
    import datagen,matplotlib.pyplot as plt
    wf = datagen.liquid.waterfall.waterfall()
    # S,t,f = wf(data)
    fig = wf.fig(data)
    # fig2,ax = plt.subplots(1)
    # ax.plot(f,np.mean(S,axis=0))
    plt.show()



        
