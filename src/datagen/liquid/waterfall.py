

import liquid
import matplotlib.pyplot as plt
import numpy as np


def fullscale_scaler(S,t,f):
    # print("-A",np.min(S),np.max(S))
    smax = np.max(S)
    smin = np.min(S)
    S = S-smin
    if not np.isclose(0,smax-smin):
        S = S/(smax-smin)
    # print("A-",np.min(S),np.max(S))
    return S,t,f

def fullscale_with_ref(S,t,f,nf,ref):
    # print("-B",np.min(S),np.max(S))
    shift = np.min(S)
    scale = np.max(S)-shift if not np.isclose(0,np.max(S)-shift) else 1.0
    S,t,f = fullscale_scaler(S,t,f)
    mapped_to = (nf-shift)/scale
    ref_shift = mapped_to-ref
    S = S-ref_shift
    # print("B-",np.min(S),np.max(S))
    return S,t,f

class waterfall(object):
    def __init__(self,sample_rate=5e6,center_frequency=2.45e9,nfft=256,window_length=250,stride=167,time_bins=200,window_type='hamming'):
        self.fs = sample_rate
        self.fc = center_frequency
        self.nfft = nfft
        self.wl = window_length
        self.stride = stride
        self.time_min_len = time_bins # max_len is 2x-1
        self.window = window_type
        self.vmin = None
        self.vmax = None
        self.spectrum_raw_scale = None
        self.spectrum_fig_scale = None
        self.cmap = 'gray'
        self.remake()

    def remake(self):
        self.maker = liquid.spwaterfall(self.nfft,self.time_min_len,self.wl,self.stride,self.window)

    def __call__(self,input):
        self.maker.execute(input)
        S,t,f = self.maker.get_psd(self.fs,self.fc)
        self.maker.reset()
        S = S.T.copy()
        if self.spectrum_raw_scale is not None and callable(self.spectrum_raw_scale):
            S,t,f = self.spectrum_raw_scale(S,t,f)
        return S,t,f

    def fig(self,input,**fig_kwargs):
        fig = plt.figure(frameon=False,**fig_kwargs)
        ax = plt.Axes(fig,[0,0,1,1])
        ax.set_axis_off()
        fig.add_axes(ax)
        S,t,f = self(input)
        if self.spectrum_fig_scale is not None and callable(self.spectrum_fig_scale):
            S,t,f = self.spectrum_fig_scale(S,t,f)
        ax.imshow(S,extent=(f[0],f[-1],t[-1],t[0]),aspect='auto',cmap=self.cmap,
                  vmin=self.vmin,vmax=self.vmax,origin='lower')
        return fig


if __name__ == '__main__':
    input = np.random.randn(int(5e6),2).astype(np.float32).view(np.csingle)
    tester = waterfall()
    fig = tester.fig(np.squeeze(input))
    plt.show()

