#!/usr/bin/env python3

import datagen
import numpy as np

if __name__ == '__main__':
    zigbee_tx = datagen.gnuradio.zigbee.zigbee_tx()
    zigbee_tx.pdu_length = 0
    burst_len = zigbee_tx.burst_length()
    burst_dur = zigbee_tx.burst_time()
    off_time = int(0.01*zigbee_tx.sample_rate)
    wf = datagen.liquid.waterfall.waterfall()
    def noise_arr(dB=-100,length=off_time):
        amp_gain = np.sqrt(0.5)*np.power(10,dB/20)
        raw_noise = np.squeeze(np.random.randn(length,2).astype(np.single).view(np.csingle))
        return amp_gain*raw_noise
    zigbee_tx.start()
    bursty = [np.zeros((off_time,),np.csingle)]
    for _ in range(3):
        bursty.extend(
            [
                np.power(10,-80/20)*np.array(zigbee_tx.step(),np.csingle),
                np.zeros((off_time,),np.csingle)
            ]
        )
    data = np.concatenate(bursty,axis=0)
    data += noise_arr(length=len(data))
    S,t,f = wf(data)
    rate = zigbee_tx.sample_rate
    dur = len(data)/rate
    print(f"Observation Duration: {dur}s at Sample Rate: {int(rate/1e6)}MHz")
    print(f"delta f: {rate/wf.nfft} :: delta t: {(dt:=np.mean(t[1:]-t[:-1]))}")
    print(f"Burst Duration: {burst_dur} :: burst_pixels: {burst_dur/dt}")
    fig = wf.fig(data)
    fig.canvas.manager.set_window_title(','.join([str(len(data)/rate),str(int(rate/1e6)),'BW~50%',f'{S.shape}']))
    import matplotlib.pyplot as plt
    plt.show()
    zigbee_tx.stop()
    zigbee_tx.wait()
