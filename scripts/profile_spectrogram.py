#!/usr/bin/env python3


import datagen
import numpy as np

if __name__ == '__main__':

    wifi_stream_min = datagen.liquid.signal_stream.SignalStream(protocol='wifi')
    zigbee_stream_min = datagen.liquid.signal_stream.SignalStream(protocol='zigbee')

    #wifi_stream_min.start()
    #zigbee_stream_min.start()

    try:
        wifi_stream_min.sample_rate = 100e6
        wifi_stream_min.set_protocol(encoding=7)
        wifi_stream_min.set_protocol_payload_length(0)

        zigbee_stream_min.sample_rate = 100e6
        zigbee_stream_min.set_protocol_payload_length(0)

        print("wifi stream sample rate:",wifi_stream_min.sample_rate)
        print("zigbee stream sample rate:",zigbee_stream_min.sample_rate)

        print("WiFi burst min duration:",wifi_stream_min.duration)
        print("Zigbee burst min duration:",zigbee_stream_min.duration)

        wifi_burst = wifi_stream_min.step()[1]
        wifi_burst *= np.exp(2j*np.pi*np.arange(len(wifi_burst))*-0.33)
        zigbee_burst = zigbee_stream_min.step()[1] 
        zigbee_burst *= np.exp(2j*np.pi*np.arange(len(zigbee_burst))*-0.1)

        print("WiFi burst length at rate:",len(wifi_burst))
        print("Zigbee burst length at rate:",len(zigbee_burst))

        print(wifi_burst.shape,zigbee_burst.shape)
        min_bursts = np.concatenate([wifi_burst,zigbee_burst],axis=0)
        print(min_bursts.shape)
    finally:
        wifi_stream_min.stop()
        zigbee_stream_min.stop()
        wifi_stream_min.wait()
        zigbee_stream_min.wait()

    wifi_stream_max = datagen.liquid.signal_stream.SignalStream(protocol='wifi')
    zigbee_stream_max = datagen.liquid.signal_stream.SignalStream(protocol='zigbee')

    #wifi_stream_max.start()
    #zigbee_stream_max.start()

    try:
        wifi_stream_max.sample_rate = 100e6
        wifi_stream_max.set_protocol(encoding=0)
        wifi_stream_max.set_protocol_payload_length(980)

        zigbee_stream_max.sample_rate = 100e6
        zigbee_stream_max.set_protocol_payload_length(116)

        print("WiFi burst max duration:",wifi_stream_max.duration)
        print("Zigbee burst max duration:",zigbee_stream_max.duration)

        wifi_burst = wifi_stream_max.step()[1]
        wifi_burst *= np.exp(2j*np.pi*np.arange(len(wifi_burst))*0.33)
        zigbee_burst = zigbee_stream_max.step()[1] 
        zigbee_burst *= np.exp(2j*np.pi*np.arange(len(zigbee_burst))*0.1)

        print("WiFi burst length at rate:",len(wifi_burst))
        print("Zigbee burst length at rate:",len(zigbee_burst))

        print(wifi_burst.shape,zigbee_burst.shape)
        max_bursts = np.concatenate([wifi_burst,zigbee_burst],axis=0)
        print(max_bursts.shape)
    finally:
        wifi_stream_max.stop()
        zigbee_stream_max.stop()
        wifi_stream_max.wait()
        zigbee_stream_max.wait()

    bursts = np.concatenate([min_bursts,max_bursts],axis=0)
    print(bursts.shape)

    noise_count = int(0.05*100e6 - len(bursts))
    front = int(noise_count//2)
    back = int(noise_count-front)
    snapshot = 0.001*np.concatenate([np.zeros((front,),dtype=bursts.dtype),
                                     bursts,
                                     np.zeros((back,),dtype=bursts.dtype)])

    print(snapshot.shape)

    snapshot += np.squeeze(np.sqrt(.5)*0.001*np.random.randn(len(snapshot),2).astype(np.single).view(np.csingle))
    

    wf = datagen.liquid.waterfall.waterfall()
    S,t,f = wf(snapshot)

    fig = wf.fig(snapshot)
    fig.canvas.manager.set_window_title(','.join([str(len(snapshot)/100e6),str(int(100e6/1e6)),f'{S.shape}']))
    import matplotlib.pyplot as plt
    plt.show()


    # zigbee_tx = datagen.gnuradio.zigbee.zigbee_tx()
    # zigbee_tx.pdu_length = 0
    # burst_len = zigbee_tx.burst_length()
    # burst_dur = zigbee_tx.burst_time()
    # off_time = int(0.01*zigbee_tx.sample_rate)
    # wf = datagen.liquid.waterfall.waterfall()
    # def noise_arr(dB=-100,length=off_time):
    #     amp_gain = np.sqrt(0.5)*np.power(10,dB/20)
    #     raw_noise = np.squeeze(np.random.randn(length,2).astype(np.single).view(np.csingle))
    #     return amp_gain*raw_noise
    # zigbee_tx.start()
    # bursty = [np.zeros((off_time,),np.csingle)]
    # for _ in range(3):
    #     bursty.extend(
    #         [
    #             np.power(10,-80/20)*np.array(zigbee_tx.step(),np.csingle),
    #             np.zeros((off_time,),np.csingle)
    #         ]
    #     )
    # data = np.concatenate(bursty,axis=0)
    # data += noise_arr(length=len(data))
    # S,t,f = wf(data)
    # rate = zigbee_tx.sample_rate
    # dur = len(data)/rate
    # print(f"Observation Duration: {dur}s at Sample Rate: {int(rate/1e6)}MHz")
    # print(f"delta f: {rate/wf.nfft} :: delta t: {(dt:=np.mean(t[1:]-t[:-1]))}")
    # print(f"Burst Duration: {burst_dur} :: burst_pixels: {burst_dur/dt}")
    # fig = wf.fig(data)
    # fig.canvas.manager.set_window_title(','.join([str(len(data)/rate),str(int(rate/1e6)),'BW~50%',f'{S.shape}']))
    # import matplotlib.pyplot as plt
    # plt.show()
    # zigbee_tx.stop()
    # zigbee_tx.wait()

