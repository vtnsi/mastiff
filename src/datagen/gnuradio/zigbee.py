

import os,sys
from gnuradio import gr,digital,blocks
from gnuradio.fft import window
import pmt

import ieee802_15_4
import numpy as np
import time
from typing import Union,List,Tuple,Dict,Any

gr.logging().set_default_level(gr.log_levels.warn)

class zigbee_msg2stream(gr.sync_block):
    def __init__(self,packet_key='packet_len',length_multiplier=128):
        self.packet_key = packet_key
        self.multplier = int(length_multiplier)
        gr.sync_block.__init__(self,'ZigbeeHandler',[],[np.uint8])
        self.in_key = pmt.intern('in')
        self.message_port_register_in(self.in_key)
        self.set_msg_handler(self.in_key,self.msg_handler)
        self.msg_q = []
        self.active_msg = None
        self.consumed = 0

    def msg_handler(self,msg):
        if pmt.is_pair(msg):
            payload = pmt.u8vector_elements(pmt.cdr(msg))
            self.msg_q.append(payload)

    def prep_message(self):
        if len(self.msg_q) == 0:
            return False
        self.active_msg = self.msg_q[0]
        self.consumed = 0
        del self.msg_q[0]
        return True

    def work(self,input_items,output_items):
        nout = len(output_items[0])
        if self.active_msg is None:
            if not self.prep_message():
                return 0
            ### self.active_msg is now a list of uint8
        available = min(nout,len(self.active_msg)-self.consumed)
        if self.consumed == 0 and available > 0:
            tag = gr.tag_t()
            tag.key = pmt.intern(self.packet_key)
            tag.offset = self.nitems_written(0)
            tag.value = pmt.from_long(len(self.active_msg)*self.multplier)
            self.add_item_tag(0,tag)
        output_items[0][:available] = self.active_msg[self.consumed:self.consumed+available]
        self.consumed += available
        if self.consumed == len(self.active_msg):
            self.active_msg = None
        return available

class zigbee_tx_hier(gr.hier_block2):
    def __init__(self,
                spc=4,
                pulse_shape=[0, float(np.sin(np.pi/4)), 1, float(np.sin(3*np.pi/4))],
                symbol_table=[
                    (1+1j), (-1+1j), (1-1j), (-1+1j), (1+1j), (-1-1j), (-1-1j), (1+1j), (-1+1j), (-1+1j), (-1-1j), (1-1j), (-1-1j), (1-1j), (1+1j), (1-1j),
                    (1-1j), (-1-1j), (1+1j), (-1-1j), (1-1j), (-1+1j), (-1+1j), (1-1j), (-1-1j), (-1-1j), (-1+1j), (1+1j), (-1+1j), (1+1j), (1-1j), (1+1j),
                    (-1+1j), (-1+1j), (-1-1j), (1-1j), (-1-1j), (1-1j), (1+1j), (1-1j), (1+1j), (-1+1j), (1-1j), (-1+1j), (1+1j), (-1-1j), (-1-1j), (1+1j),
                    (-1-1j), (-1-1j), (-1+1j), (1+1j), (-1+1j), (1+1j), (1-1j), (1+1j), (1-1j), (-1-1j), (1+1j), (-1-1j), (1-1j), (-1+1j), (-1+1j), (1-1j),
                    (-1-1j), (1-1j), (1+1j), (1-1j), (1+1j), (-1+1j), (1-1j), (-1+1j), (1+1j), (-1-1j), (-1-1j), (1+1j), (-1+1j), (-1+1j), (-1-1j), (1-1j),
                    (-1+1j), (1+1j), (1-1j), (1+1j), (1-1j), (-1-1j), (1+1j), (-1-1j), (1-1j), (-1+1j), (-1+1j), (1-1j), (-1-1j), (-1-1j), (-1+1j), (1+1j),
                    (1+1j), (-1-1j), (-1-1j), (1+1j), (-1+1j), (-1+1j), (-1-1j), (1-1j), (-1-1j), (1-1j), (1+1j), (1-1j), (1+1j), (-1+1j), (1-1j), (-1+1j),
                    (1-1j), (-1+1j), (-1+1j), (1-1j), (-1-1j), (-1-1j), (-1+1j), (1+1j), (-1+1j), (1+1j), (1-1j), (1+1j), (1-1j), (-1-1j), (1+1j), (-1-1j),
                    (1+1j), (1-1j), (1+1j), (-1+1j), (1-1j), (-1+1j), (1+1j), (-1-1j), (-1-1j), (1+1j), (-1+1j), (-1+1j), (-1-1j), (1-1j), (-1-1j), (1-1j),
                    (1-1j), (1+1j), (1-1j), (-1-1j), (1+1j), (-1-1j), (1-1j), (-1+1j), (-1+1j), (1-1j), (-1-1j), (-1-1j), (-1+1j), (1+1j), (-1+1j), (1+1j),
                    (-1-1j), (1+1j), (-1+1j), (-1+1j), (-1-1j), (1-1j), (-1-1j), (1-1j), (1+1j), (1-1j), (1+1j), (-1+1j), (1-1j), (-1+1j), (1+1j), (-1-1j),
                    (-1+1j), (1-1j), (-1-1j), (-1-1j), (-1+1j), (1+1j), (-1+1j), (1+1j), (1-1j), (1+1j), (1-1j), (-1-1j), (1+1j), (-1-1j), (1-1j), (-1+1j),
                    (-1-1j), (1-1j), (-1-1j), (1-1j), (1+1j), (1-1j), (1+1j), (-1+1j), (1-1j), (-1+1j), (1+1j), (-1-1j), (-1-1j), (1+1j), (-1+1j), (-1+1j),
                    (-1+1j), (1+1j), (-1+1j), (1+1j), (1-1j), (1+1j), (1-1j), (-1-1j), (1+1j), (-1-1j), (1-1j), (-1+1j), (-1+1j), (1-1j), (-1-1j), (-1-1j),
                    (1-1j), (-1+1j), (1+1j), (-1-1j), (-1-1j), (1+1j), (-1+1j), (-1+1j), (-1-1j), (1-1j), (-1-1j), (1-1j), (1+1j), (1-1j), (1+1j), (-1+1j),
                    (1+1j), (-1-1j), (1-1j), (-1+1j), (-1+1j), (1-1j), (-1-1j), (-1-1j), (-1+1j), (1+1j), (-1+1j), (1+1j), (1-1j), (1+1j), (1-1j), (-1-1j)],
                symbol_table_dimension=16,
                step_size=2**16,
                packet_key='packet_len',
                debug=False):
        gr.hier_block2.__init__(
            self, "ZigbeeProtocol",
                gr.io_signature(0, 0, 0),
                gr.io_signature(1, 1, gr.sizeof_gr_complex*1),
        )
        self.message_port_register_hier_in("in")

        ##################################################
        # Variables
        ##################################################


        ##################################################
        # Blocks
        ##################################################

        self.pdu2bytes = zigbee_msg2stream('packet_len',2*spc*symbol_table_dimension)
        self.pdu2bytes.set_min_noutput_items(1)
        self.pdu2bytes.set_min_output_buffer(1024)
        self.bytes_shaper = blocks.packed_to_unpacked_bb(4, gr.GR_LSB_FIRST)
        self.bytes_shaper.set_min_noutput_items(1)
        self.bytes_shaper.set_min_output_buffer(1024)
        self.bytes2symbols = digital.chunks_to_symbols_bc(symbol_table, symbol_table_dimension)
        self.bytes2symbols.set_min_noutput_items(1)
        self.bytes2symbols.set_min_output_buffer(262144)
        self.shape_source = blocks.vector_source_c(pulse_shape[:spc], True, 1, [])
        self.shape_source.set_min_noutput_items(1)
        self.shape_source.set_min_output_buffer(262144)
        self.sq_interp = blocks.repeat(gr.sizeof_gr_complex*1, spc)
        self.sq_interp.set_min_noutput_items(1)
        self.sq_interp.set_min_output_buffer(262144)
        self.tag_blocker = blocks.tag_gate(gr.sizeof_float * 1, False)
        self.tag_blocker.set_single_key("")
        self.tag_blocker.set_min_noutput_items(1)
        self.tag_blocker.set_min_output_buffer(262144)
        self.apply_shape = blocks.multiply_vcc(1)
        self.c2f = blocks.complex_to_float(1)
        self.c2f.set_min_noutput_items(1)
        self.c2f.set_min_output_buffer(262144)
        self.delayer = blocks.delay(gr.sizeof_float*1, 2)
        self.delayer.set_min_noutput_items(1)
        self.delayer.set_min_output_buffer(262144)
        self.f2c = blocks.float_to_complex(1)
        self.f2c.set_min_noutput_items(1)
        self.f2c.set_min_output_buffer(262144)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self, 'in'), (self.pdu2bytes, 'in'))
        # self.connect((self.pdu2bytes, 0), (self.tag_multipler, 0))
        # self.connect((self.tag_multipler, 0), (self.bytes_shaper, 0))
        self.connect((self.pdu2bytes, 0), (self.bytes_shaper, 0))
        self.connect((self.bytes_shaper, 0), (self.bytes2symbols, 0))
        self.connect((self.bytes2symbols, 0), (self.sq_interp, 0))
        self.connect((self.sq_interp, 0), (self.apply_shape, 0))
        self.connect((self.shape_source, 0), (self.apply_shape, 1))
        self.connect((self.apply_shape, 0), (self.c2f, 0))
        self.connect((self.c2f, 0), (self.f2c, 0))
        self.connect((self.c2f, 1), (self.tag_blocker, 0))
        self.connect((self.delayer, 0), (self.f2c, 1))
        self.connect((self.tag_blocker, 0), (self.delayer, 0))
        self.connect((self.f2c, 0), (self, 0))

    def post(self,msg:pmt.pmt_base):
        if isinstance(msg,pmt.pmt_base):
            if pmt.is_pair(msg):
                data = pmt.cdr(msg)
            else:
                data = msg
            self.pdu2bytes.to_basic_block()._post(pmt.intern("in"),pmt.cons(pmt.PMT_NIL,data))




class zigbee_tx(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self,'zigbee_tx')

        ##########################################
        self.digital_gain = 1.0
        self.sample_rate = 4.0e6
        self.bw = 2.0e6
        self.pdu_length = 116
        self.pdu_range = [0,117] #-> [0,117)
        self.period = 0.06
        self.period_range = [0.006,1.0]
        self.channels_carrier = np.arange(2405e6,2455e6+1,5e6).tolist()
        self.channel = 0
        self.encoding_types = list(range(8))
        self.encoding = 0
        self.rng = np.random.default_rng()

        self.auto_hop = False

        access_preamble=0x00000000

        ##########################################

        self.mac = ieee802_15_4.mac(True,
                                    self.rng.integers(0,2**16).item(),
                                    self.rng.integers(0,2**16).item(),
                                    self.rng.integers(0,2**16).item(),
                                    self.rng.integers(0,2**16).item(),
                                    self.rng.integers(0,2**16).item())
        self.phy_wrap = ieee802_15_4.access_code_prefixer(
            (access_preamble>>24)&0xff,
            ((access_preamble&0xffffff)<<80)+0xA7)

        self.phy = zigbee_tx_hier()

        self.gain = blocks.multiply_const_cc(self.digital_gain)
        self.sink = blocks.vector_sink_c()
        self.gain.set_min_noutput_items(1)

        ##########################################
        self.msg_connect((self.mac,'pdu out'),(self.phy_wrap,"in"))
        self.msg_connect((self.phy_wrap,'out'),(self.phy,"in"))
        self.connect((self.phy,0),(self.gain,0))
        self.connect((self.gain,0),(self.sink,0))

        self.running = False

    def gen_msg(self,length):
        return np.frombuffer(self.rng.bytes(length),np.uint8).tolist()

    def burst_length(self):
        return int(2176 + 128*self.pdu_length)

    def burst_time(self):
        return self.burst_length()/self.sample_rate

    def step(self):
        if not self.running:
            return list()
        message = self.gen_msg(self.pdu_length)
        self.last_pdu = pmt.cons(pmt.PMT_NIL,pmt.init_u8vector(len(message),message))
        self.mac.to_basic_block().post(pmt.intern('app in'),self.last_pdu)
        while len(self.sink.data()) < self.burst_length() and self.running:
            time.sleep(0.001)
        data = self.sink.data()
        self.sink.reset()
        # print("zigbee:",len(data))
        if self.auto_hop:
            self.hop()
        return data
    
    def hop(self):
        setattr(self,'frequency',int(self.rng.integers(0,len(self.channels_carrier))))

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

    def randomize(self):
        self.frequency = int(self.rng.integers(len(self.channels_carrier)))
        self.pdu_length = int(self.rng.integers(max(self.pdu_range)))


    def start(self):
        super().start()
        self.running=True
    def stop(self):
        super().stop()
        self.running=0
    def wait(self):
        super().wait()
        self.running = False

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
    tester = zigbee_tx()
    tester.pdu_length = 116
    tester.start()
    bursty = [np.squeeze(np.random.randn(int(0.5*tester.sample_rate),2).astype(np.single).view(np.csingle))]
    for _ in range(3):
        bursty.extend(
            [
                np.array(tester.step(),np.csingle),
                np.squeeze(np.random.randn(int(0.5*tester.sample_rate),2).astype(np.single).view(np.csingle))
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

