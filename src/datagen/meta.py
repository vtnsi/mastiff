
import numpy as np
import pandas as pd
from typing import Union,List,Tuple,Dict,Any

import json as jn
import yaml as ym

try:
    from .liquid.signal_stream import SignalStream
except ImportError:
    from datagen.liquid.signal_stream import SignalStream

class MetaItemDescription(object):
    def __init__(self,*,name='',**kwargs):
        self.signal = None
        self.band_center = None
        self.signal_mobility = None
        self.signal_qualifier = None
        self.source_type = None
        self.channel_type = None
        self.imperfections = None
        self.period = None
        self.duration = None
        self.span = None
        self.channels = None
        self.gain = None
        self.snr = None



class MetaGen(object):
    def __init__(self,*,
                 json:str=None,
                 yaml:str=None,
                 description:Dict=None,
                 signals:List[SignalStream]=None,
                 probabilites:List[float]=None):
        if sum([x is not None for x in [json,yaml,description,signals]]) != 1:
            raise ValueError("Must intialize the meta generator with some form of parameter space")
        self.init_condition = dict()
        if json:
            self.init_condition['json'] = json
            self.items = meta_json_loader(json,probabilites)
        if yaml:
            self.init_condition['yaml'] = yaml
            self.items = meta_yaml_loader(yaml,probabilites)
        if description:
            self.init_condition['description'] = description
            self.items = meta_description_loader(description,probabilites)
        if signals:
            self.init_condition['signals'] = signals
            self.items = meta_description_builder(signals,probabilites)
        self.init_condition['probabilites'] = probabilites

    def step(self):
        return "spectrum info","signal info"


def meta_json_loader(filepath:str,probabilities:List[float]=None):
    return [MetaItemDescription(name='test')]
def meta_yaml_loader(filepath:str,probabilities:List[float]=None):
    return [MetaItemDescription(name='test')]
def meta_description_loader(description:Dict,probabilities:List[float]=None):
    return [MetaItemDescription(name='test')]
def meta_description_builder(signals:List[SignalStream],probabilities:List[float]=None):
    return [MetaItemDescription(name='test')]
