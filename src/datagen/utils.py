

import numpy as np

try:
    from .liquid.signal_stream import SignalStream
except ImportError:
    from datagen.liquid.signal_stream import SignalStream

def resovle_random_range(param_range,rng=None,seed=None):
    rng = rng if rng is not None else np.random.default_rng(seed)
    if not isinstance(param_range,dict):
        return param_range
    else:
        if 'items' in param_range:
            value = rng.choice(param_range['items'],p=param_range['weight'] if 'weight' in param_range else None)
        elif 'std' in param_range or 'mean' in param_range:
            descale = False
            if 'scale' in param_range:
                if param_range['scale'].lower() == 'log':
                    scale = np.log10(param_range['scale'])
                else:
                    scale = param_range['scale']
            else:
                scale = 1.0
            value = rng.normal(loc=param_range['mean'] if 'mean' in param_range else 0.0,
                               scale=scale)
            if 'min' in param_range and value < param_range['min']:
                value = param_range['min']
            if 'max' in param_range and value > param_range['max']:
                value = param_range['max']
            if descale:
                value = np.power(10.0,value)
        elif 'min' in param_range or 'max' in param_range:
            descale = False
            if 'scale' in param_range and param_range['scale'].lower() == 'log':
                vmin = np.log10(param_range['min'])
                vmax = np.log10(param_range['max'])
                descale = True
            else:
                vmin = param_range['min']
                vmax = param_range['max']
            value = rng.uniform(vmin,vmax)
            if descale:
                value = np.power(10.0,value)
        else:
            raise ValueError(f"Not sure how to parse this range: {param_range}")
    if issubclass(value.__class__,np.number):
        if isinstance(value.__class__,np.integer):
            value = int(value)
        else:
            value = float(value)
    elif isinstance(value,np.ndarray):
        value = value.tolist()
    return value

def stream_creation(params,rng=None,seed=None):
    rng = rng if rng is not None else np.random.default_rng(seed)
    required_keys = ['protocol','modulation']

    config = dict()
    for param in required_keys:
        config[param] = resovle_random_range(params[param],rng)
    return SignalStream(modulation=config['modulation'],protocol=config['protocol'])

def eval_random_range_config(config:dict,rng=None,seed=None):
    rng = rng if rng is not None else np.random.default_rng(seed)
    required_keys = ['protocol','modulation','f0','relative_gain','t0','period','max_count']
    if any([x not in config for x in required_keys]):
        raise RuntimeError(f"Missing keys from {required_keys}")

    config = config.copy()
    N = resovle_random_range(config['max_count'],rng)
    del config['max_count']

    configs = [None]*N
    for idx in range(N):
        params = dict()
        for param in required_keys[:-1]:
            params[param] = resovle_random_range(config[param],rng)
        configs[idx] = params
    return configs











