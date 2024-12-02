import argparse
import numpy as np
import json
import os
import datagen
from typing import Union
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tqdm

# example:
### python scripts/dataset_gen.py --out-fmt yolo --yolo-dims 640,640 --root-out ../TESTING spectrum --nfft 256 --duration 0.05 --rate 100e6 --carrier 2.45e9 --scaling nfref --count 5 --nf -100.0 --nfref 0.125 --reset-steps 2 --waveform-config scenarios/default_spectrum_gen.json

def parse_args():
    p = argparse.ArgumentParser(prog='dataset_generator')

    sub = p.add_subparsers(
        title='target',
        dest='target',
        help='target signal output [-h] [--help] for more about the target'
    )

    waveform_template = {
        'type': str,
        'default': None,
        'help': 'The configuration file describing the waveforms desired'
    }

    modulation = sub.add_parser('modulation',help=(
        'Creating a database of modulations (after burst isolation)'))
    stream = sub.add_parser('stream',help=(
        'Creating a database of bursts from waveforms (after frequency isolation)'))
    spectrum = sub.add_parser('spectrum',help=(
        'Creating a spectrum of signals with no isolation'))


    modulation.add_argument('--waveform-config', **waveform_template)
    stream.add_argument('--waveform-config', **waveform_template)
    spectrum.add_argument('--waveform-config', **waveform_template)


    spectrum.add_argument('--nfft',default=256,type=int,help='FFT size for the spectrogram')
    spectrum.add_argument('--duration',default=0.05,type=float,help='Duration of the spectrogram for generation (s)')
    spectrum.add_argument('--carrier',default=2.45e9,type=float,help='Center frequency to use')
    spectrum.add_argument('--rate',default=100e6,type=float,help='Sample rate to use')
    spectrum.add_argument('--color',action='store_true',help='Save the image in color, rather than grayscale')
    spectrum.add_argument('--scaling',choices=['minmax','fullscale','nfref','native'],
        default='minmax',
        help=('How should the image be scaled:'
              '    (minmax: clip in dB by the min and max values given)'
              '    (fullscale: directly adjust [min,max] to the [0,1] range)'
              '    (nfref: fullscale, but consistent noise floor value, clip outside [0,1])'
              '    (native: save directly as numpy array, no scaling)'
              ))
    spectrum.add_argument('--vmin',type=float,default=-110,help='Under minmax scale, the lower clip value (dB)')
    spectrum.add_argument('--vmax',type=float,default=-30,help='Under minmax scale, the upper clip value (dB)')
    spectrum.add_argument('--nfref',type=float,default=0.1,help='Under nfref scale, the target noise floor, clips outside 0,1 after noise floor adjustment (dB)')
    spectrum.add_argument('--count',type=int,default=100,help='How many spectrograms should be generated?')
    spectrum.add_argument('--nf',type=float,default=-100.0,help='What should be the noise floor for generation')
    spectrum.add_argument('--reset-steps',type=int,default=None,help='Number of steps before randomly regenerating specturm object')

    def cs_pair(value):
        if ',' not in value:
            raise argparse.ArgumentError("Expected a value of the type 'int,int'")
        try:
            val = [int(x) for x in value.split(',')]
        except:
            raise argparse.ArgumentError("Expected a value of the type 'int,int'")
        return val

    p.add_argument("--out-fmt",choices=['yolo','rfml'],default='yolo',help='Type of output')
    p.add_argument("--yolo-dims",default=[640,640],type=cs_pair,help='Image dimensions (def: %(default)s)')
    p.add_argument("--root-out",type=str,default='data_gen_database',help='Where the output should be written to (folder)')
    p.add_argument("--seed",nargs="+",type=int,default=None,help='Set seed for generation')

    return p.parse_args()

def modulation_creation(args):
    raise NotImplementedError("Not implemented yet")
def stream_creation(args):
    raise NotImplementedError("Not implemented yet")
def spectrum_creation(args):
    args.root_out = os.path.abspath(args.root_out)
    if not os.path.isdir(args.root_out):
        if not os.path.exists(args.root_out):
            os.makedirs(args.root_out)
        else:
            raise RuntimeError("The provided root path is not a directory, but it exists")
    rng = np.random.default_rng(args.seed)
    count = args.count
    resetting = args.reset_steps
    steps = 0
    spectrum = make_spectrum_gen(args,rng)
    spectrum._disable_fig_gen = True

    try:
        for idx in tqdm.tqdm(range(count)):
            if resetting and steps >= resetting:
                spectrum.close()
                spectrum = make_spectrum_gen(args,rng)
                steps = 0
            _,matrix,t,f,s,bounds = spectrum.step()
            signals_info = [x.signal for x in spectrum.signals]

            # plt.close(fig)
            # plt.show()
            steps += 1
            if args.out_fmt in ['yolo']:
                write_yolo_output(args.root_out,matrix,args.yolo_dims,
                                  args.color,
                                  'step_{0:04d}.png'.format(idx),
                                  bounds)
            else:
                print("Not configured yet")
                break;

            write_info_output(signals_info,args.root_out,idx)

        # plt.close('all')
    finally:
        spectrum.close()

def write_yolo_output(root,spectrum,desired_dims,in_color,label,bounds):
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(os.path.join(root,'images')):
        os.mkdir(os.path.join(root,'images'))
    if not os.path.exists(os.path.join(root,'bboxes')):
        os.mkdir(os.path.join(root,'bboxes'))
    if not os.path.exists(os.path.join(root,'labels')):
        os.mkdir(os.path.join(root,'labels'))
    if not os.path.exists(os.path.join(root,'signals')):
        os.mkdir(os.path.join(root,'signals'))
    # print(np.min(spectrum),np.max(spectrum))
    raw = np.uint8(np.minimum(255,np.maximum(0,np.round(spectrum*255))))
    # print(np.min(raw),np.max(raw))
    img = cv2.resize(src=raw,dsize=desired_dims, interpolation=cv2.INTER_LINEAR)
    if not in_color:
        img3 = np.stack((img,img,img))
    else:
        img3 = img
    ### img3 (channels,height,width)
    img3 = np.swapaxes(img3,0,2)
    ### img3 (width,height,channels)
    img3 = np.swapaxes(img3,0,1)
    ### img3 (height,width,channels)
    img3 = np.flipud(img3)
    img3 = cv2.cvtColor(img3,cv2.COLOR_RGB2BGR)
    filepath = os.path.join(root,'images',label)
    if os.path.exists(filepath):
        os.remove(filepath)
    cv2.imwrite(filepath,img3)
    img3 = np.flipud(img3)
    # new_fig, ax = plt.subplots(1,constrained_layout=True)
    new_fig= plt.figure(frameon=False)
    ax = plt.Axes(new_fig,[0,0,1,1])
    ax.set_axis_off()
    new_fig.add_axes(ax)
    # ax.margins(0,0)
    # ax.xaxis.set_major_locator(plt.NullLocator())
    # ax.yaxis.set_major_locator(plt.NullLocator())
    img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
    ax.imshow(img3,origin='lower')
    truth_labels = [None]*len(bounds)
    for idx,b in enumerate(bounds):
        sid,l,r,d,t = b
        left_edge = l*desired_dims[0]
        bottom_edge = d*desired_dims[1]
        xy_anchor = (left_edge,bottom_edge)     ## pixels (ish)
        width = (r-l)*desired_dims[0]           ## pixels (ish)
        height = (t-d)*desired_dims[1]          ## pixels (ish)

        xc = (l+r)/2
        yc = (2-t-d)/2
        xd = (r-l)
        yd = (t-d)

        truth_labels[idx] = f"{sid} {xc} {yc} {xd} {yd}"

        bbox = patches.Rectangle(
            # ((b[1]*640),(b[3]*640)),(b[2]-b[1])*640, (b[4]-b[3])*640,
            xy_anchor, width, height,
            linewidth=2, linestyle='--', edgecolor='blue', fill=False)
        ax.add_patch(bbox)
    ax.axis('off')
    filepath = os.path.join(root,'bboxes',label)
    if os.path.exists(filepath):
        os.remove(filepath)
    # new_fig.savefig(filepath,bbox_inches='tight')
    new_fig.savefig(filepath)
    filepath = os.path.join(root,'labels',label.replace('.png','.txt'))
    with open(filepath,'w') as fp:
        fp.write("\n".join(truth_labels))
    plt.close(new_fig)

def write_info_output(signals,root,step):
    if not os.path.exists(root):
        os.mkdir(root)
    info = {'signals':[]}
    for idx,sig in enumerate(signals):
        sig_info = {
            'label':idx,
            'modulation':sig.modulation,
            'protocol':sig.protocol,
            'duration':sig.duration,
            'carrier':sig.carrier,
            'bandwidth':sig.bandwidth,
            'sample_rate':sig.sample_rate
        }
        info['signals'].append(sig_info)
    step_file = 'step_{0:04d}.json'.format(step)
    root_dir = os.path.join(root,'signals')
    with open(f'{os.path.join(root_dir,step_file)}','w') as fp:
        json.dump(info,fp,indent=2)



def make_spectrum_gen(args,rng):
    duration = args.duration
    nfft = args.nfft
    nf = args.nf
    carrier = args.carrier
    sample_rate = args.rate
    in_color = args.color
    if args.scaling.lower() == 'minmax':
        scaling = {
            'type':'minmax',
            'min':args.vmin,
            'max':args.vmax
        }
    elif args.scaling.lower() == 'fullscale':
        scaling = {
            'type':'fullscale'
        }
    elif args.scaling.lower() == 'nfref':
        scaling = {
            'type':'nfref',
            'nf':args.nf,
            'ref':args.nfref
        }
    else:
        scaling = {'type':'native'}
    # print("SCALING?",scaling)
    seedling = rng.integers(np.iinfo(np.int64).max,size=(10,))
    spec = datagen.liquid.spectrum.Spectrum(sample_rate,carrier,duration,nf,seed=seedling)
    if scaling['type'] == 'minmax':
        spec.wf.vmin = scaling['min']
        spec.wf.vmax = scaling['max']
    elif scaling['type'] in ['fullscale','nfref']:
        spec.wf.vmin = 0.0
        spec.wf.vmax = 1.0
        if scaling['type'] == 'fullscale':
            spec.wf.spectrum_raw_scale = datagen.liquid.waterfall.fullscale_scaler
            # print('fullscale:',spec.spectrum_scaler)
        else:
            spec.wf.spectrum_raw_scale = lambda S,t,f: datagen.liquid.waterfall.fullscale_with_ref(S,t,f,scaling['nf'],scaling['ref'])
            # print('nfref:',spec.spectrum_scaler)
    if in_color:
        spec.wf.cmap = None
    spec.wf.nfft = nfft
    spec.wf.remake()
    signal_info = parse_signal_space(args.waveform_config)
    if 'signals' not in signal_info:
        raise RuntimeError("Provided config was missing 'signals' key")
    signal_names = signal_info['signals']
    signal_protos = ['_'.join([x,r'{0:03d}']) for x in signal_names]
    signal_list = []
    for idx,name in enumerate(signal_names):
        proto = signal_protos[idx]
        configs = datagen.utils.eval_random_range_config(signal_info[name],rng)
        signals = [(datagen.utils.stream_creation(c),c) for c in configs]
        if 'inflate_bbox' in signal_info[name]:
            for sig in signals:
                sig[0].inflate_bbox = signal_info[name]['inflate_bbox']
        for ind,sig in enumerate(signals):
            sig[1]['name'] = proto.format(ind)
        signal_list.extend(signals)
    for stream,config in signal_list:
        stream.sample_rate = spec.sample_rate
        stream.carrier = config['f0']
        spec.add_signal(stream,config['period'],config['relative_gain']+spec.N0,config['t0'])
    return spec

def parse_signal_space(filepath:Union[os.PathLike,str]):
    if not os.path.isfile(filepath):
        try:
            sig_info = json.loads(filepath)
        except:
            raise RuntimeError("Could not parse the path/str given")
    else:
        with open(filepath,'r') as fp:
            sig_info = json.load(fp)
    return sig_info

def main():
    args = parse_args()
    if args.target.lower() == 'modulation':
        modulation_creation(args)
    elif args.target.lower() == 'stream':
        stream_creation(args)
    elif args.target.lower() == 'spectrum':
        spectrum_creation(args)
    else:
        raise RuntimeError("Invalid target at the time.")



if __name__ == '__main__':
    main()



