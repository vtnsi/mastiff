

import os
import argparse
import numpy as np
import json
import glob
import re

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--database',default=None,type=str,help=("Path to database to prune labels in."))
    p.add_argument('--no-backup',dest='backup',action='store_false',help=("Should the changes be backed up in the database (def: backup enabled)"))
    p.add_argument('--keep-protocol',default=None,action='append',type=str,help=('Any signals using this protocol should be kept ("ignore remove")'))
    p.add_argument('--keep-modulation',default=None,action='append',type=str,help=('Any signals using this modulation should be kept ("ignore remove")'))
    p.add_argument('--remove-protocol',default=None,action='append',type=str,help=('Any signals using this protocol should be removed'))
    p.add_argument('--remove-modulation',default=None,action='append',type=str,help=('Any signals using this modulation should be removed'))
    p.add_argument('--restore',action='store_true',help=("Restore any backed up signal info"))

    args = p.parse_args()
    if not isinstance(args.keep_protocol,list):
        args.keep_protocol = [] if args.keep_protocol is None else [args.keep_protocol]
    if not isinstance(args.keep_modulation,list):
        args.keep_modulation = [] if args.keep_modulation is None else [args.keep_modulation]
    if not isinstance(args.remove_protocol,list):
        args.remove_protocol = [] if args.remove_protocol is None else [args.remove_protocol]
    if not isinstance(args.remove_modulation,list):
        args.remove_modulation = [] if args.remove_modulation is None else [args.remove_modulation]
    return args


def find_removable_protocols(root,remove_proto):
    step_files = sorted(glob.glob(os.path.join(root,'signals','step*.json')))
    purgable = []
    for sf in step_files:
        filename = os.path.basename(sf)
        step_n = [int(x) for x in re.findall(r'\d+',filename)][0]
        with open(sf,'r') as fp:
            step_info = json.load(fp)
        for sig_info in step_info['signals']:
            if sig_info['protocol'] in remove_proto:
                purgable.append([step_n,sig_info['label'],sig_info])
    return purgable

def find_removable_modulations(root,remove_mod):
    step_files = sorted(glob.glob(os.path.join(root,'signals','step*.json')))
    purgable = []
    for sf in step_files:
        filename = os.path.basename(sf)
        step_n = [int(x) for x in re.findall(r'\d+',filename)][0]
        with open(sf,'r') as fp:
            step_info = json.load(fp)
        for sig_info in step_info['signals']:
            if sig_info['modulation'] in remove_mod:
                purgable.append([step_n,sig_info['label'],sig_info])
    return purgable

def keep_protocols(potential_purging,keep_proto):
    nix = dict()
    for idx,purging in enumerate(potential_purging):
        step,label,meta = purging
        if meta['protocol'] in keep_proto:
            key = (step,label)
            if key in nix:
                nix[key].append(idx)
            else:
                nix[key] = [idx]
    all_nix = []
    for val in nix.values():
        all_nix.extend(val)
    all_nix = sorted(list(set(all_nix)))
    for idx in reversed(all_nix):
        del potential_purging[idx]
    return potential_purging

def keep_modulations(potential_purging,keep_mod):
    nix = dict()
    for idx,purging in enumerate(potential_purging):
        step,label,meta = purging
        if meta['modulation'] in keep_mod:
            key = (step,label)
            if key in nix:
                nix[key].append(idx)
            else:
                nix[key] = [idx]
    all_nix = []
    for val in nix.values():
        all_nix.extend(val)
    all_nix = sorted(list(set(all_nix)))
    for idx in reversed(all_nix):
        del potential_purging[idx]
    return potential_purging

def run(database=None,keep_proto=[],keep_mod=[],remove_proto=[],remove_mod=[],backup=True,restore=False):
    if database is None:
        try:
            args = parse_args()
        except:
            raise RuntimeError("Could not find any args for an empty database request")
        database = args.database
        keep_proto = args.keep_protocol
        keep_mod = args.keep_modulation
        remove_proto = args.remove_protocol
        remove_mod = args.remove_modulation
        backup = args.backup
        restore = args.restore

    if restore:
        if not os.path.exists(os.path.join(database,'backup')):
            raise RuntimeError(f"Backups not found in database ({database})")
        
        label_bak = sorted(glob.glob(os.path.join(database,'backup','step_*.txt')))
        remap_bak = sorted(glob.glob(os.path.join(database,'backup','step_*_lbl_map.json')))

        for bak,rmp in zip(label_bak,remap_bak):
            with open(bak,'r') as fp:
                rmd = fp.readlines()
            with open(rmp,'r') as fp:
                rem = json.load(fp)
            
            invert_map = dict()
            for k,v in rem.items():
                invert_map[v] = k
            with open(os.path.join(database,'labels',os.path.basename(bak)),'r') as fp:
                kept = fp.readlines()
            for idx,line in enumerate(kept):
                items = line.split(' ')
                items[0] = invert_map[items[0]]
                line = ' '.join(items)
                kept[idx] = line

            all_again = rmd+kept
            all_again = sorted(all_again,key=lambda x: x.split(" ")[0])
            with open(os.path.join(database,'labels',os.path.basename(bak)),'w') as fp:
                fp.write(''.join(all_again))

            os.remove(bak)
            os.remove(rmp)
        try:
            os.rmdir(os.path.join(database,'backup'))
        except:
            print("Unable to remove backup directory from database")

    else:
        # print(database,keep_proto,keep_mod,remove_proto,remove_mod,backup,restore)
        if backup:
            if not os.path.exists(os.path.join(database,'backup')):
                os.mkdir(os.path.join(database,'backup'))

        if not os.path.exists(database):
            raise RuntimeError(f"Database : {database} not found")
        if not os.path.exists(os.path.join(database,'signals')):
            raise RuntimeError(f"Database({database}) does not have the expected 'signals' directory within")
        rm_proto = find_removable_protocols(database,remove_proto)
        rm_mod = find_removable_modulations(database,remove_mod)
        reduced_rm = keep_modulations(keep_protocols(rm_proto+rm_mod,keep_proto),keep_mod)

        reduced_rm = sorted(reduced_rm,key=lambda x: (x[0],x[1]))
        shortcut = dict()
        for step,label,meta in reduced_rm:
            if step not in shortcut:
                shortcut[step] = [label]
            else:
                shortcut[step].append(label)
        for step,labels in shortcut.items():
            with open(os.path.join(database,'labels','step_{0:04d}.txt'.format(step)),'r') as fp:
                lines = fp.readlines()
            items = [x.split(" ") for x in lines]
            sigs = [x for x in items if int(x[0]) not in labels]
            label_remap = dict()
            new_idx = 0
            for lbl in sigs:
                if lbl[0] not in label_remap:
                    label_remap[lbl[0]] = str(new_idx)
                    new_idx += 1
                lbl[0] = label_remap[lbl[0]]
            sigs = [' '.join(x) for x in sigs]
            if backup:
                rmd = [' '.join(x) for x in items if int(x[0]) in labels]
                with open(os.path.join(database,'backup','step_{0:04d}.txt'.format(step)),'w') as fp:
                    fp.write(''.join(rmd))
                with open(os.path.join(database,'backup','step_{0:04d}_lbl_map.json'.format(step)),'w') as fp:
                    json.dump(label_remap,fp)
            with open(os.path.join(database,'labels','step_{0:04d}.txt'.format(step)),'w') as fp:
                fp.write(''.join(sigs))


def main():
    args = parse_args()
    run(args.database,
        args.keep_protocol,args.keep_modulation,
        args.remove_protocol,args.remove_modulation,
        args.backup,args.restore)

if __name__ == '__main__':
    main()
