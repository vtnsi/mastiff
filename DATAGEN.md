# Data generation tools within MASTIFF

## `dataset_generator`

This is an application installed with MASTIFF. Running it with `--help` should produce the following (or similar).

```bash
(mastiff) user@host:~$ dataset_generator --help
usage: dataset_generator [-h] [--out-fmt {yolo,rfml}] [--yolo-dims YOLO_DIMS]
                         [--root-out ROOT_OUT] [--seed SEED [SEED ...]]
                         {modulation,stream,spectrum} ...

optional arguments:
  -h, --help            show this help message and exit
  --out-fmt {yolo,rfml}
                        Type of output
  --yolo-dims YOLO_DIMS
                        Image dimensions (def: [640, 640])
  --root-out ROOT_OUT   Where the output should be written to (folder)
  --seed SEED [SEED ...]
                        Set seed for generation

target:
  {modulation,stream,spectrum}
                        target signal output [-h] [--help] for more about the
                        target
    modulation          Creating a database of modulations (after burst
                        isolation)
    stream              Creating a database of bursts from waveforms (after
                        frequency isolation)
    spectrum            Creating a spectrum of signals with no isolation
```


At this time, `--out-fmt yolo` should be the only formating in use.

`--yolo-dims` and `--root-out` specify the image size and the root directory of the dataset being created.

`--seed` is still being incorporated, but should be able to reproduce any other run of the same version with the same value.

### Modulation

Not Implemented

Intent is to create snap shots of modulations at baseband given some parameter space.


### Stream

Not Implemented

TBD

### Spectrum

The current working dataset tool. Running help after the `spectrum` keyword results in the following help.

```bash
(mastiff) user@host:~$ dataset_generator spectrum --help
usage: dataset_generator spectrum [-h] [--waveform-config WAVEFORM_CONFIG]
                                  [--nfft NFFT] [--duration DURATION]
                                  [--carrier CARRIER] [--rate RATE] [--color]
                                  [--scaling {minmax,fullscale,nfref,native}]
                                  [--vmin VMIN] [--vmax VMAX] [--nfref NFREF]
                                  [--count COUNT] [--nf NF]
                                  [--reset-steps RESET_STEPS]

optional arguments:
  -h, --help            show this help message and exit
  --waveform-config WAVEFORM_CONFIG
                        The configuration file describing the waveforms
                        desired
  --nfft NFFT           FFT size for the spectrogram
  --duration DURATION   Duration of the spectrogram for generation (s)
  --carrier CARRIER     Center frequency to use
  --rate RATE           Sample rate to use
  --color               Save the image in color, rather than grayscale
  --scaling {minmax,fullscale,nfref,native}
                        How should the image be scaled: (minmax: clip in dB by
                        the min and max values given) (fullscale: directly
                        adjust [min,max] to the [0,1] range) (nfref:
                        fullscale, but consistent noise floor value, clip
                        outside [0,1]) (native: save directly as numpy array,
                        no scaling)
  --vmin VMIN           Under minmax scale, the lower clip value (dB)
  --vmax VMAX           Under minmax scale, the upper clip value (dB)
  --nfref NFREF         Under nfref scale, the target noise floor, clips
                        outside 0,1 after noise floor adjustment (dB)
  --count COUNT         How many spectrograms should be generated?
  --nf NF               What should be the noise floor for generation
  --reset-steps RESET_STEPS
                        Number of steps before randomly regenerating specturm
                        object
```

The MASTIFF stand-in sensor expects an FFT size (`--nfft`) of 256, with a duration (`--duration`) of 0.05, or 50ms.
Additionally, the expected scaling is using the Noise Floor as a reference (`--scaling nfref`) and by default the noise floor (`--nf`) is set to -100dB.
To keep the noise floor from being clipped in the images produced, the noise floor can be adjusted to something higher than 0, and has been set with `--nfref 0.125`.
Using the values of `--vmin -110` and `--vmax -30` to act as the clipping point of the image, the values [-110, -100, -30] are mapped to [0,0.125,1] for grayscale.

Now the spectrum generator is designed to hold onto signals and if the signals themselves aren't dynamic after instance creation, the `--reset-steps` flag is there
to force the spectrum to create a new instance and allow for the random space to be explored. For example, assuming the generation of 20 images (`--count 20`) the
spectrum instance can be reset every 5 steps (`--reset-steps 5`) to provide better variance within the characteristics of the signals.

The signals themselves are defined in a JSON file to ease generation, and prevent massive commandline requests. This file is pointed to using the `--waveform-config` flag.

### Waveform Config Files

An example waveform config json file is provided in [scripts/default_spectrum_gen.json](scripts/default_spectrum_gen.json).

```json
{
    "signals":[ "unique_name" ], // the list of unique names to extract further down
    "unique_name":{
        "protocol": "wifi", // generate the waveform using the wifi protocol scripts
        "modulation": -1, // -1 == don't replace any modulation over the base protocol
        "f0": 2412e6, // starting frequency of the waveform (Hz)
        "t0": 0.01, // starting time of the waveform (sec)
        "relative_gain": 0.0, // apply gain to the signal relative to the noise floor (dB)
        "period": 0.01, // length of bursts in this waveform
        "hop": false, // should it enable hopping mode for the waveform
        "max_count": 3 // do not add more than 3 instances of this waveform to the spectrum
    }
}
```

Additionally more dynamic parameters can be used to prevent manually tuning for each run.

#### Starting frequency
```json
"f0": {"items": [2402e6, 2408e6, 2414e6]}
```
Using the `"items"` key within the defined dictionary tells the system to randomly choose one item every time a new instance is created.

#### Starting time
```json
"t0": {"min":0.001,"max":0.1,"scale":"linear"}
```
Using the `"min","max","scale"` keys within the defined dictionary tells the system to randomly choose a value within the range using a uniform random variable, the scale controls whether that's uniform in linear scale [0.001,0.1] or in `"log"` scale between [-3,-1].

## `dataset_pruner`

Adjusts the dataset after generation to 'hide' labels that can be thought of as "don't care" or "background" for the problem space. (WIP)


```bash
(mastiff) user@host:~/workspace/mastiff/src/mastiff$ dataset_pruner --help
usage: dataset_pruner [-h] [--database DATABASE] [--no-backup]
                      [--keep-protocol KEEP_PROTOCOL]
                      [--keep-modulation KEEP_MODULATION]
                      [--remove-protocol REMOVE_PROTOCOL]
                      [--remove-modulation REMOVE_MODULATION] [--restore]

optional arguments:
  -h, --help            show this help message and exit
  --database DATABASE   Path to database to prune labels in.
  --no-backup           Should the changes be backed up in the database (def:
                        backup enabled)
  --keep-protocol KEEP_PROTOCOL
                        Any signals using this protocol should be kept
                        ("ignore remove")
  --keep-modulation KEEP_MODULATION
                        Any signals using this modulation should be kept
                        ("ignore remove")
  --remove-protocol REMOVE_PROTOCOL
                        Any signals using this protocol should be removed
  --remove-modulation REMOVE_MODULATION
                        Any signals using this modulation should be removed
  --restore             Restore any backed up signal info
```

