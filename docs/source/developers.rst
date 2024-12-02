
Devopler Info
=============

Datagen
-------

Lingo
^^^^^

First, let's work out the terminology here to minimize what confusion we can.

#. :IQ: The complex baseband representation in time of RF sampled data.
#. :Modulation: The way information (digital: bits, analog: amplitude(floats?)) is converted to complex baseband IQ.
#. :Protocol: The combination of modulation and allowable spectral usage for transmission.
#. :SignalStream: Object responsible for maintaining a single modulation/protocol across spectrogram steps. Used interchangeably with **Signal**.
#. :Waterfall: This is the construction of how IQ from a set of **SignalStream** objects is mapped into a spectrogram, a time-frequency plot in dB relative to the noise floor.
#. :Spectrum: This refers to the object that makes spectrograms within a defined frequency range.
#. :Spectrums: The set of **Spectrum** objects that represent possble observation spaces (still in development).
#. :Scenario: The operational space for a training run within the RFRL-Gym.


Scenario Files
^^^^^^^^^^^^^^

All signals are defined within the scenario file within the *"signals"* field.
Conceptually, this is the definition of what is being transmitted.
The expectation is that each signal will have a unique name, with a dictionary structure of parameters.

.. code-block:: json

   {
      "signals":{
         "unique-name":{
            "type": "WiFi",
            "protocol": "wifi",
            "modulation": null,
            "f0": 2.412e9,
            "t0": 0.003,
            "hop": false,
            "hop_hold": 5
         }
      }
   }

* :type: A key to help filter the entities on as needed by application
* :protocol: One of \{"wifi", "zigbee", "ble", null\}
* :modulation: This is set to *null* when the protocol is desired exactly as is. If protocol is not null, this modulation will be sent as a substitute (bit data random), if protocol is null this will just transmit the modulation.
* :f0: Carrier frequency where the signal should start
* :t0: The delay into a **Spectrum** objects internal time clock.
* :hop: Should this signal use frequency hopping.
* :hop_hold: how many bursts should the signal transmit at a frequency before hopping.

With all signals defined, the observation are created in the *"spectrums"*, or what the receiver might see.
This is mostly a conceptual field at the moment, but for a static observer, a singal spectrum is all that's needed.
Again, a unique name per spectrum.

.. code-block:: json

   {
      "spectrums":{
         "bkgd":
         {
            "type": "Spectrum",
            "signal_list":[
                "bkgd_sig_001"
            ],
            "observation_bandwidth": 100e6,
            "observation_carrier": 2.45e9,
            "observation_duration": 0.05,
            "noise_power_dB": -100.0,
            "power_lower_bound": -110.0,
            "power_upper_bound": -30.0,
            "seed": null
         }
      }
   }

* :type: A key to help filter entities on as needed by application
* :signal_list: This is a list of every signal that is *possible* to show up in the spectrum range.
* :observation_bandwidth: How wide is the observation
* :observation_carrier: What is the center frequency the observer is looking at
* :observation_duration: How long should a spectrum generation step last
* :noise_power_dB: What is the gain on the noise to set a noise floor (spectrograms and gain reference this for consistency)
* :power_lower_bound: Set the minimum value for the spectrogram to represent, anything lower is clamped to this value visually.
* :power_upper_bound: Set the maximum value for the spectrogram to represent, anything higher is clamped to this value visually.
* :seed: A means of repeatablility such that the same configuration will be simular.

Setting a value space rather than a static value
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a discrete set of options, define "items"

.. code-block:: json

   "f0":{
      "items": [2412e6, 2417e6, 2422e6, 2427e6, 2432e6,
               2437e6, 2442e6, 2447e6, 2452e6, 2457e6,
               2462e6]
   }

For a uniform range define "min", "max", "scale"

.. code-block:: json

   "t0":{
      "min": 0.01,
      "max": 0.02,
      "scale": "linear"
   }

The value of "scale" is there in case it makes more sense to uniformly randoms on a log scale rather than linear.
The absence of "scale" will assume "linear".

It is also possible to define with the normal distribution with "mean", "std", "scale".

.. code-block:: json

   "t0":{
      "mean": 0.015,
      "std": 0.003,
      "scale": "linear"
   }

If "min", "max" are defined in this setup, the normal random variable will be clipped.


Protocols
^^^^^^^^^

The current list of protocols

- wifi
- zigbee
- ble
- null

Modulations
^^^^^^^^^^^

The current list of modulations available.

- psk2
- psk4
- psk8
- psk16
- psk32
- psk64
- psk128
- psk256
- dpsk2
- dpsk4
- dpsk8
- dpsk16
- dpsk32
- dpsk64
- dpsk128
- dpsk256
- ask2
- ask4
- ask8
- ask16
- ask32
- ask64
- ask128
- ask256
- qam4
- qam8
- qam16
- qam32
- qam64
- qam128
- qam256
- apsk4
- apsk8
- apsk16
- apsk32
- apsk64
- apsk128
- apsk256
- bpsk
- qpsk
- ook
- sqam32
- sqam128
- v29
- arb16opt
- arb32opt
- arb64opt
- arb128opt
- arb256opt
- arb64vt
- pi4dqpsk
