import numpy as np
import quantities as pq

from elephant.objects.examples.utils import get_analog_signal
from elephant.objects import PSDObject
from elephant.spectral import welch_psd

import elephant.objects as objects

import matplotlib.pyplot as plt


def main():
    objects.USE_ANALYSIS_OBJECTS = True

    signal = get_analog_signal(frequency=30*pq.Hz, n_channels=5, t_stop=4*pq.s,
                               sampling_rate=30000*pq.Hz, amplitude=50*pq.uV)

    obj = welch_psd(signal, frequency_resolution=0.25*pq.Hz)
    if isinstance(obj, PSDObject):
        print(obj.computation_method, obj.computation_parameters,
              obj.elephant_params)
        print(obj.frequency_resolution)
    else:
        print("Old function was used, returning a tuple.")

    freqs, psd = obj

    # Using tuple as with returns of the original function
    plot_freqs = np.where(freqs < 100)
    plt.plot(freqs[plot_freqs], psd[0, plot_freqs].flat)
    plt.show()


if __name__ == "__main__":
    main()
