from pprint import pprint

import neo
import numpy as np
import quantities as pq

from elephant import asset


def assset_example():
    spiketrains = [
        neo.SpikeTrain(np.arange(start, start + 100, step=20) * pq.ms,
                       t_stop=200 * pq.ms)
        for start in range(4)
    ]
    asset_obj = asset.ASSET(spiketrains, verbose=False)
    sses = asset_obj.extract_synchronous_events()
    pprint(sses)


if __name__ == '__main__':
    assset_example()
