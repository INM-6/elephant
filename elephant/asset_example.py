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
    imat = asset_obj.intersection_matrix()
    pmat = asset_obj.probability_matrix_montecarlo(imat)
    jmat = asset_obj.joint_probability_matrix(pmat, filter_shape=(5, 1))
    mmat = asset_obj.mask_matrices([pmat, jmat], thresholds=0.9999)
    cmat = asset_obj.cluster_matrix_entries(mmat)

    sses = asset_obj.extract_synchronous_events(cmat)
    pprint(sses)


if __name__ == '__main__':
    assset_example()
