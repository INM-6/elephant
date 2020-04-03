import neo
import numpy as np
import quantities as pq

from elephant import asset


def assset_example():
    jitter = 10 * pq.ms
    alpha = 0.9
    filter_shape = (5, 1)
    n_largest = 3
    eps = 3
    min_neighbors = 3
    stretch = 5
    binsize = 3 * pq.ms

    spiketrains = [
        neo.SpikeTrain(np.arange(start, start + 100, step=20) * pq.ms,
                       t_stop=200 * pq.ms)
        for start in range(10)
    ]

    pmat, imat, x_bins, y_bins = asset.probability_matrix_analytical(
        spiketrains,
        binsize=binsize)

    # calculate probability matrix montecarlo
    pmat_montecarlo, imat, x_bins, y_bins = \
        asset.probability_matrix_montecarlo(
            spiketrains,
            jitter=jitter,
            binsize=binsize,
            n_surrogates=10)

    # calculate joint probability matrix
    jmat = asset.joint_probability_matrix(pmat,
                                          filter_shape=filter_shape,
                                          n_largest=n_largest)

    # calculate mask matrix and cluster matrix
    mmat = asset.mask_matrices([pmat, jmat], [alpha, alpha])
    cmat = asset.cluster_matrix_entries(mmat,
                                        eps=eps,
                                        min_neighbors=min_neighbors,
                                        stretch=stretch)

    # extract sses and test them
    sses = asset.extract_synchronous_events(spiketrains, binsize, cmat)
    from pprint import pprint
    pprint(sses, width=70)
    print(len(sses))
    # print(sses)


if __name__ == '__main__':
    assset_example()
