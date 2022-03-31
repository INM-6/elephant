from copy import deepcopy
from .base import AnalysisObject
from collections import namedtuple


_PSDObjectTuple = namedtuple('PSDObject', 'freqs psd')


class PSDObject(AnalysisObject, _PSDObjectTuple):
    """
    Class to store outputs of the `elephant.spectral.welch_psd` function.

    Parameters
    ----------
    freqs : np.ndarray or pq.Quantity
        Values of the frequency array.
    psd : np.ndarray or pq.Quantity
        Values of the power spectral density array.
    computation_method : str
        Name of the Python function used for the computation.
        If the object is used inside a wrapper, this should be the actual
        function used (e.g., from `scipy`).
    computation_params : dict, optional
        Dictionary with the parameters used for `computation_method`.
        Default: None
    kwargs : dict
        Additional parameters to store. Each key in the dictionary will be
        set as an object attribute.

    See Also
    --------
    elephant.spectral.welch_psd

    """

    def __new__(cls, freqs, psd, computation_method, computation_params=None,
                **kwargs):
        return super().__new__(cls, freqs, psd)

    def __init__(self, freqs, psd, computation_method,
                 computation_params=None, **kwargs):
        super().__init__(self, freqs, psd)
        self._computation_method = computation_method
        self._computation_params = None
        if computation_params is not None:
            self._computation_params = deepcopy(computation_params)
        self._store_kwargs(kwargs)

    @property
    def computation_method(self):
        return self._computation_method

    @property
    def computation_parameters(self):
        return self._computation_params

    @property
    def frequency_resolution(self):
        return self.freqs[1] - self.freqs[0]
