from copy import deepcopy

import numpy as np
import quantities as pq


class MeanOnline(object):
    def __init__(self, batch_mode=False):
        self.mean = None
        self.count = 0
        self.units = None
        self.batch_mode = batch_mode

    def update(self, new_val):
        units = None
        if isinstance(new_val, pq.Quantity):
            units = new_val.units
            new_val = new_val.magnitude
        if self.batch_mode:
            batch_size = new_val.shape[0]
            new_val_sum = new_val.sum(axis=0)
        else:
            batch_size = 1
            new_val_sum = new_val
        self.count += batch_size
        if self.mean is None:
            self.mean = deepcopy(new_val_sum / batch_size)
            self.units = units
        else:
            if units != self.units:
                raise ValueError("Each batch must have the same units.")
            self.mean += (new_val_sum - self.mean * batch_size) / self.count

    def as_units(self, val):
        if self.units is None:
            return val
        return pq.Quantity(val, units=self.units, copy=False)

    def get_mean(self):
        return self.as_units(deepcopy(self.mean))

    def reset(self):
        self.mean = None
        self.count = 0
        self.units = None


class VarianceOnline(MeanOnline):
    def __init__(self, batch_mode=False):
        super(VarianceOnline, self).__init__(batch_mode=batch_mode)
        self.variance_sum = 0.

    def update(self, new_val):
        units = None
        if isinstance(new_val, pq.Quantity):
            units = new_val.units
            new_val = new_val.magnitude
        if self.mean is None:
            self.mean = 0.
            self.variance_sum = 0.
            self.units = units
        elif units != self.units:
            raise ValueError("Each batch must have the same units.")
        delta_var = new_val - self.mean
        if self.batch_mode:
            batch_size = new_val.shape[0]
            self.count += batch_size
            delta_mean = new_val.sum(axis=0) - self.mean * batch_size
            self.mean += delta_mean / self.count
            delta_var *= new_val - self.mean
            delta_var = delta_var.sum(axis=0)
        else:
            self.count += 1
            self.mean += delta_var / self.count
            delta_var *= new_val - self.mean
        self.variance_sum += delta_var

    def get_mean_std(self, unbiased=False):
        if self.mean is None:
            return None, None
        if self.count > 1:
            count = self.count - 1 if unbiased else self.count
            std = np.sqrt(self.variance_sum / count)
        else:
            # with 1 update biased & unbiased sample variance is zero
            std = 0.
        mean = self.as_units(deepcopy(self.mean))
        std = self.as_units(std)
        return mean, std

    def reset(self):
        super(VarianceOnline, self).reset()
        self.variance_sum = 0.


class CovarianceOnline(object):
    def __init__(self, batch_mode=False):
        self.batch_mode = batch_mode
        self.var_x = VarianceOnline(batch_mode=batch_mode)
        self.var_y = VarianceOnline(batch_mode=batch_mode)
        self.units = None
        self.covariance_sum = 0.
        self.count = 0

    def update(self, new_val_pair):
        units = None
        if isinstance(new_val_pair, pq.Quantity):
            units = new_val_pair.units
            new_val_pair = new_val_pair.magnitude
        if self.count == 0:
            self.var_x.mean = 0.
            self.var_y.mean = 0.
            self.covariance_sum = 0.
            self.units = units
        elif units != self.units:
            raise ValueError("Each batch must have the same units.")
        if self.batch_mode:
            self.var_x.update(new_val_pair[0])
            self.var_y.update(new_val_pair[1])
            delta_var_x = new_val_pair[0] - self.var_x.mean
            delta_var_y = new_val_pair[1] - self.var_y.mean
            delta_covar = delta_var_x * delta_var_y
            batch_size = len(new_val_pair[0])
            self.count += batch_size
            delta_covar = delta_covar.sum(axis=0)
            self.covariance_sum += delta_covar
        else:
            delta_var_x = new_val_pair[0] - self.var_x.mean
            delta_var_y = new_val_pair[1] - self.var_y.mean
            delta_covar = delta_var_x * delta_var_y
            self.var_x.update(new_val_pair[0])
            self.var_y.update(new_val_pair[1])
            self.count += 1
            self.covariance_sum += ((self.count - 1) / self.count) * delta_covar

    def get_cov(self, unbiased=False):
        if self.var_x.mean is None and self.var_y.mean is None:
            return None
        if self.count > 1:
            count = self.count - 1 if unbiased else self.count
            cov = self.covariance_sum / count
        else:
            cov = 0.
        return cov

    def reset(self):
        self.var_x.reset()
        self.var_y.reset()
        self.units = None
        self.covariance_sum = 0.
        self.count = 0
