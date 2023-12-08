import numpy as np
from scipy import stats


def make_zero_symmetric_axis(ax, margin_percent):
    y_lim = (1 + margin_percent / 100) * np.max(np.abs(np.array(ax.get_ybound())))
    ax.set_ylim(-y_lim, y_lim)  # symmetric y limits
    return ax


def set_auger_energy_ticks(axis, which="x"):
    axis.tick_params(axis='both', which='major')
    min_, max_ = axis.get_xbound()
    x_lab = ['$1$', '$3$', '$10$', '$30$', '$100$', '$160$']
    x_t = [1, 3, 10, 30, 100, 160]
    ticks = [tick for tick in x_t if tick > 0.9 * min_ and tick < 1.1 * max_]
    tick_labels = [label for tick, label in zip(x_t, x_lab) if tick > 0.9 * min_ and tick < 1.1 * max_]

    if which == "x":
        try:
            axis.xaxis.set_ticks(ticks)
            axis.xaxis.set_ticklabels(tick_labels)
        except:  # noqa
            pass
    else:
        axis.yaxis.set_ticks(ticks)
        axis.yaxis.set_ticklabels(tick_labels)
    return ticks, tick_labels


def calc_angulardistance(y_tr, y_pr):
    return 180. / np.pi * np.arccos(np.clip(np.sum(y_tr * y_pr, axis=-1) / np.linalg.norm(y_pr, axis=-1) / np.linalg.norm(y_tr, axis=-1), -1, 1))


def calc_distance(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred, axis=-1)


def percentile68(x):
    ''' Calculates 68% quantile of distribution'''
    return np.percentile(x, 68)


def bootstrap_independent_2d(x, y, function, ci=68, iterations=1000):
    ''' Add bootstrapping algorithm for independent bootstrapping of two variables. ci=Confidence interval.
    e.g. for merit factor P|Fe '''
    def sample(x, y, function, iterations):
        n_x = len(x)
        n_y = len(y)
        vec = []

        for i in range(iterations):
            val_x = np.random.choice(x, n_x, replace=True)
            val_y = np.random.choice(y, n_y, replace=True)
            vec.append(function(val_x, val_y))

        return np.array(vec)

    def confidence_interval(data, ci):
        low_end = (100 - ci) / 2
        high_end = 100 - low_end
        low_bound = np.percentile(data, low_end)
        high_bound = np.percentile(data, high_end)
        return low_bound, high_bound

    if type(x) == np.ndarray:
        x = np.array(x)

    if type(y) == np.ndarray:
        y = np.array(y)

    vals = sample(x, y, function, iterations)
    interval = confidence_interval(vals, ci)
    mean = np.mean(vals)
    return mean, interval


def bootstrap(x, y=None, function=np.mean, ci=68, iterations=1000, samples=None):
    '''Bootstrapping algorithm for up to 2 variables. ci=Confidence interval. Note that seaborn regplot uses 95 as default. '''
    def sample(x, y, function, iterations):
        idx = np.arange(0, len(x))

        if samples is None:
            n = len(idx)
        else:
            n = int(samples)

        vec = []

        for i in range(iterations):
            idx_ = np.random.choice(idx, n, replace=True)
            val_x = x[idx_]

            if y is not None:
                val_y = y[idx_]
                vec.append(function(val_x, val_y))
            else:
                vec.append(function(val_x))

        return np.array(vec)

    def confidence_interval(data, ci):
        low_end = (100 - ci) / 2
        high_end = 100 - low_end
        low_bound = np.percentile(data, low_end)
        high_bound = np.percentile(data, high_end)
        return low_bound, high_bound

    if type(x) == np.ndarray:
        x = np.array(x)

    if type(y) == np.ndarray:
        y = np.array(y)

    vals = sample(x, y, function, iterations)
    interval = confidence_interval(vals, ci)
    mean = np.mean(vals)
    return mean, interval


def bin_and_btrp(z, z_bins, vals_x, vals_y=None, fn=np.mean, ci=68, n_events=False, resamples=1000):
    """ Apply binning and boostrapping of data in one go.
        Assume you have events which have property z and vals_x (and optional val_y).
        Bin property z in "z_bins" and apply transformation function "fn" for the values "vals_x" in the obtained bins.

    Args:
        z (_type_): Observable to study its dependency of fn(val_s) in bins of z_bins.
        z_bins (tuple(float, float, int)): Tuple for the definition of the bins for binning property z.
        vals_x (arr): Property to investigate using transformation function and to study as a function of z.
                         I.e., study fn(vals_x) as function of z (binned in z_bins)
        vals_y (arr, optional): Property to investigate using transformation function and to study as a function of z.
                         I.e., study fn(vals_x) as function of z (binned in z_bins). Defaults to None.
        fn (__function__, Default=np.mean): Transformation to be applied to values vals_x in bins of z. Optional transfomration can be applied to *(vals_x, vals_y)
                                     at the same time, i.e. fn(vals_x, vals_y)). Defaults to np.mean.
        ci (float, optional): Confifence interval for uncertainty estimation during bootstrapping. Defaults to 68.
        n_events (bool, optional): Return number of events per bin in z. Defaults to False.
        resamples (int, Default=1000): Number of samples used for bootstrapping. Defaults to 1000.

    Returns:
        com, btrp_vals, btrp_errs.T, edges, (n_events, if set to True)
        Bin centers of bins in z, estimated fn(vals_x), uncertainty of fn(vals_x), bin edges, number of events (if n_events = True)
    """
    if z_bins is None:
        z_bins = np.linspace(z.min(), z.max(), 100)

    com, bd_vals_x, edges = bin_to(z, vals_x, z_bins)

    if vals_y is not None:
        _, bd_vals_y, _ = bin_to(z, vals_y, z_bins)
        btrp_vals, btrp_errs = [], []

        for valx, valy in zip(bd_vals_x, bd_vals_y):
            vals, errs = bootstrap(valx, valy, fn, ci=ci, iterations=resamples)
            btrp_vals.append(vals)
            btrp_errs.append(errs)

    else:
        btrp_vals, btrp_errs = [], []

        for val in bd_vals_x:
            vals, errs = bootstrap(val, function=fn, ci=ci, iterations=resamples)
            btrp_vals.append(vals)
            btrp_errs.append(errs)

    btrp_vals = np.array(btrp_vals)
    btrp_errs = - btrp_vals[:, np.newaxis] + np.array(btrp_errs)

    if n_events:
        return com, btrp_vals, btrp_errs.T, edges, np.array([len(val) for val in bd_vals_x])
    else:
        return com, btrp_vals, btrp_errs.T, edges


def fill_empty_arr_with_0(arr, zero_element=np.array([0])):
    for i, entry in enumerate(arr):
        if entry.size == 0:
            arr[i] = zero_element
    return arr


def add_first_last_nan(x):
    x = np.insert(x, 0, np.nan, axis=0)
    x = np.append(x, [np.nan], axis=0)
    return x


def bin_to(x, values, x_bins):
    """ Function bins statistics values in x
        params:
            x: data to bin
            values: values of the data
            x_bins: binning
        return:
            bin_center: list with values of the bin centers. (first and last bins are underflow = np.nan and overflow = np.nan)
            binned_statistic: list of arrays. Length = number of bins+1 (first and last bins are underflow and overflow bins)
            edges: bin edges of the x _bins
    """
    # _, bin_edges, bin_idx = stats.binned_statistic(x, values, statistic=np.mean, bins=x_bins)  # get y_bins

    mean_x_bin, bin_edges, bin_idx = stats.binned_statistic(x, x, statistic=np.mean, bins=x_bins)  # get x_bins
    binned_stats = []
    if values[0].size > 1:
        zero_element = np.zeros((1,) + (values.shape[1:]), dtype=np.float32)
    else:
        zero_element = np.array([0])

    for bin_ in range(len(x_bins) + 1):
        binned_stats.append(values[bin_idx == bin_])
        binned_stats = fill_empty_arr_with_0(binned_stats, zero_element)

    bin_edges = add_first_last_nan(bin_edges)
    edges = get_edges(bin_edges)
    print("underflow:", sum(binned_stats[0] != 0), "overflow:", sum(binned_stats[-1] != 0))
    return add_first_last_nan(mean_x_bin), binned_stats, edges


def get_edges(bins):
    return np.array([[bins[i], bins[i + 1]] for i in range(len(bins) - 1)])
