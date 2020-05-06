import numba as nb
import numpy as np

__all__ = [
    'sstd', 'nb_sstd', '_qcomb_setup', '_qcomb_wrapup',
    '_qcomb_med_sigclip_1d',
    'COMB_FUNCS'
]


def sstd(a, **kwargs):
    ''' Sample standard deviation function
    '''
    return np.std(a, ddof=1, **kwargs)


@nb.njit(fastmath=True)
def nb_sstd(a, ddof=1):
    ''' Sample standard deviation function for 1-D array
    '''
    n = a.size
    return np.sqrt(n/(n-ddof))*np.std(a)


def _qcomb_setup(arr, mask, sigma, sigma_upper, sigma_lower, maxiters,
                 combine, reject, full, dtype=float):
    # * Set mask
    if mask is None:
        mask = np.zeros(arr.shape, dtype=bool)
    else:
        mask = np.array(mask, dtype=bool)
        if arr.shape != mask.shape:
            raise ValueError(
                "mask must have the identical shape as array."
                f"Shapes: mask = {mask.shape}, array = {arr.shape}"
            )

    # * Set sigma
    if sigma_upper is None:
        sigma_upper = sigma
    if sigma_lower is None:
        sigma_lower = sigma

    # * Set maxiters
    if maxiters < 1:
        # Define np.inf so that output lo/hi are +-inf values.
        sigma_upper = np.inf
        sigma_lower = np.inf

    # * Set reject method
    if reject in ['sig', 'sigma', 'sigclip', 'sigmaclip', 'sigma clip']:
        reject = 'sigclip'
    elif reject in ['minmax', 'mm']:
        reject = 'minmax'
    elif reject in ['pclip', 'pc']:
        reject = 'pclip'
    else:
        raise ValueError(f"reject method of {reject} not understood.")

    # * Set output arrays (essential for 2-D or higher)
    outsh = arr.shape[1:]  # shape of output (combined) array
    msksh = arr.shape      # shape of mask (rejection) array

    if combine in ['med', 'medi', 'median']:
        combine = 'median'
        if full:
            out_comb  = np.atleast_1d(np.zeros(outsh, dtype=dtype))
            out_std   = np.atleast_1d(np.zeros(outsh, dtype=dtype))
            out_lo    = np.atleast_1d(np.zeros(outsh, dtype=dtype))
            out_hi    = np.atleast_1d(np.zeros(outsh, dtype=dtype))
            out_niter = np.atleast_1d(np.zeros(outsh, dtype=int  ))
            out_nrej  = np.atleast_1d(np.zeros(outsh, dtype=int  ))
            out_mask  = np.atleast_1d(np.zeros(msksh, dtype=bool ))
        else:
            out_comb  = np.atleast_1d(np.zeros(outsh, dtype=dtype))
            out_std   = None
            out_lo    = None
            out_hi    = None
            out_niter = None
            out_nrej  = None
            out_mask  = None
    elif combine in ['mean', 'avg', 'average']:
        combine = 'mean'
        if full:
            out_comb  = np.atleast_1d(np.zeros(outsh, dtype=dtype))
            out_std   = np.atleast_1d(np.zeros(outsh, dtype=dtype))
            out_lo    = np.atleast_1d(np.zeros(outsh, dtype=dtype))
            out_hi    = np.atleast_1d(np.zeros(outsh, dtype=dtype))
            out_niter = np.atleast_1d(np.zeros(outsh, dtype=int  ))
            out_nrej  = np.atleast_1d(np.zeros(outsh, dtype=int  ))
            out_mask  = np.atleast_1d(np.zeros(msksh, dtype=bool ))
        else:
            out_comb = np.atleast_1d(np.zeros(outsh, dtype=dtype))
            out_std   = None
            out_lo    = None
            out_hi    = None
            out_niter = None
            out_nrej  = None
            out_mask  = None

    outs = dict(
        out_comb=out_comb,
        out_std=out_std,
        out_lo=out_lo,
        out_hi=out_hi,
        out_niter=out_niter,
        out_nrej=out_nrej,
        out_mask=out_mask
    )

    return mask, sigma_upper, sigma_lower, maxiters, combine, reject, outs


def _qcomb_wrapup(res):
    # for N-D full results:
    if isinstance(res, dict):
        nres = dict()
        for k, v in res.items():
            nres[k.split('out_')[-1]] = v
        return nres

    # for simple (``full=False```) results:
    else:
        return res


@nb.njit(fastmath=True)
def _qcomb_med_sigclip_1d(_arr1d, mask, sigma_upper=3, sigma_lower=3,
                          maxiters=5, nkeep=3):
    ''' Median sigma-clip combine for 1-D array.
    Note
    ----
    This is made for internal use. Maybe there's no need to use this
    function except for development purposes.

    Parameters
    ----------
    _arr1d : 1-D array
        The data to be processed.

    mask : 1-D boolean array
        The mask to be used to ignore values in ``_arr1d``.

    sigma_upper, sigma_lower : float, optional.
        The upper/lower sigma values to reject data.

    maxiters : int, optional.
        The maximum iteration numbers for sigma-clipping. If ``maxiters
        = 0``, no sigma-clipping will occur, but the outputs are
        calculated for homogeneous return values.

    nkeep : int, optional.
        The minimum number of pixels to proceed sigma-clipping. If the
        remaining pixels are fewer than this value, no further iteration
        will be made. At least 3 is recommended as the sigma (sample
        standard deviation) is defined only if we have more than 3 data.

    Returns
    -------
    cen : float
        The resulting central value (median) after sigma-clipping.

    std : float
        The final sigma (sample standard deviation) of the remaining
        data after sigma-clipping.

    lo/hi : float
        The lower/upper boundaries from the final sigma-clipping (only
        data of ``lo < x < hi`` are survived).

    i : int
        The number of iterations when the sigma-clipping is halted. If
        no sigma-clipping was done, this is ``0``.

    nrej : int
        The number of rejected data.

    mask_new : 1-D boolean array
        The boolean array (same shape as ``_arr1d``) which is the final
        mask after the sigma-clipping, propagated from the initial
        ``mask`` provided by the user.
    '''
    n_original = _arr1d.size
    n_old = _arr1d.size

    # 0-th iteration
    i = 0
    mask_old = mask.copy()
    _a = _arr1d[~mask]
    cen = np.median(_a)
    std = nb_sstd(_a)
    lo = cen - sigma_lower*std
    hi = cen + sigma_upper*std
    n_new = n_original - np.sum(mask)

    for i in range(maxiters):
        mask_new = mask_old | (_arr1d < lo) | (hi < _arr1d)
        _a = _arr1d[~mask_new]
        n_new = n_original - np.sum(mask_new)

        if (_a.shape[0] < max(3, nkeep)  # if too few pixels left
                or (n_new == n_old)):      # or if no more rejection
            mask_new = mask_old  # revert to previous mask
            break

        i += 1  # i-th iteration
        cen = np.median(_a)
        std = nb_sstd(_a)
        lo = cen - sigma_lower*std
        hi = cen + sigma_upper*std
        n_old = n_new
        mask_old = mask_new

    return (cen, std, lo, hi, i, n_original - n_new, mask_new)


@nb.njit(fastmath=True)
def _qcomb_mean_sigclip_1d(_arr1d, mask, sigma_upper=3, sigma_lower=3,
                           maxiters=5, nkeep=3):
    ''' Mean sigma-clip combine for 1-D array.
    Note
    ----
    This is made for internal use. Maybe there's no need to use this
    function except for development purposes.

    Parameters
    ----------
    _arr1d : 1-D array
        The data to be processed.

    mask : 1-D boolean array
        The mask to be used to ignore values in ``_arr1d``.

    sigma_upper, sigma_lower : float, optional.
        The upper/lower sigma values to reject data.

    maxiters : int, optional.
        The maximum iteration numbers for sigma-clipping. If ``maxiters
        = 0``, no sigma-clipping will occur, but the outputs are
        calculated for homogeneous retun values.

    nkeep : int, optional.
        The minimum number of pixels to proceed sigma-clipping. If the
        remaining pixels are fewer than this value, no further iteration
        will be made. At least 3 is recommended as the sigma (sample
        standard deviation) is defined only if we have more than 3 data.

    Returns
    -------
    cen : float
        The resulting central value (mean) after sigma-clipping.

    std : float
        The final sigma (sample standard deviation) of the remaining
        data after sigma-clipping.

    lo/hi : float
        The lower/upper boundaries from the final sigma-clipping (only
        data of ``lo < x < hi`` are survived).

    i : int
        The number of iterations when the sigma-clipping is halted. If
        no sigma-clipping was done, this is ``0``.

    nrej : int
        The number of rejected data.

    mask_new : 1-D boolean array
        The boolean array (same shape as ``_arr1d``) which is the final
        mask after the sigma-clipping, propagated from the initial
        ``mask`` provided by the user.
    '''
    n_original = _arr1d.size
    n_old = _arr1d.size

    # 0-th iteration
    i = 0
    mask_old = mask.copy()
    _a = _arr1d[~mask]
    cen = np.mean(_a)
    std = nb_sstd(_a)
    lo = cen - sigma_lower*std
    hi = cen + sigma_upper*std
    n_new = n_original - np.sum(mask)

    for i in range(maxiters):
        mask_new = mask_old | (_arr1d < lo) | (hi < _arr1d)
        _a = _arr1d[~mask_new]
        n_new = n_original - np.sum(mask_new)

        if (_a.shape[0] < max(3, nkeep)  # if too few pixels left
                or (n_new == n_old)):      # or if no more rejection
            mask_new = mask_old  # revert to previous mask
            break

        i += 1  # i-th iteration
        cen = np.mean(_a)
        std = nb_sstd(_a)
        lo = cen - sigma_lower*std
        hi = cen + sigma_upper*std
        n_old = n_new
        mask_old = mask_new

    return (cen, std, lo, hi, i, n_original - n_new, mask_new)


@nb.njit(fastmath=True, parallel=True)
def _qcomb_med_sigclip_full(_arr, mask, out_comb, out_std, out_lo, out_hi,
                            out_niter, out_nrej, out_mask,
                            sigma_upper=3, sigma_lower=3, maxiters=5, nkeep=3):
    nim = _arr.shape[0]
    stepsize = _arr.strides[0]//_arr.itemsize
    # offsets = stepsize * np.arange(_arr.shape[0])
    # offsets is used as
    #   _a = _arr.flat[offsets + idx]
    # and/or
    #   out_mask.flat[offsets + idx] = res[6]
    # rather than using complicated
    #   _a = np.zeros(nim)
    #   for i in range(nim):
    #       k = i*stepsize + idx
    # in the for loop below in pure python (and numpy) case, but it
    # cannot be used by numba 0.50.0 YPB (2020-05-06 19:32:37 (KST:
    # GMT+09:00))

    for idx in nb.prange(stepsize):
        _a = np.zeros(nim)
        _m = np.zeros(nim, dtype=mask.dtype)
        for i in range(nim):
            k = i*stepsize + idx
            _a[i] = _arr.flat[k]
            _m[i] = mask.flat[k]

        res = _qcomb_med_sigclip_1d(
            _arr1d=_a,  # all values along x0 at ...
            mask=_m,  # ... (x1, x2, ...) = idx
            sigma_upper=sigma_upper,
            sigma_lower=sigma_lower,
            maxiters=maxiters,
            nkeep=nkeep
        )

        out_comb.flat[idx] = res[0]
        out_std.flat[idx] = res[1]
        out_lo.flat[idx] = res[2]
        out_hi.flat[idx] = res[3]
        out_niter.flat[idx] = res[4]
        out_nrej.flat[idx] = res[5]
        for i in range(nim):
            k = i*stepsize + idx
            out_mask.flat[k] = res[6][i]


@nb.njit(fastmath=True, parallel=True)
def _qcomb_med_sigclip_simple(_arr, mask, out_comb,
                              sigma_upper=3, sigma_lower=3, maxiters=5,
                              nkeep=3):
    nim = _arr.shape[0]
    stepsize = _arr.strides[0]//_arr.itemsize
    # offsets = stepsize * np.arange(_arr.shape[0])
    # offsets is used as
    #   _a = _arr.flat[offsets + idx]
    # rather than using complicated
    #   _a = np.zeros(nim)
    #   for i in range(nim):
    #       k = i*stepsize + idx
    # in the for loop below in pure python (and numpy) case, but it
    # cannot be used by numba 0.50.0 YPB (2020-05-06 19:32:37 (KST:
    # GMT+09:00))

    for idx in nb.prange(stepsize):
        _a = np.zeros(nim)
        _m = np.zeros(nim, dtype=mask.dtype)
        for i in range(nim):
            k = i*stepsize + idx
            _a[i] = _arr.flat[k]
            _m[i] = mask.flat[k]

        res = _qcomb_med_sigclip_1d(
            _arr1d=_a,  # all values along x0 at ...
            mask=_m,  # ... (x1, x2, ...) = idx
            sigma_upper=sigma_upper,
            sigma_lower=sigma_lower,
            maxiters=maxiters,
            nkeep=nkeep
        )

        out_comb.flat[idx] = res[0]


@nb.njit(fastmath=True, parallel=True)
def _qcomb_mean_sigclip_full(_arr, mask, out_comb, out_std, out_lo, out_hi,
                             out_niter, out_nrej, out_mask,
                             sigma_upper=3, sigma_lower=3, maxiters=5,
                             nkeep=3):
    nim = _arr.shape[0]
    stepsize = _arr.strides[0]//_arr.itemsize
    # offsets = stepsize * np.arange(_arr.shape[0])
    # offsets is used as
    #   _a = _arr.flat[offsets + idx]
    # and/or
    #   out_mask.flat[offsets + idx] = res[6]
    # rather than using complicated
    #   _a = np.zeros(nim)
    #   for i in range(nim):
    #       k = i*stepsize + idx
    # in the for loop below in pure python (and numpy) case, but it
    # cannot be used by numba 0.50.0 YPB (2020-05-06 19:32:37 (KST:
    # GMT+09:00))

    for idx in nb.prange(stepsize):
        _a = np.zeros(nim)
        _m = np.zeros(nim, dtype=mask.dtype)
        for i in range(nim):
            k = i*stepsize + idx
            _a[i] = _arr.flat[k]
            _m[i] = mask.flat[k]

        res = _qcomb_mean_sigclip_1d(
            _arr1d=_a,  # all values along x0 at ...
            mask=_m,  # ... (x1, x2, ...) = idx
            sigma_upper=sigma_upper,
            sigma_lower=sigma_lower,
            maxiters=maxiters,
            nkeep=nkeep
        )

        out_comb.flat[idx] = res[0]
        out_std.flat[idx] = res[1]
        out_lo.flat[idx] = res[2]
        out_hi.flat[idx] = res[3]
        out_niter.flat[idx] = res[4]
        out_nrej.flat[idx] = res[5]
        for i in range(nim):
            k = i*stepsize + idx
            out_mask.flat[k] = res[6][i]


@nb.njit(fastmath=True, parallel=True)
def _qcomb_mean_sigclip_simple(_arr, mask, out_comb,
                               sigma_upper=3, sigma_lower=3, maxiters=5,
                               nkeep=3):
    nim = _arr.shape[0]
    stepsize = _arr.strides[0]//_arr.itemsize
    # offsets = stepsize * np.arange(_arr.shape[0])
    # offsets is used as
    #   _a = _arr.flat[offsets + idx]
    # rather than using complicated
    #   _a = np.zeros(nim)
    #   for i in range(nim):
    #       k = i*stepsize + idx
    # in the for loop below in pure python (and numpy) case, but it
    # cannot be used by numba 0.50.0 YPB (2020-05-06 19:32:37 (KST:
    # GMT+09:00))

    for idx in nb.prange(stepsize):
        _a = np.zeros(nim)
        _m = np.zeros(nim, dtype=mask.dtype)
        for i in range(nim):
            k = i*stepsize + idx
            _a[i] = _arr.flat[k]
            _m[i] = mask.flat[k]

        res = _qcomb_mean_sigclip_1d(
            _arr1d=_a,  # all values along x0 at ...
            mask=_m,  # ... (x1, x2, ...) = idx
            sigma_upper=sigma_upper,
            sigma_lower=sigma_lower,
            maxiters=maxiters,
            nkeep=nkeep
        )

        out_comb.flat[idx] = res[0]


COMB_FUNCS = {
    ('median' , 'sigclip', True ): _qcomb_med_sigclip_full   ,
    ('median' , 'sigclip', False): _qcomb_med_sigclip_simple ,
    ('mean'   , 'sigclip', True ): _qcomb_mean_sigclip_full  ,
    ('mean'   , 'sigclip', False): _qcomb_mean_sigclip_simple
}


"""
def qcomb_nd(arr, mask=None, sigma=3, sigma_upper=None, sigma_lower=None,
             maxiters=5, combine='med', full=False):
    ''' Median combine along axis=0
    Note
    ----
    ``axis=0`` is simple and reasonable because if you do ``lines = []``
    and ``lines.append(line)``, the ``lines`` will need to be combined
    along ``axis=0``.
    '''
    mask, sigma_lower, sigma_upper, maxiters, combine, outs = _qcomb_setup(
        arr=arr,
        mask=mask,
        sigma=sigma,
        sigma_upper=sigma_upper,
        sigma_lower=sigma_lower,
        maxiters=maxiters,
        combine=combine,
        full=full
    )
    if full:
        res = _qcomb(_arr=arr,
                     **outs,
                     mask=mask,
                     sigma_upper=sigma_upper,
                     sigma_lower=sigma_lower,
                     maxiters=maxiters)
        res = _qcomb_wrapup(outs)
    else:
        res = _qcomb(_arr=arr,
                     out_comb=outs['out_comb'],
                     mask=mask,
                     sigma_upper=sigma_upper,
                     sigma_lower=sigma_lower,
                     maxiters=maxiters)
        res = outs['out_comb']
    return res


# ----------------------------------------------------------------------------


@nb.njit(fastmath=True)
def _qcomb_med_1d_full(_arr, mask, sigma_upper=3, sigma_lower=3, maxiters=5):
    n_original = _arr.size
    n_old = _arr.size
    for i in range(maxiters + 1):
        cen = np.np.median(_arr[~mask])
        std = nb_sstd(_arr[~mask])
        lo = cen - sigma_lower*std
        hi = cen + sigma_upper*std
        mask = mask | (_arr < lo) | (hi < _arr)
        n_new = n_original - np.sum(mask)
        if n_new == n_old:  # if no more rejection
            break
        n_old = n_new

    return (cen, std, lo, hi, i, n_original - n_new, mask)


@nb.njit(fastmath=True)
def _qcomb_med_1d_simple(_arr, mask, sigma_upper=3, sigma_lower=3, maxiters=5):
    n_original = _arr.size
    n_old = _arr.size
    for i in range(maxiters + 1):
        cen = np.np.median(_arr[~mask])
        std = nb_sstd(_arr[~mask])
        lo = cen - sigma_lower*std
        hi = cen + sigma_upper*std
        mask = mask | (_arr < lo) | (hi < _arr)
        n_new = n_original - np.sum(mask)
        if n_new == n_old:  # if no more rejection
            break
        n_old = n_new

    return cen


@nb.njit(fastmath=True)
def _qcomb_mean_1d_full(_arr, mask=None, sigma_upper=3, sigma_lower=3, maxiters=5):
    n_original = _arr.size
    n_old = _arr.size
    for i in range(maxiters + 1):
        cen = np.mean(_arr[~mask])
        std = nb_sstd(_arr[~mask])
        lo = cen - sigma_lower*std
        hi = cen + sigma_upper*std
        mask = mask | (_arr < lo) | (hi < _arr)
        n_new = n_original - np.sum(mask)
        if n_new == n_old:  # if no more rejection
            break
        n_old = n_new

    return (cen, std, lo, hi, i, n_original - n_new, mask)


@nb.njit(fastmath=True)
def _qcomb_mean_1d_simple(_arr, mask, sigma_upper=3, sigma_lower=3, maxiters=5):
    n_original = _arr.size
    n_old = _arr.size
    for i in range(maxiters + 1):
        cen = np.mean(_arr[~mask])
        std = nb_sstd(_arr[~mask])
        lo = cen - sigma_lower*std
        hi = cen + sigma_upper*std
        mask = mask | (_arr < lo) | (hi < _arr)
        n_new = n_original - np.sum(mask)
        if n_new == n_old:  # if no more rejection
            break
        n_old = n_new

    return cen


@nb.njit(fastmath=True, parallel=True)
def _qcomb_med_2d_full(_arr, mask, out_comb, out_std, out_lo, out_hi, out_niter,
                       out_nrej, out_mask, sigma_upper=3, sigma_lower=3,
                       maxiters=5):
    if _arr.ndim == 2:
        for i1 in nb.prange(out_comb.shape[0]):  # must be == _arr.shape[1]
            res = _qcomb_med_1d_full(
                _arr[:, i1],
                mask=mask[:, i1],
                sigma_upper=sigma_upper,
                sigma_lower=sigma_lower,
                maxiters=maxiters
            )
            out_comb[i1] = res[0]
            out_std[i1] = res[1]
            out_lo[i1] = res[2]
            out_hi[i1] = res[3]
            out_niter[i1] = res[4]
            out_nrej[i1] = res[5]
            out_mask[:, i1] = res[6]


@nb.njit(fastmath=True, parallel=True)
def _qcomb_med_2d_simple(_arr, out_comb, mask, sigma_upper=3, sigma_lower=3,
                         maxiters=5):
    ''' Median combine along axis = 0; result only
'''
    for i1 in nb.prange(out_comb.shape[0]):  # must be == _arr.shape[1]
        out_comb[i1] = _qcomb_med_1d_simple(
            _arr[:, i1],
            mask=mask[:, i1],
            sigma_upper=sigma_upper,
            sigma_lower=sigma_lower,
            maxiters=maxiters
        )


@nb.njit(fastmath=True, parallel=True)
def _qcomb_mean_2d_full(_arr, mask, out_comb, out_std, out_lo, out_hi, out_niter,
                        out_nrej, out_mask, sigma_upper=3, sigma_lower=3,
                        maxiters=5):
    for i1 in nb.prange(out_comb.shape[0]):  # must be == _arr.shape[1]
        res = _qcomb_mean_1d_full(
            _arr[:, i1],
            mask=mask[:, i1],
            sigma_upper=sigma_upper,
            sigma_lower=sigma_lower,
            maxiters=maxiters
        )
        out_comb[i1] = res[0]
        out_std[i1] = res[1]
        out_lo[i1] = res[2]
        out_hi[i1] = res[3]
        out_niter[i1] = res[4]
        out_nrej[i1] = res[5]
        out_mask[:, i1] = res[6]


@nb.njit(fastmath=True, parallel=True)
def _qcomb_mean_2d_simple(_arr, out_comb, mask, sigma_upper=3, sigma_lower=3,
                          maxiters=5):
    ''' Median combine along axis = 0
result only
'''
    for i1 in nb.prange(out_comb.shape[0]):  # must be == _arr.shape[1]
        out_comb[i1] = _qcomb_mean_1d_simple(
            _arr[:, i1],
            mask=mask,
            sigma_upper=sigma_upper,
            sigma_lower=sigma_lower,
            maxiters=maxiters
        )

"""
