import numpy as np

from util import COMB_FUNCS, _qcomb_setup, _qcomb_wrapup

__all__ = ['qcomb']


def qcomb(arr, mask=None, sigma=3, sigma_upper=None, sigma_lower=None,
          maxiters=5, combine='median', reject='sigclip', full=False,
          nkeep=3, dtype=None):
    """ Median combine along axis=0
    Note
    ----
    ``axis=0`` is simple and reasonable because if you do ``lines = []``
    and ``lines.append(line)``, the ``lines`` will need to be combined
    along ``axis=0``.

    Parameters
    ----------
    arr : N-D array
        The data to be processed.

    mask : N-D boolean array
        The mask to be used to ignore values in ``arr``. Must have the
        identical shape to ``arr``.

    sigma_upper, sigma_lower : float, optional.
        The upper/lower sigma values to reject data if ``reject =
        'sigclip'``.

    maxiters : int, optional.
        The maximum iteration numbers for rejection. If ``maxiters =
        0``, no rejection will occur, but the outputs are calculated for
        homogeneous return values.

    combine/reject : str, optional
        The combine and rejection methods.

    nkeep : int, optional.
        The minimum number of pixels to proceed sigma-clipping. If the
        remaining pixels are fewer than this value, no further iteration
        will be made. At least 3 is recommended as the sigma (sample
        standard deviation) is defined only if we have more than 3 data.

    dtype : data type, optional.
        The data type of the final output combined array. If ``None``
        (default), it will be inferred from the input ``arr``. If dtype
        cannot be inferred from ``arr``, numpy default ('float64' as of
        2020-05-06) is used.

    Returns
    -------
    res : dict
        A dict object which contains the following key-value pairs:
        * ``'cen'`` : ndarray of ``dtype``
            The resulting combined array after rejection, in the data
            type of ``dtype``.

        * ``'std'`` : ndarray of ``dtype``
            The final sigma (sample standard deviation) of the remaining
            data after rejection.

        * ``'lo/hi'`` : ndarray of ``dtype``
            The lower/upper boundaries from the final sigma-clipping
            (only data of ``lo < x < hi`` are survived).

        * ``'niter'`` : ndarray of int
            The number of iterations when the sigma-clipping is halted.
            If no rejection was done, this is ``0``.

        * ``'nrej'`` : ndarray of int
            The number of rejected data.

        * mask : 1-D array of bool
            The boolean array (same shape as ``_arr1d``) which is the
            final mask after the sigma-clipping, propagated from the
            initial ``mask`` provided by the user.
    """
    # * Use arr in C-contiguous
    # if not, indexing is slow:
    # https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html?highlight=flat#the-flat-object
    arr = np.array(arr, order='C', dtype=dtype)
    dtype = arr.dtype

    setup = _qcomb_setup(
        arr=arr,
        mask=mask,
        sigma=sigma,
        sigma_upper=sigma_upper,
        sigma_lower=sigma_lower,
        maxiters=maxiters,
        combine=combine,
        reject=reject,
        full=full,
        dtype=dtype
    )
    mask = setup[0]
    sigma_upper = setup[1]
    sigma_lower = setup[2]
    maxiters = setup[3]
    combine = setup[4]
    reject = setup[5]
    outs = setup[6]
    _qcomb = COMB_FUNCS[(combine, reject, full)]

    if full:
        _qcomb(
            _arr=arr,
            **outs,
            mask=mask,
            sigma_upper=sigma_upper,
            sigma_lower=sigma_lower,
            maxiters=maxiters,
            nkeep=nkeep
        )
        res = _qcomb_wrapup(outs)
    else:
        _qcomb(
            _arr=arr,
            out_comb=outs['out_comb'],
            mask=mask,
            sigma_upper=sigma_upper,
            sigma_lower=sigma_lower,
            maxiters=maxiters,
            nkeep=nkeep
        )
        res = outs['out_comb']
    return res
