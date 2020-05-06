# nbcombine
Array combination with rejection using numba.



I made this for astronomical image combination (to achieve combine speed comparable to IRAF) using python. I am especially in a need for median combine of FITS files with sigma-clipping, but [``ccdproc``](https://github.com/astropy/ccdproc) is too slow for me.



## Requirements

* numpy (tested 1.18.0)

* numba (tested 0.50.0)



## Limits

### Likely to be updated

1. **It is still slow compared to what I expected.** Maybe I am doing something wrong.
2. Currently only sigma-clip is supported for rejection method.
3. Memory problem: ``nbcombine`` is blind to any memory limit. Be careful when you're dealing with too many images or large-sized images.



### *Un*likely to be updated

1. Only mean/median are supported for combination.



## Example

To compare it with IRAF (IRAF time benchmark [here](https://astro.uni-bonn.de/~sysstw/lfa_html/iraf/images.imcombine.html#reject), but it's too old machine...), I made 10 images with 1000 by 200 pixels with real values. In numpy, I used 64-bit float.

```python
import numpy as np
from nbcombine import qcomb

np.random.seed(123)

images = []
for i in range(10):
    image = np.random.normal(size=(1000, 200))
    images.append(image)
images = np.array(images)
qcomb(images, combine='mean', dtype='float32')
qcomb(images, combine='median', dtype='float32')

res = qcomb(images, combine='median', sigma=1, maxiters=5, dtype='float32', full=True)
# try playing with the followings: 
#   res['comb']  - final combined map
#   res['std']   - sample stdev map
#   res['niter'] - number of iteration map
#   res['nrej']  - number of rejected pixel map
#   res['mask']  - mask map (3-D)
```

For IPython magic, you can use

```
%timeit -n 5 -r 5 qcomb(images, combine='mean', sigma=1, maxiters=0, dtype='float32')
%timeit -n 5 -r 5 qcomb(images, combine='mean', sigma=1, maxiters=5, dtype='float32')
%timeit -n 5 -r 5 qcomb(images, combine='median', sigma=1, maxiters=0, dtype='float32')
%timeit -n 5 -r 5 qcomb(images, combine='median', sigma=1, maxiters=5, dtype='float32')

# 74.5 ms ± 1.5 ms per loop (mean ± std. dev. of 5 runs, 5 loops each)
# 361 ms ± 15.4 ms per loop (mean ± std. dev. of 5 runs, 5 loops each)
# 110 ms ± 887 µs per loop (mean ± std. dev. of 5 runs, 5 loops each)
# 516 ms ± 8.86 ms per loop (mean ± std. dev. of 5 runs, 5 loops each)
```

* **NOTE**: You must have run ``qcomb`` with proper input arguments prior to ``%timeit`` for fair timing, because numba's jit takes some time for compilation.
* The results are from my Mac Book Pro 2018 15" (2.6 GHz Intel Core i7; 16 GB 2400 MHz DDR4; macOS 10.14.6).

**I think what I wrote is not so efficient as it has no much gain compared to astropy or numpy:**

```
from astropy.stats import sigma_clip
%timeit -n 5 -r 5 sigma_clip(images, cenfunc='median', stdfunc='std', sigma=1, maxiters=5, axis=0)
%timeit -n 5 -r 5 np.mean(np.array(images[np.ones(images.shape, dtype=bool)], dtype='float32', order='C'), axis=0)
%timeit -n 5 -r 5 np.median(np.array(images[np.ones(images.shape, dtype=bool)], dtype='float32', order='C'), axis=0)
%timeit -n 5 -r 5 np.std(images[np.ones(images.shape, dtype=bool)], axis=0, ddof=1)
# 203 ms ± 4.02 ms per loop (mean ± std. dev. of 5 runs, 5 loops each)
# 8.33 ms ± 415 µs per loop (mean ± std. dev. of 5 runs, 5 loops each)
# 29.8 ms ± 352 µs per loop (mean ± std. dev. of 5 runs, 5 loops each)
# 13.1 ms ± 428 µs per loop (mean ± std. dev. of 5 runs, 5 loops each)
```





IRAF used 3-sigma with unknown ``maxiters``, and hence it may have done a less aggressive rejection, i.e., quicker calculation. 