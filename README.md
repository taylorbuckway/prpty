# prpty

`prpty` is a Python toolkit for preparing, simulating, and analyzing ptychography data. It covers the steps around a reconstruction: loading and cleaning diffraction data, building probes and scan patterns, propagating wavefields, and post-processing reconstructed objects (for example, phase-ramp removal).

It is designed to work alongside reconstruction packages such as [tike](https://github.com/AdvancedPhotonSource/tike).

## Features

| Module | What it does |
|---|---|
| `load.py` | Loaders for HDF5 and text data: Andor `.asc` files, APS 2-ID-D data, simulated datasets, and saved reconstructions (`LoadAPS_2IDD`, `LoadRecon`, `extract_from_hdf5`, …) |
| `dataprocessing.py` | Background subtraction, hot/dead-pixel removal, filtering, centering, and batching of diffraction patterns (`imageprocessing`, `ptychoprocessing`); phase-ramp correction of reconstructions (`post_processing.ramp_correction_MG`); subpixel image registration |
| `probe_create.py` | Analytic probe generation (Gaussian, Hermite–Gaussian, and Laguerre–Gaussian beams) via the `probes` class |
| `initialize_probe.py` | Initial probe guess from averaged diffraction data with Fresnel propagation (`add_guessed_phase`) |
| `zernike.py` | Zernike polynomials and probes that vary with scan position (tilt, defocus, astigmatism) |
| `scans.py` | Scan-pattern generators: raster, concentric, spiral, and Fermat spiral, plus path-length minimization |
| `affine_transform.py` | Apply affine distortions (rotation, shear, scale) to scan positions |
| `fourier_operators.py` | Far-field FFT propagation, its inverse, and Fresnel propagation |
| `centering.py` | Image centering by center of mass, cross-correlation, or maximum value |
| `measurements.py` | Beam-waist measurement, per-frame center of mass, image similarity and cross-correlation |
| `plot.py` | Plotting helpers for complex images, eigen-probes, weights, scan positions, and animated image stacks |
| `random_utils.py` | Random padding utilities |

## Installation

Clone the repository into a folder on your Python path:

```bash
git clone https://github.com/taylorbuckway/prpty.git
```

Install the dependencies:

```bash
pip install numpy scipy matplotlib h5py scikit-image scikit-learn lmfit natsort psutil
```

### Optional GPU support

`dataprocessing.py` automatically uses [CuPy](https://cupy.dev/) when a CUDA GPU is available and falls back to NumPy/SciPy otherwise. To enable GPU support:

```bash
pip install cupy-cuda12x   # choose the build matching your CUDA version
```

## Quick start

```python
import numpy as np
from prpty import scans, centering
from prpty.probe_create import probes
from prpty.fourier_operators import prop_fft

# Generate a Fermat-spiral scan (center, width, height, step)
positions = scans.fermat_spiral(center=(0, 0), width=10e-6, height=10e-6, step=0.5e-6)

# Build a probe grid: wavelength, real-space pixel size, number of pixels
p = probes(lam=1.24e-10, pixel_size=10e-9, pixel_number=256)

# Find the center of an image
image = np.random.random((256, 256))
center = centering.find_center(image, method="CM")
```

### Loading data

```python
from prpty import load

data = load.extract_from_hdf5("data", "path/to/your_file.h5")
```

Keep data paths out of the code you commit. Pass them in at run time, or read them from an environment variable or a local config file that is listed in `.gitignore`.

## Requirements

- Python 3.10+
- NumPy, SciPy, Matplotlib, h5py, scikit-image, scikit-learn, lmfit, natsort, psutil
- CuPy (optional, for GPU acceleration)

## References

- M. Guizar-Sicairos *et al.*, "Phase tomography from x-ray coherent diffractive imaging projections," *Opt. Express* **19**, 21345–21357 (2011). [doi:10.1364/OE.19.021345](https://doi.org/10.1364/OE.19.021345). Used for phase-ramp correction.
- J. Peatross and M. Ware, *Physics of Light and Optics*, Ch. 11. Used for the Gaussian beam models.

## Citation

If you use this code in published work, please cite:

```
T. Buckway, prpty (2026). https://github.com/taylorbuckway/prpty
```

## License

<!-- TODO: choose a license (e.g., MIT or BSD-3-Clause) and add a LICENSE file. Check BYU / lab policy first. -->

## Author

Taylor Buckway, Department of Physics and Astronomy, Brigham Young University
