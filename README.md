# PixelPop

Package for nonparameteric (AKA weakly modeled, data-driven) Bayesian inference of a gravitational wave population, built on `JAX` and `numpyro`. 
Aimed particularly at correlated nonparameteric inference in spaces with dimension 2-3.

This method works by binning the space into a cartesian grid, and inferring the log-rate density in each bin, each of which is a free parameter. 
Each bin is coupled to its nearest-neighbors using an intrinsic conditional-autoregressive (ICAR) model.

The dimension of the inference problem can become very large (e.g. 10^4 for a 2-dimensional space with a density of 100 bins along each axis), and 
we leverage auto-differentiation and GPU acceleration in `JAX`, as well as the efficient No-U-Turn HMC sampler in `numpyro` to sample the posterior.

#### Running PixelPop

Please see the example run scripts in the `examples/` directory.

#### Attribution

Please cite [Heinzel et al. (2025)](https://arxiv.org/abs/2406.16813) if you use PixelPop in your research.

```
@article{Heinzel:2024jlc,
    author = "Heinzel, Jack and Mould, Matthew and {\'A}lvarez-L{\'o}pez, Sof{\'\i}a and Vitale, Salvatore",
    title = "{High resolution nonparametric inference of gravitational-wave populations in multiple dimensions}",
    eprint = "2406.16813",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    doi = "10.1103/PhysRevD.111.063043",
    journal = "Phys. Rev. D",
    volume = "111",
    number = "6",
    pages = "063043",
    year = "2025"
}
```
Additionally, consider citing [Heinzel et al. (2025)](https://arxiv.org/abs/2406.16844) which applies PixelPop to GWTC-3
```
@article{Heinzel:2024hva,
    author = "Heinzel, Jack and Mould, Matthew and Vitale, Salvatore",
    title = "{Nonparametric analysis of correlations in the binary black hole population with LIGO-Virgo-KAGRA data}",
    eprint = "2406.16844",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    doi = "10.1103/PhysRevD.111.L061305",
    journal = "Phys. Rev. D",
    volume = "111",
    number = "6",
    pages = "L061305",
    year = "2025"
},
```
and [Alvarez-Lopez et al. (2025)](https://arxiv.org/abs/2506.20731) which shows PixelPop can accurately recover the complex, multi-dimensional correlations in a realistic population-synthesis population.
```
@article{Alvarez-Lopez:2025ltt,
    author = "Alvarez-Lopez, Sofia and Heinzel, Jack and Mould, Matthew and Vitale, Salvatore",
    title = "{Nowhere left to hide: revealing realistic gravitational-wave populations in high dimensions and high resolution with PixelPop}",
    eprint = "2506.20731",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "6",
    year = "2025"
}
```