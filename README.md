Package for non-parameteric inference of a gravitational wave population, built on JAX. This is aimed particularly at non-parameteric inference in spaces with dimension 2, but in theory can work in a space of any dimension.

This method works by binning the space into a cartesian grid, and inferring the log-probability density in each bin, each of which is a free parameter. 
Each bin is coupled to its nearest-neighbors, with the strength of this coupling another free parameter. 
In order to be robust, the bins should be reasonably dense in the space, hence the dimension of the inference problem can become very large (e.g. 10^4 for a 2-dimensional space with a density of 100 bins along each axis). 
By leveraging auto-differentiation in JAX, sparse matrices for the nearest-neighbor coupling, and the efficient No-U-Turn HMC sampler in numpyro, this problem is tractable.


