import numpy as jnp
import scipy
from tqdm import tqdm
import warnings

def is_valid(l, base, dimension):
    """
    Validate if a multidimensional coordinate is within the grid boundaries.

    Checks if a given coordinate vector (indices) is valid for a grid of a 
    specified dimension and density. A coordinate is valid if it has the 
    correct number of dimensions and every component $i$ satisfies 
    $0 \le \text{coord}_i < \text{base}_i$.

    Parameters
    ----------
    l : array_like
        The coordinate vector to check (e.g., a list of bin indices).
    base : int or list of int
        The upper boundary for the indices. If an integer, the same boundary 
        is applied to all dimensions. If a list, it must specify the boundary 
        for each dimension individually.
    dimension : int
        The expected dimensionality of the coordinate vector.

    Returns
    -------
    bool
        True if the coordinate is within bounds and the dimension matches, 
        False otherwise.

    Examples
    --------
    >>> is_valid([56, 21], base=100, dimension=2)
    True
    >>> # 100 is out of bounds for a density of 100 (valid indices are 0-99)
    >>> is_valid([100, 21], base=100, dimension=2)
    False
    """

    if len(l) != dimension:
        return False
    if isinstance(base, int):
        for coef in l:
            if coef < 0 or coef >= base:
                return False
        return True
    elif isinstance(base, list):
        for ii, coef in enumerate(l):
            if coef < 0 or coef >= base[ii]:
                return False
        return True
    

def coordinate_to_index(coordinate, density, dimension):
    """
    Convert multi-dimensional grid coordinates to flattened C-style indices.

    This function maps a set of coordinates (row, col, ...) to a single integer 
    index based on a row-major (C-style) flattening of the grid. It supports 
    broadcasting where 'coordinate' contains arrays of samples.

    Parameters
    ----------
    coordinate : sequence of array_like
        A list or tuple of coordinate arrays, e.g., ``[x_coords, y_coords]``.
        Each element can be a scalar or a JAX/NumPy array.
    density : int or list of int
        The number of bins along each axis. If an integer, the same density 
        is assumed for all dimensions.
    dimension : int
        The dimensionality of the grid (e.g., 1, 2, or 3).

    Returns
    -------
    jax.numpy.ndarray
        An array of flattened indices corresponding to the input coordinates.

    Raises
    ------
    IndexError
        If the length of ``density`` does not match ``dimension``.
    TypeError
        If ``density`` is not an integer or a list.

    See Also
    --------
    jax.numpy.ravel_multi_index : The underlying JAX function used for mapping.
    """
    if isinstance(density, int):
        # This means equal along all directions
        density = [density] * dimension
    elif isinstance(density, list):
        if len(density) != dimension:
            raise IndexError('Length of densities is different from dimension')
    else:
        raise TypeError('density must be an integer or list')
    
    coordinates = tuple(jnp.asarray(c, dtype=int) for c in coordinate)

    for c, d in zip(coordinates, density):
        if jnp.any(c < 0) or jnp.any(c >= d):
            print(c)

    indices = jnp.ravel_multi_index(coordinates, dims=density, order='C')
    return indices

def index_to_coordinate(index, dimension, density):
    """
    Convert flattened C-style indices back to multi-dimensional coordinates.

    Parameters
    ----------
    index : int or array_like
        The index or indices in the flattened array to be converted.
    dimension : int
        The dimensionality of the space.
    density : int or list of int
        The number of bins along each axis. If an integer, the same density 
        is assumed for all dimensions.

    Returns
    -------
    list of jax.numpy.ndarray
        A list of coordinate arrays, one for each dimension.

    Raises
    ------
    IndexError
        If the length of ``density`` does not match ``dimension``.
    """
    if isinstance(density, int):
        density = [density] * dimension
    elif isinstance(density, list):
        if len(density) != dimension:
            raise IndexError('Length of densities is different from dimension')
    else:
        raise TypeError('density must be an integer or list')
    
    coordinates = jnp.unravel_index(index, shape=density, order='C')
    return list(coordinates)
        
def nearest_neighbors(density, dimension, isVisible=False):
    """
    Identify the indices of immediate neighbors for every cell in a grid.

    Constructs an adjacency list for a Euclidean grid without periodic 
    (boundary) conditions. For a d-dimensional grid, internal cells 
    will have up to $2d$ neighbors.

    Parameters
    ----------
    density : int or list of int
        Number of bins along each dimension.
    dimension : int
        The dimensionality of the grid.
    isVisible : bool, optional
        If True, displays a progress bar (tqdm) during computation. 
        Default is False.

    Returns
    -------
    i_vals : list of int
        Source indices (the "from" nodes in a graph).
    j_vals : list of int
        Neighbor indices (the "to" nodes in a graph).
    """
    if isinstance(density, int):
        indices = jnp.arange(0, density**dimension)
        density = [density]*dimension
        powers = jnp.eye(dimension) #[generalized_number(density**d, base=density, dimension=dimension) for d in range(dimension)]

    elif isinstance(density, list) or isinstance(density, tuple):
        # raise exception if len(density) != dimension
        if len(density) != dimension:
            raise IndexError('Length of densities is different from dimension')
        indices = jnp.arange(0, jnp.prod(density))
        powers = jnp.eye(dimension)
        #print(powers)
    else:
        raise TypeError('density must be an integer or list / tuple')
    i_vals = []
    j_vals = []

    if isVisible:
        print(f'Computing nearest neighbors list for {dimension}-dimensional grid of size {density}')
        array = tqdm(indices)
    else:
        array = indices
    for index in array:
        converted = jnp.array(jnp.unravel_index(index, shape=density, order='C')) 
        for d in range(dimension):
            #print(index, converted + powers[d], is_valid(converted + powers[d], density, dimension))
            #print(index, converted - powers[d], is_valid(converted - powers[d], density, dimension))
            if is_valid(converted + powers[d], density, dimension):
                i_vals.append(index)
                j_vals.append(coordinate_to_index(converted+powers[d], density=density, dimension=dimension))
            if is_valid(converted - powers[d], density, dimension):
                i_vals.append(index)
                j_vals.append(coordinate_to_index(converted-powers[d], density=density, dimension=dimension))
            
    return i_vals, j_vals

def create_CAR_coupling_matrix(density, dimension, isVisible=False):
    """
    Create a sparse adjacency matrix for a Conditional Auto-Regressive (CAR) model.

    The resulting matrix is a symmetric sparse matrix where an entry is 1 
    if two grid cells are nearest neighbors and 0 otherwise.

    Parameters
    ----------
    density : int or list of int
        Number of bins along each dimension.
    dimension : int
        The dimensionality of the grid.
    isVisible : bool, optional
        If True, displays a progress bar during neighbor calculation.

    Returns
    -------
    scipy.sparse.coo_array
        A sparse coordinate-format array representing the grid connectivity.

    Notes
    -----
    This implementation assumes a Euclidean grid and does not apply 
    Born-von Karman (periodic) boundary conditions.
    """

    i, j = nearest_neighbors(density, dimension, isVisible=isVisible) 
    adjancency_matrix = scipy.sparse.coo_array((jnp.ones(len(i)), (i, j)))  

    return adjancency_matrix


def place_samples_in_bins(bin_axes, sample_coordinates, reshape=False):
    """
    Discretize real number sample coordinates into multi-dimensional bin indices.

    This function determines which bin each sample falls into based on provided 
    axis boundaries. It supports returning either the multi-dimensional 
    bin indices (default) or a single flattened index per sample.

    Parameters
    ----------
    bin_axes : list of array_like
        A list of arrays where each array defines the bin edges for a specific 
        dimension (e.g., ``[edges_x, edges_y]``).
    sample_coordinates : sequence of array_like
        A sequence of arrays containing the coordinates of the samples. 
        The length of the sequence must match the length of ``bin_axes``.
    reshape : bool, optional
        If True, returns a single flattened index for each sample using 
        C-style ordering. If False, returns a tuple of index arrays (one per 
        dimension). Default is False.

    Returns
    -------
    indices : tuple of jax.numpy.ndarray or jax.numpy.ndarray
        If ``reshape=False``, a tuple of arrays containing the bin index 
        for each dimension.
        If ``reshape=True``, a single array of flattened indices.

    Notes
    -----
    The binning logic uses 0-based indexing. A sample $x$ falls into bin $i$ 
    if $\text{edge}_i \le x < \text{edge}_{i+1}$.
    """
    dimension = len(sample_coordinates)
    # density = len(bin_axes[0]) - 1

    if isinstance(bin_axes, list):
        # Assuming bin_axes is a list of arrays
        density = [len(bin_axes[i]) - 1 for i in range(dimension)]
        if len(set(density)) == 1:
            density = density[0]
    else:
        # Assuming bin_axes is a single array 
        density = len(bin_axes[0]) - 1

    print(f'dimension = {dimension}, density = {density}')
    if not reshape:
        _data_nd_bins = ()
        for i in range(dimension):
            _data_nd_bins += (jnp.digitize(sample_coordinates[i], bin_axes[i]) - 1,)
        return _data_nd_bins
    else:
        _data_nd_bins = [jnp.digitize(sample_coordinates[i], bin_axes[i]) - 1 for i in range(dimension)]
    #print(_data_nd_bins)
    jnp.array(_data_nd_bins)
    _data_bins = coordinate_to_index(_data_nd_bins, density, dimension)
    return _data_bins


########################
# Deprecated functions #
########################


def place_grid_in_bins(bin_axes, minimums, maximums, grid_density):
    warnings.warn('place_grid_in_bins is deprecated, and will be removed in future versions.')
    dimension = len(minimums)
    if isinstance(bin_axes, list):
        # Assuming bin_axes is a list of arrays
        density = [len(bin_axes[i]) - 1 for i in range(dimension)]
        if len(set(density)) == 1:
            density = density[0]
    else:
        # Assuming bin_axes is a single array 
        density = len(bin_axes[0]) - 1
    m_axes = [jnp.linspace(minimums[d], maximums[d], grid_density) for d in range(dimension)]    
    ax_grids = jnp.meshgrid(*m_axes)
    grid = [ax_grids[ii].reshape(grid_density**dimension) for ii in range(dimension)]
    # pts_grid = jnp.array([ax_grids[ii] for ii in range(dimension)]).T.reshape((grid_density**dimension, dimension))
    # dV = jnp.prod([bin_axes[ii][1] - bin_axes[ii][0] for ii in range(len(bin_axes))])

    _data_2d_bins = jnp.array([jnp.digitize(grid[i], bin_axes[i]) for i in range(dimension)]).T - 1
    # print(data_2d_bins)
    _data_bins = jnp.array(
        [coordinate_to_index(_data_2d_bins[ii], density, dimension) for ii in range(len(_data_2d_bins))]
    )
    return _data_bins, m_axes, grid