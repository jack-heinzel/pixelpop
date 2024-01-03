import jax.numpy as jnp
import scipy
from tqdm import tqdm

class generalized_number(object):
    def __init__(self, b10, base=10, dimension=1):
        self.base_10 = b10
        self.base = base
        self.dimension = dimension
        self.base10_to_base()

    def base10_to_base(self):
        r = []
        i = self.base_10
        for d in range(self.dimension):
            power = self.base**(self.dimension-1-d)
            c = i // power
            i -= c*power
            r.append(c)
        self.coefficients = r

    def __add__(self, obj):
        if len(self.coefficients) != len(obj.coefficients):
            print('error')
            return None
        return [self.coefficients[ii] + obj.coefficients[ii] for ii in range(self.dimension)]

    def __sub__(self, obj):
        if len(self.coefficients) != len(obj.coefficients):
            print('error')
            return None
        return [self.coefficients[ii] - obj.coefficients[ii] for ii in range(self.dimension)]


def is_valid(l, base, dimension):
    if len(l) != dimension:
        return False
    for coef in l:
        if coef < 0 or coef >= base:
            return False
    return True

def coordinate_to_index(coordinate, density, dimension):
    '''
    Computes the index of a C-style flattening of a dimension-d array with axis length density
    See https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html

    Parameters
    =====================
    (np.ndarray) coordinate: array of coordinates, ie. [coordinate0, coordinate1, ...]
        where each coordinate may be an ND-array
    
    (int) density: length along each dimension, usually the number of bins
    (int) dimension: the dimension of the space, usually 1 or 2
    '''
    r = jnp.zeros_like(coordinate[0])
    for ii, c in enumerate(coordinate):
        r += c * density**(dimension - 1 - ii)
    return r

def index_to_coordinate(index, dimension, density):
    r = []
    i = index
    for d in range(dimension):
        power = density**(dimension-1-d)
        c = i // power
        i -= c*power
        r.append(c)
    return r

def nearest_neighbors(density, dimension, isVisible=False):
    
    indices = jnp.arange(0, density**dimension)

    i_vals = []
    j_vals = []

    powers = [generalized_number(density**d, base=density, dimension=dimension) for d in range(dimension)]
    if isVisible:
        print(f'Computing nearest neighbors list for {dimension}-dimensional grid of size {density}')
        array = tqdm(indices)
    else:
        array = indices
    for index in array:
        converted = generalized_number(index, base=density, dimension=dimension)
        for d in range(dimension):
            if is_valid(converted + powers[d], density, dimension):
                i_vals.append(index)
                j_vals.append(coordinate_to_index(converted+powers[d], density=density, dimension=dimension))
            if is_valid(converted - powers[d], density, dimension):
                i_vals.append(index)
                j_vals.append(coordinate_to_index(converted-powers[d], density=density, dimension=dimension))
            
    return i_vals, j_vals

def create_CAR_coupling_matrix(density, dimension, isVisible=False):
    '''
    Parameters
    ===================
    density     Arraylike of integers or int
    dimension   int
    minimums    Arraylike
    maximums    Arraylike
    '''

    i, j = nearest_neighbors(density, dimension, isVisible=isVisible) # on a Euclidean grid there are 2*dimension neighbors for each location, we should not use BvK boundary conditions 
    adjancency_matrix = scipy.sparse.coo_array((jnp.ones(len(i)), (i, j)))  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.html#scipy.sparse.coo_array

    return adjancency_matrix


def place_grid_in_bins(bin_axes, minimums, maximums, grid_density):
    dimension = len(minimums)
    if dimension > 1:
        density = len(bin_axes[0])
    else:
        density = len(bin_axes)
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

def place_samples_in_bins(bin_axes, sample_coordinates):
    '''
    Computes the bin index for each sample in sample_coordinates. Assumes a hyper-cubic 
    lattice with bins along each dimension provided in bin_axes.

    Parameters
    ===================
    (np.ndarray) bin_axes: list of bin boundaries, ie. [bins0, bins1, ...] where
        each bins is a 1-dimensional list of boundaries between bins
    (np.ndarray) sample coordinates: list of samples, the 0th axis should correspond 
        to the 0th axis of bin_axes, ie. sample_coordinates[0] should be m1 samples 
        if bin_axes[0] defines the m1 bins

    Returns
    ===================
    (np.ndarray) array of coordinates in a flattened set of N-dimensional bins. Same
        shape as sample_coordinates[0]
    '''
    dimension = len(sample_coordinates)
    density = len(bin_axes[0]) - 1
    # print(f'dimension = {dimension}, density = {density}')
    _data_nd_bins = [jnp.digitize(sample_coordinates[i], bin_axes[i]) - 1 for i in range(dimension)]
    # print(_data_nd_bins)
    _data_bins = coordinate_to_index(_data_nd_bins, density, dimension)
    return _data_bins