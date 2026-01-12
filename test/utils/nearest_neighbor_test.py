import unittest
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_array
from pixelpop.utils.nearest_neighbor import (
    is_valid,
    coordinate_to_index,
    index_to_coordinate,
    nearest_neighbors,
    create_CAR_coupling_matrix,
    place_samples_in_bins
)

class TestCoordinateTransformations(unittest.TestCase):

    def test_is_valid(self):
        # Test with integer density (uniform)
        self.assertTrue(is_valid([0, 0], base=5, dimension=2))
        self.assertTrue(is_valid([4, 4], base=5, dimension=2))
        self.assertFalse(is_valid([-1, 0], base=5, dimension=2))
        self.assertFalse(is_valid([5, 0], base=5, dimension=2))
        
        # Test with list density (non-uniform)
        density = [5, 10]
        self.assertTrue(is_valid([4, 9], base=density, dimension=2))
        self.assertFalse(is_valid([4, 10], base=density, dimension=2))

    def test_coordinate_to_index_1d(self):
        # 1D case: coordinate [5] in density 10 should be index 5
        coord = [jnp.array([5])]
        idx = coordinate_to_index(coord, density=10, dimension=1)
        self.assertEqual(idx, 5)

    def test_coordinate_to_index_2d(self):
        # 2D case: 3x3 grid. Coordinate [1, 1] should be index 4 (C-order: 1*3 + 1)
        coord = [jnp.array([1]), jnp.array([1])]
        idx = coordinate_to_index(coord, density=3, dimension=2)
        self.assertEqual(idx, 4)

    def test_index_to_coordinate(self):
        # Inverse of the above: index 4 in 3x3 grid -> [1, 1]
        coords = index_to_coordinate(4, dimension=2, density=3)
        self.assertEqual(list(coords), [1, 1])

class TestNeighborLogic(unittest.TestCase):

    def test_nearest_neighbors_1d(self):
        # 1D grid of size 3. Indices: 0, 1, 2
        # Neighbors: 0-1, 1-0, 1-2, 2-1
        i, j = nearest_neighbors(density=3, dimension=1)
        
        expected_pairs = {(0, 1), (1, 0), (1, 2), (2, 1)}
        actual_pairs = set(zip(map(int, i), map(int, j)))
        self.assertEqual(actual_pairs, expected_pairs)

    def test_nearest_neighbors_2d_corners(self):
        # 2x2 grid. Index 0 (coord 0,0) should have neighbors at 1 (0,1) and 2 (1,0)
        i, j = nearest_neighbors(density=2, dimension=2)
        i_arr, j_arr = np.array(i), np.array(j)
        
        neighbors_of_zero = j_arr[i_arr == 0]
        self.assertIn(1, neighbors_of_zero)
        self.assertIn(2, neighbors_of_zero)
        self.assertEqual(len(neighbors_of_zero), 2)

class TestCARMatrix(unittest.TestCase):

    def test_create_CAR_coupling_matrix_shape(self):
        density = 5
        dimension = 2
        adj = create_CAR_coupling_matrix(density, dimension)
        
        # Matrix should be (density^dimension) x (density^dimension)
        expected_size = density ** dimension
        self.assertEqual(adj.shape, (expected_size, expected_size))
        
        # Matrix should be symmetric for a Euclidean grid
        adj_dense = adj.toarray()
        np.testing.assert_array_equal(adj_dense, adj_dense.T)

    def test_adjacency_connectivity(self):
        # In a 3x3 grid, the center cell (index 4) should have 4 neighbors
        adj = create_CAR_coupling_matrix(density=3, dimension=2)
        adj_dense = adj.toarray()
        self.assertEqual(np.sum(adj_dense[4]), 4)
        
        # A corner cell (index 0) should have 2 neighbors
        self.assertEqual(np.sum(adj_dense[0]), 2)

class TestBinningBoundaries(unittest.TestCase):

    def setUp(self):
        # Create a simple 2D grid: 
        # Dim 0: [0, 5, 10] (2 bins)
        # Dim 1: [0, 2, 4, 6] (3 bins)
        self.bin_axes = [
            jnp.array([0.0, 5.0, 10.0]),
            jnp.array([0.0, 2.0, 4.0, 6.0])
        ]
        self.density = [2, 3]
        self.dimension = 2

    def test_boundary_conditions(self):
        """Test if samples exactly on the edges fall into the expected bins."""
        # Sample A: Exactly on the lower bound (0.0, 0.0) -> Bin (0, 0)
        # Sample B: Exactly on an internal edge (5.0, 2.0) -> Bin (1, 1)
        # Sample C: Exactly on the upper bound (10.0, 6.0) -> Bin (2, 3) 
        #           (Outside valid range in both dimensions)
        sample_coords = [
            jnp.array([0.0, 5.0, 10.0]),
            jnp.array([0.0, 2.0, 6.0])
        ]
        
        # reshape=False returns a tuple of index arrays
        indices = place_samples_in_bins(self.bin_axes, sample_coords, reshape=False)
        
        # Sample A
        self.assertEqual(indices[0][0], 0)
        self.assertEqual(indices[1][0], 0)
        
        # Sample B (Digitize behavior: x >= edge is the next bin)
        self.assertEqual(indices[0][1], 1)
        self.assertEqual(indices[1][1], 1)
        
        # Sample C (Overflow check)
        # jnp.digitize returns len(bins) for values >= last edge. 
        # After subtracting 1, it should be equal to the density.
        self.assertEqual(indices[0][2], 2) 
        self.assertEqual(indices[1][2], 3)

    def test_flattening_consistency(self):
        """Test if reshaped (flattened) indices map back to correct coordinates."""
        # A sample in the middle of bin (1, 2)
        # Dim 0: 7.5 (Bin 1)
        # Dim 1: 5.0 (Bin 2)
        sample_coords = [jnp.array([7.5]), jnp.array([5.0])]
        
        flat_idx = place_samples_in_bins(self.bin_axes, sample_coords, reshape=True)
        
        # Unravel the index to see if it matches bin (1, 2)
        # Note: coordinate_to_index uses C-order
        recovered_coords = index_to_coordinate(
            flat_idx[0], 
            dimension=self.dimension, 
            density=self.density
        )
        
        self.assertEqual(int(recovered_coords[0]), 1)
        self.assertEqual(int(recovered_coords[1]), 2)

if __name__ == '__main__':
    unittest.main()