import unittest
import pytest

from src.maths.vectors import *


class TestVectors(unittest.TestCase):

    def test_creation_vector(self):
        v = Vector([1, 1.3, 5, 2])
        self.assertEqual(v.__str__(), '[1, 1.3, 5, 2]')

    def test_vectors_has_same_size(self):
        v1 = Vector([1, 1])
        v2 = Vector([2, 2])
        self.assertTrue(vectors_has_same_length([v1, v2]))

    def test_vectors_has_not_same_size(self):
        v1 = Vector([1, 1, 0])
        v2 = Vector([2, 2])
        self.assertFalse(vectors_has_same_length([v1, v2]))

    def test_num_differences_between_vectors(self):
        v1 = Vector([1, 2, 3])
        v2 = Vector([2, 1, 3])
        self.assertEqual(num_differences(v1, v2), 2)

        v1 = Vector([1, 2, 3])
        v2 = Vector([1, 2, 3])
        self.assertEqual(num_differences(v1, v2), 0)

    def test_vector_mean_vectors_have_same_size(self):
        # Vectors with different size raise an error
        v1 = [3, 4]
        v2 = [5, 6]
        v3 = [1, 2, 3]
        with pytest.raises(Exception):
            vector_mean(v1, v2, v3)

    def test_vector_mean_vectors_not_have_same_size(self):
        v1 = [1, 2]
        v2 = [2, 1, 3]
        with pytest.raises(Exception):
            num_differences(v1, v2)

    def test_vector_mean(self):
        # Vectors with different size raise an error
        v1 = Vector([1, 2])
        v2 = Vector([3, 4])
        v3 = Vector([5, 6])
        mean_vector = vector_mean([v1, v2, v3])
        self.assertEqual(mean_vector.values, [3.0, 4.0])
        print(mean_vector)

    def test_sum_vectors(self):
        v1 = Vector([3, 4])
        v2 = Vector([5, 6])
        sum_vector = vector_sum([v1, v2])
        self.assertEqual(sum_vector.values, [8, 10])

    def test_subtract_vectors(self):
        v1 = Vector([3, 4])
        v2 = Vector([5, 1])
        sum_vector = vector_substract(v1, v2)
        self.assertEqual(sum_vector.values, [-2, 3])

    def test_sum_of_squares(self):
        v1 = Vector([1, 2, 3])
        self.assertEqual(sum_of_squares(v1), 14)

    def test_distance(self):
        v1 = Vector([1, 0, 5])
        v2 = Vector([0, 2, 4])

        self.assertEqual(distance(v1, v2), math.sqrt(6))
