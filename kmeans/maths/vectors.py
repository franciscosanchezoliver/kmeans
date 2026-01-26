from typing import List
import math


class Vector:
    """
    A simple vector class to represent a list of numbers.
    """

    def __init__(self, list_of_numbers: List[float]):
        """
        Initialize the vector with a list of numbers.
        """
        self.values = list_of_numbers

    def get(self, pos: int):
        """
        Get the value at the specified position.
        """
        return self.values[pos]

    def size(self):
        """
        Get the size of the vector.
        """
        return len(self.values)

    def __str__(self):
        """
        String representation of the vector.
        """
        return self.values.__str__()

    def __repr__(self):
        """
        String representation of the vector.
        """
        return self.values.__str__()


def num_differences(v1: Vector, v2: Vector):
    """
    Count the number of differing elements between two vectors.
    """

    # Ensure both vectors have the same length
    if not vectors_have_same_length([v1, v2]):
        raise VectorSameSizeException

    differences = 0

    # Iterate over each number to count the differences
    for i in range(v1.size()):
        if v1.get(i) != v2.get(i):
            differences += 1

    return differences


def vectors_have_same_length(list_vectors: List[Vector]):
    """
    Given a list of vectors, check if they all have the same length.
    """
    # Iterate through the list of vectors
    for i in range(len(list_vectors) - 1):
        # If the size of the current vector is not equal to the next one,
        # return False
        if list_vectors[i].size() != list_vectors[i + 1].size():
            return False
    return True


def vector_mean(list_vectors: List[Vector]):
    mean = []

    num_dimensions = list_vectors[0].size()
    for i in range(num_dimensions):
        sum_dimension = 0.0
        for j in range(len(list_vectors)):
            sum_dimension += list_vectors[j].get(i)

        mean.append(sum_dimension / len(list_vectors))

    return Vector(mean)


def vector_sum(list_vectors):
    """
    Sum a list of vectors element-wise.
    """
    sum = []

    if not vectors_have_same_length(list_vectors):
        raise VectorSameSizeException

    # The number of dimensions is determined by the first vector
    num_dimensions = list_vectors[0].size()

    # Iterate over each dimension
    for i in range(num_dimensions):

        # Initialize the sum for this dimension
        sum_dimension = 0

        # Sum the values from each vector at this dimension
        for j in range(len(list_vectors)):
            sum_dimension += list_vectors[j].get(i)

        # Add the summed value to the result vector
        sum.append(sum_dimension)

    return Vector(sum)


def vector_subtract(v1: Vector, v2: Vector):
    """
    Subtract two vectors element-wise.
    """
    # Initialize an empty list to hold the result
    subtract = []

    # Ensure both vectors have the same length
    if not vectors_have_same_length([v1, v2]):
        raise VectorSameSizeException

    # Iterate over each element to perform subtraction
    for i in range(v1.size()):
        subtract.append(v1.get(i) - v2.get(i))

    return Vector(subtract)


def sum_of_squares(vector: Vector):
    """
    Calculate the sum of squares of the vector's elements.
    """
    # Initialize sum of squares to 0
    sum_squares = 0

    # Iterate over each value in the vector
    for val in vector.values:
        # Square the value and add it to the sum
        square = val ** 2
        sum_squares += square
    return sum_squares


def distance(v1: Vector, v2: Vector):
    # Formula to calculate distance between 2 vectors:
    return math.sqrt(sum_of_squares(vector_subtract(v1, v2)))


class VectorSameSizeException(Exception):
    def __init__(self):
        self.message = "Vectors must have same size"
        super().__init__(self.message)
