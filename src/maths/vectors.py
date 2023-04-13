from typing import List
import math


class Vector:

    def __init__(self, number_list: float):
        self.values = number_list

    def get(self, pos: int):
        return self.values[pos]

    def size(self):
        return len(self.values)

    def __str__(self):
        return self.values.__str__()

    def __repr__(self):
        return self.values.__str__()


def num_differences(v1: Vector, v2: Vector):
    if not vectors_has_same_length([v1, v2]):
        raise VectorSameSizeException

    differences = 0
    # Iterate over each number to count the differences
    for i in range(v1.size()):
        if v1.get(i) != v2.get(i):
            differences += 1

    return differences


def vectors_has_same_length(list_vectors: List[Vector]):
    for i in range(len(list_vectors) - 1):
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
    sum = []

    num_dimensions = list_vectors[0].size()
    for i in range(num_dimensions):
        sum_dimension = 0
        for j in range(len(list_vectors)):
            sum_dimension += list_vectors[j].get(i)

        sum.append(sum_dimension)

    return Vector(sum)


def vector_substract(v1: Vector, v2: Vector):
    subtract = []

    if not vectors_has_same_length([v1, v2]):
        raise VectorSameSizeException

    for i in range(v1.size()):
        subtract.append(v1.get(i) - v2.get(i))

    return Vector(subtract)


def sum_of_squares(vector: Vector):
    sum_squares = 0
    for val in vector.values:
        square = val ** 2
        sum_squares += square
    return sum_squares


def distance(v1: Vector, v2: Vector):
    # Formula to calculate distance between 2 vectors:
    # https://www.varsitytutors.com/calculus_3-help/distance-between-vectors
    return math.sqrt(sum_of_squares(vector_substract(v1, v2)))


class VectorSameSizeException(Exception):
    def __init__(self):
        self.message = "Vectors must have same size"
        super().__init__(self.message)
