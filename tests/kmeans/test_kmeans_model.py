import unittest
from src.kmeans.kmeans_model import KMeans
from src.maths.vectors import Vector
import numpy as np
import random


class TestKmeansModel(unittest.TestCase):

    def test_kmeans(self):
        # Create 2 points, and each point has 2 coordinates
        random.seed(12)
        n_points = 20
        n_clusters = 3
        points = []

        y_max = 40
        y_min = -30

        x_max = 30
        x_min = -60

        # Generate coordinates
        x = [random.randrange(x_min, x_max) for x in range(n_points)]
        y = [random.randrange(y_min, y_max) for y in range(n_points)]

        x = np.array(x)
        y = np.array(y)

        # Creation of the vector objects:
        vectors = []
        for i in range(len(x)):
            vectors.append(Vector([x[i], y[i]]))

        kmeans = KMeans(num_clusters=3)
        kmeans.train(inputs=vectors)

        print(points)
