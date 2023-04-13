import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.maths.vectors import Vector, vector_mean, distance, num_differences
from typing import List
import itertools
import random
import tqdm


class KMeans:
    def __init__(self, num_clusters: int):
        self.num_clusters = num_clusters

        # The mean of each cluster is not yet calculated, that is why we initialize every cluster
        # to an empty list
        self.clusters_mean = [[] for _ in range(num_clusters)]

    def train(self, inputs: List[Vector]):
        # We start by assigning randomly the inputs to the clusters
        assignments = [random.randrange(self.num_clusters) for _ in inputs]

        # Iterate until the there are no changes in the clusters.
        # As we don't know in advance how many iteration we will have to make, then create an infinite loop
        with tqdm.tqdm(itertools.count()) as t:
            for iteration in t:
                # Calculate the mean of each cluster
                self.calculate_mean_for_each_cluster(inputs, assignments)

                # self.visualize_training(inputs, assignments)

                # Classify each input into the closest cluster
                new_assignments = [self.classify(each_input) for each_input in inputs]

                # Now we can check if there have been changes in the assignments
                # If there are changes then we have to continue iterating
                # Else we can finish
                num_changes = num_differences(Vector(assignments), Vector(new_assignments))
                if num_changes == 0:
                    # Stop if there haven't been any changes
                    return

                # If there have been changes then we need to recalculate the means of each clusters with the
                # new assignments
                assignments = new_assignments

                t.set_description(f"iteration: {iteration} | Changes: {num_changes} / {len(inputs)}")

    def classify(self, v: Vector):
        """
        Return the index of the cluster closest to the input
        """
        # Calculate the distance between the input given and each cluster

        # Initialize with the max distance possible
        min_distance = float('inf')
        # Initialize to a cluster that doesn't exist
        closest_cluster = -1

        for cluster, each_cluster_mean in enumerate(self.clusters_mean):
            distance_to_cluster = distance(v, each_cluster_mean)
            if distance_to_cluster < min_distance:
                min_distance = distance_to_cluster
                closest_cluster = cluster

        return closest_cluster

    def calculate_mean_for_each_cluster(self,
                                        inputs: List[Vector],
                                        assignments: List[int]):
        # Create as many groups as clusters
        # Every group will be represented by a list

        # Initialize the clusters
        clusters = [[] for _ in range(self.num_clusters)]

        # Assign every input to its cluster
        for index, cluster in enumerate(assignments):
            clusters[cluster].append(inputs[index])

        # Now that every point is assigned to its cluster, we can calculate the mean for each cluster

        # Calculate the mean for each cluster
        self.clusters_mean = [vector_mean(each_cluster) for each_cluster in clusters]

    def visualize_training(self, inputs, assignments):

        # Print the point colored by their clusters
        x = [_.get(0) for _ in inputs]
        y = [_.get(1) for _ in inputs]
        point_type = ["o" for point in range(len(x))]
        sizes = [10 for point in range(len(x))]
        cluster_selected = assignments.copy()

        # Add the point for the cluster means
        for i in range(len(self.clusters_mean)):
            cluster = self.clusters_mean[i]
            x.append(cluster.get(0))
            y.append(cluster.get(1))
            point_type.append("*")
            cluster_selected.append(i)
            sizes.append(20)

        df = pd.DataFrame({"x": x,
                           "y": y,
                           "cluster": cluster_selected,
                           "point_type": point_type,
                           "size": sizes})

        plt.figure()

        sns.scatterplot(data=df, x='x', y='y', hue='cluster', style='point_type', ec=None,
                        legend=False, size='size')
        print("process")
