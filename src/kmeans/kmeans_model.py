from src.maths.vectors import Vector, vector_mean, distance, num_differences
from typing import List
import itertools
import random
import tqdm


class KMeans:
    """
    KMeans is a popular unsupervised machine learning algorithm used for clustering data points into groups
    or clusters based on their similarity.
    """

    def __init__(self,
                 num_clusters: int,
                 save_intermediate_steps: bool = False):
        """
        Create the K-Means model object.

        :param num_clusters: Numbers of groups to divide the inputs
        :param save_intermediate_steps: True if we want to save all the intermediates steps to further analysis.
        """
        self.num_clusters = num_clusters

        self.save_intermediate_steps = save_intermediate_steps
        self.intermediate_steps = []

        # The mean of each cluster is not yet calculated, that is why we initialize every cluster
        # to an empty list
        self.clusters_mean = [[] for _ in range(num_clusters)]

    def train(self, inputs: List[Vector]):
        """
        During the training process, the model will separate the inputs into the number of clusters given.
        After the training the model will be able to select one cluster to an input it has never seen it before.

        :param inputs: the data we want to separate into groups.
        """

        # We start by assigning randomly the inputs to the clusters
        assignments = [random.randrange(self.num_clusters) for _ in inputs]

        # Iterate until the there are no changes in the clusters.
        # As we don't know in advance how many iteration we will have to make, then create an infinite loop
        with tqdm.tqdm(itertools.count()) as t:
            for iteration in t:

                # Calculate the mean of each cluster
                self._calculate_mean_for_each_cluster(inputs, assignments)

                # If the 'save_intermediate_steps' is set to True then save each step
                # to review it later
                if self.save_intermediate_steps:
                    self.save_intermediate_state(assignments)

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

        :param v: The input we want to classify
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

    def _calculate_mean_for_each_cluster(self,
                                         inputs: List[Vector],
                                         assignments: List[int]):
        """
        KMeans remembers the mean of the samples in each cluster. These are called "centroids".

        :param inputs: the input data
        :param assignments: which cluster each input belongs to
        """
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

    def save_intermediate_state(self, assignments):
        """
        In case we want to store each step given by the algorithm for further analysis.

        :param assignments: The current state for each cluster
        """
        new_state = {
            "state": assignments,
            "centroids": self.clusters_mean
        }
        self.intermediate_steps.append(new_state)
