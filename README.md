# :robot: KMeans
KMeans is a popular **unsupervised machine learning algorithm** used for **clustering** data points into groups or clusters based on their similarity. Building your own version of KMeans is an excellent way to deepen your understanding of the algorithm and improve your skills in machine learning.

## :brain: Algorithm (traning)
The algorithm follows these _steps_:
1. **Define the number of clusters**: KMeans requires you to specify the number of clusters in advance. You can choose any number of clusters based on your data and problem domain.
2. **Initialize centroids**: _Choose K points at random_ from the dataset to serve as the initial centroids for each cluster.
3. **Assign data points to clusters**: For each data point, calculate its _distance from each centroid and assign it to the closest cluster_.
4. **Update centroids**: Calculate the mean of all the points assigned to each cluster and set the centroid of that cluster to the mean.
5. **Repeat steps 3-4 until convergence**: Keep repeating steps 3-4 until the centroids stop changing or a maximum number of iterations is reached.

## :heavy_check_mark: Metrics 
After the execution we can evaluate the quality of the clusters using metrics such as **within-cluster sum of squares** or **silhouette score**.

## :mechanical_arm: Improvements
You can also consider adding some variations to the basic KMeans algorithm, such as **using different distance metrics**, using more **advanced initialization methods**, or **implementing parallelization for faster performance** on large datasets.

## 👩‍🏫 Example
[Notebook](http://localhost:8888/notebooks/notebooks/example_delivery_company.ipynb) with a little example.
