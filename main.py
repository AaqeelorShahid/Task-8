import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import dataset 
dataset = pd.read_csv('UK.csv')
X = dataset.iloc[:, [1, 2]].values

# Using Elbow method to find number of Clusters 
from sklearn.cluster import KMeans

# Applying K-Means to dataset
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Inserting Cluster data into dataset
dataset_cluster = dataset.copy()
dataset_cluster ['cluster'] = y_kmeans

#Splitting data for testing and trainning - 25% of data for testing
X = dataset_cluster.iloc[:, [1, 2]].values
y = dataset_cluster.iloc[:, 4].values #Kmeans output 
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size= 0.25, random_state=0)

# Tree classification with Random Forest Classifer - Good Accuracy
rf = RandomForestClassifier(n_estimators=100,
                            random_state=0)

# Training the model on the data
rf.fit(X_train, Y_train)

# Visulizing data in tree viewer
fn = dataset.iloc[:, 0].values
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=1000)
tree.plot_tree(rf.estimators_[0],
               feature_names=fn,
               filled=True)
fig.savefig("decision_tree.png")
