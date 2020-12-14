#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:41:48 2020

@author: root
"""

### Needed imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import seaborn as sns
from sklearn.metrics.cluster import contingency_matrix


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



### Obtaining data
raw = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
original_labels_raw = raw["NObeyesdad"]
original_labels = []
data = raw.drop(["NObeyesdad"], axis=1)
samples = data.shape[0]
attributes = data.shape[1]
print("Number of records: ", samples)
print("Number of attributes: ", attributes)



### Data preprocess
sex_dict = dict()
sex_dict["Female"] = 0
sex_dict["Male"] = 1

yesno_dict = dict()
yesno_dict["no"] = 0
yesno_dict["yes"] = 1

caec_and_calc_dict = dict()
caec_and_calc_dict["no"] = 0
caec_and_calc_dict["Sometimes"] = 1
caec_and_calc_dict["Frequently"] = 2
caec_and_calc_dict["Always"] = 3

mtrans_dict = dict()
mtrans_dict["Automobile"] = 0
mtrans_dict["Motorbike"] = 1
mtrans_dict["Bike"] = 2
mtrans_dict["Public_Transportation"] = 3
mtrans_dict["Walking"] = 4

obesity_dict = dict()
obesity_dict["Insufficient_Weight"] = 0
obesity_dict["Normal_Weight"] = 1
obesity_dict["Overweight_Level_I"] = 2
obesity_dict["Overweight_Level_II"] = 3
obesity_dict["Obesity_Type_I"] = 4
obesity_dict["Obesity_Type_II"] = 5
obesity_dict["Obesity_Type_III"] = 6

for i in range(samples):
    data.at[i, "Gender"] = sex_dict[data.at[i, "Gender"]]
    data.at[i, "family_history_with_overweight"] = yesno_dict[data.at[i, "family_history_with_overweight"]]
    data.at[i, "FAVC"] = yesno_dict[data.at[i, "FAVC"]]
    data.at[i, "CAEC"] = caec_and_calc_dict[data.at[i, "CAEC"]]
    data.at[i, "SMOKE"] = yesno_dict[data.at[i, "SMOKE"]]
    data.at[i, "SCC"] = yesno_dict[data.at[i, "SCC"]]
    data.at[i, "CALC"] = caec_and_calc_dict[data.at[i, "CALC"]]
    data.at[i, "MTRANS"] = mtrans_dict[data.at[i, "MTRANS"]]

for i in original_labels_raw:
    original_labels.append(obesity_dict[i])
    
data["Gender"] = data["Gender"].astype("int")
data["family_history_with_overweight"] = data["family_history_with_overweight"].astype("int")
data["FAVC"] = data["FAVC"].astype("int")
data["CAEC"] = data["CAEC"].astype("int")
data["SMOKE"] = data["SMOKE"].astype("int")
data["SCC"] = data["SCC"].astype("int")
data["CALC"] = data["CALC"].astype("int")
data["MTRANS"] = data["MTRANS"].astype("int")



### PCA with original components

# Needed computations
pca = PCA()
pca.fit(data)
obesity_pc = pca.transform(data)

# Visualizing of clustering in the principal components space
fig = plt.figure(1)
plt.title('Obesity data after PCA with original labels')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(obesity_pc[:,0],obesity_pc[:,1],s=50,c=original_labels)
plt.savefig("PCA_diagram_original_labels.png")
plt.close()



####### KMEANS
### Creating ideal number of clusters with KMeans

# Needed variables
kmeans_min_DB = 1.00
kmeans_ideal_clusters = 0
kmeans_ideal_centers = []
kmeans_ideal_labels = []
kmeans_ideal_score = 0.00

# Finding optimal cluster number with kmeans
SSE_kmeans = np.zeros((29))
DB_kmeans = np.zeros((29))
for i in range(29):
    n_c = i+2
    kmeans = KMeans(n_clusters=n_c, random_state=2020)
    kmeans.fit(data)
    kmeans_labels = kmeans.labels_
    SSE_kmeans[i] = kmeans.inertia_
    DB_kmeans[i] = davies_bouldin_score(data,kmeans_labels)
    if DB_kmeans[i] < kmeans_min_DB:
        kmeans_min_DB = DB_kmeans[i]
        kmeans_ideal_clusters = n_c
        kmeans_ideal_centers = kmeans.cluster_centers_
        kmeans_ideal_labels = kmeans_labels
        kmeans_ideal_score = kmeans_min_DB
        
print("Best number of clusters with KMeans: ", kmeans_ideal_clusters)



### Visualization of SSE values    
fig = plt.figure(2)
plt.title('Sum of squares of error curve (KMeans)')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.plot(np.arange(2,31),SSE_kmeans, color='red')
plt.savefig("sse_diagram_kmeans.png")
plt.close()


### Visualization of DB scores
fig = plt.figure(3)
plt.title('Davies-Bouldin score curve (KMeans)')
plt.xlabel('Number of clusters')
plt.ylabel('DB index')
plt.plot(np.arange(2,31),DB_kmeans, color='blue')
plt.savefig("db_kmeans_diagram.png")
plt.close()

    
### PCA with limited components
#calculating centers
kmeans_centers_pc = pca.transform(kmeans_ideal_centers)

# Visualizing of clustering in the principal components space
fig = plt.figure(4)
plt.title('Clustering of the Obesity data after PCA (KMeans)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(obesity_pc[:,0],obesity_pc[:,1],s=50,c=kmeans_ideal_labels)
plt.scatter(kmeans_centers_pc[:,0],kmeans_centers_pc[:,1],s=200,c='red',marker='X')
plt.savefig("PCA_diagram_kmeans.png")
plt.close()

# Kmeans clustering with idealcluster numbers based on DB score
kmeans = KMeans(n_clusters=kmeans_ideal_clusters, random_state=2020)
kmeans.fit(data)
kmeans_labels = kmeans.labels_
kmeans_centers = kmeans.cluster_centers_
kmeans_distX = kmeans.transform(data)
kmeans_dist_center = kmeans.transform(kmeans_centers)

# Visualizing of clustering in the distance space
fig = plt.figure(5)
plt.title('Obesity data in the distance space (KMeans)')
plt.xlabel('Cluster 1')
plt.ylabel('Cluster 2')
plt.scatter(kmeans_distX[:,0],kmeans_distX[:,1],s=50,c=kmeans_labels)
plt.scatter(kmeans_dist_center[:,0],kmeans_dist_center[:,1],s=200,c='red',marker='X')
plt.savefig("dist_space_diagram_kmeans.png")
plt.close()

####### KMEANS END






####### SINGLE AGGLOMERATIVE


### Agglomerative with single linkage
# Building the full tree
single_cluster = AgglomerativeClustering(distance_threshold=0, 
                            n_clusters=None,linkage='single')
single_cluster.fit(data)

# Plot the top p levels of the dendrogram
fig = plt.figure(6)
plt.title('Hierarchical Clustering Dendrogram (single linkage)')
plot_dendrogram(single_cluster, truncate_mode='level', p=4)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.savefig("dendogram_single.png")
plt.close()

single_min_db = 1.00
single_min_cluster_numbers = 0
single_min_labels = []
DB_single = np.zeros((29))

for i in range(29):
    single_cluster = AgglomerativeClustering(n_clusters=i+2,linkage='single')
    single_cluster.fit(data)
    ypred_single = single_cluster.labels_
    db_single = davies_bouldin_score(data,ypred_single)
    DB_single[i] = db_single
    if db_single < single_min_db:
        single_min_db = db_single
        single_min_cluster_numbers = i+2
        single_min_labels = single_cluster.labels_
        single_ideal_score = db_single
        
print("Best number of clusters with single-Agglomerative: ", single_min_cluster_numbers)


### Visualization of DB scores
fig = plt.figure(7)
plt.title('Davies-Bouldin score curve with single linkage agglomerative method')
plt.xlabel('Number of clusters')
plt.ylabel('DB index')
plt.plot(np.arange(2,31),DB_single, color='blue')
plt.savefig("db_single_diagram.png")
plt.close()

# Visualizing datapoints using cluster label of single agglomerative method
fig = plt.figure(8)
plt.title('Scatterplot of datapoints with single linkage clustering')
plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(obesity_pc[:,0],obesity_pc[:,1],s=50,c=single_min_labels)
plt.savefig("PCA_diagram_single.png")
plt.close()


####### SINGLE AGGLOMERATIVE END





####### COMPLETE AGGLOMERATIVE

### Agglomerative with complete linkage
# Building the full tree
complete_cluster = AgglomerativeClustering(distance_threshold=0, 
                            n_clusters=None,linkage='complete')
complete_cluster.fit(data)

# Plot the top p levels of the dendrogram
fig = plt.figure(9)
plt.title('Hierarchical Clustering Dendrogram (complete linkage)')
plot_dendrogram(complete_cluster, truncate_mode='level', p=4)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.savefig("dendogram_complete.png")
plt.close()

complete_min_db = 1.00
complete_min_cluster_numbers = 0
complete_min_labels = []
DB_complete = np.zeros((29))

for i in range(29):
    complete_cluster = AgglomerativeClustering(n_clusters=i+2,linkage='complete')
    complete_cluster.fit(data)
    ypred_complete = complete_cluster.labels_
    db_complete = davies_bouldin_score(data,ypred_complete)
    DB_complete[i] = db_complete
    if db_complete < complete_min_db:
        complete_min_db = db_complete
        complete_min_cluster_numbers = i+2
        complete_min_labels = complete_cluster.labels_
        complete_ideal_score = db_complete
        
print("Best number of clusters with complete-Agglomerative: ", complete_min_cluster_numbers)


### Visualization of DB scores
fig = plt.figure(10)
plt.title('Davies-Bouldin score curve with complete linkage agglomerative method')
plt.xlabel('Number of clusters')
plt.ylabel('DB index')
plt.plot(np.arange(2,31),DB_complete, color='blue')
plt.savefig("db_complete_diagram.png")
plt.close()

# Visualizing datapoints using cluster label of complete agglomerative method
fig = plt.figure(11)
plt.title('Scatterplot of datapoints with complete linkage clustering')
plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(obesity_pc[:,0],obesity_pc[:,1],s=50,c=complete_min_labels)
plt.savefig("PCA_diagram_complete.png")
plt.close()

####### COMPLETE AGGLOMERATIVE END






####### WARD AGGLOMERATIVE

### Agglomerative with ward linkage
# Building the full tree
ward_cluster = AgglomerativeClustering(distance_threshold=0, 
                            n_clusters=None,linkage='ward')
ward_cluster.fit(data)

# Plot the top p levels of the dendrogram
fig = plt.figure(12)
plt.title('Hierarchical Clustering Dendrogram (ward linkage)')
plot_dendrogram(ward_cluster, truncate_mode='level', p=4)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.savefig("dendogram_ward.png")
plt.close()

ward_min_db = 1.00
ward_min_cluster_numbers = 0
ward_min_labels = []
DB_ward = np.zeros((29))

for i in range(29):
    ward_cluster = AgglomerativeClustering(n_clusters=i+2,linkage='ward')
    ward_cluster.fit(data)
    ypred_ward = ward_cluster.labels_
    db_ward = davies_bouldin_score(data,ypred_ward)
    DB_ward[i] = db_ward
    if db_ward < ward_min_db:
        ward_min_db = db_ward
        ward_min_cluster_numbers = i+2
        ward_min_labels = ward_cluster.labels_
        ward_ideal_score = db_ward
        
print("Best number of clusters with ward-Agglomerative: ", ward_min_cluster_numbers)


### Visualization of DB scores
fig = plt.figure(13)
plt.title('Davies-Bouldin score curve with ward linkage agglomerative method')
plt.xlabel('Number of clusters')
plt.ylabel('DB index')
plt.plot(np.arange(2,31),DB_ward, color='blue')
plt.savefig("db_ward_diagram.png")
plt.close()

# Visualizing datapoints using cluster label of ward agglomerative method
fig = plt.figure(14)
plt.title('Scatterplot of datapoints with ward linkage clustering')
plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(obesity_pc[:,0],obesity_pc[:,1],s=50,c=ward_min_labels)
plt.savefig("PCA_diagram_ward.png")
plt.close()

####### WARD AGGLOMERATIVE END





####### DBSCAN

# DBSCAN clustering
radius = 0.00
inner_points = 5
dbscan_score = 300
best_radius = 0.00
dbscan_labels = []
x_values_to_plot = []
DB_dbscan = np.zeros((150))
for i in range(150):
    radius += 0.01
    radius = float("{:.2f}".format(radius)) # correcting floating point number to be double precisioned
    x_values_to_plot.append(radius)
    dbscan_cluster = DBSCAN(eps=radius, min_samples=inner_points)
    dbscan_cluster.fit(data)
    ypred_dbscan = dbscan_cluster.labels_
    score = davies_bouldin_score(data,ypred_dbscan)
    DB_dbscan[i] = score
    if score < dbscan_score:
        dbscan_score = score
        dbscan_labels = ypred_dbscan
        best_radius = radius
        dbscan_ideal_score = score
                
print("Best number of clusters with DBSCAN: ", len(set(dbscan_labels)))

### Visualization of DB scores
fig = plt.figure(15)
plt.title('Davies-Bouldin score curve with dbscan method (searching for ideal radius)')
plt.xlabel('Number of clusters')
plt.ylabel('DB index')
plt.plot(x_values_to_plot,DB_dbscan, color='blue')
plt.savefig("db_dbscan_diagram.png")
plt.close()

# Visualizing datapoints using cluster label
fig = plt.figure(16)
plt.title('Scatterplot of datapoints with DBSCAN')
plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(obesity_pc[:,0],obesity_pc[:,1],s=50,c=dbscan_labels)
plt.savefig("PCA_diagram_dbscan.png")
plt.close()

####### DBSCAN END



####### COMPARING DAVIES-BOULDIN SCORES

plot_kmeans_score = pd.DataFrame({'type':"KMeans",'score':[kmeans_ideal_score]})
plot_single_score = pd.DataFrame({'type':"Single",'score':[single_ideal_score]})
plot_complete_score = pd.DataFrame({'type':"Complete",'score':[complete_ideal_score]})
plot_ward_score = pd.DataFrame({'type':"Ward",'score':[ward_ideal_score]})
plot_dbscan_score = pd.DataFrame({'type':"DBSCAN",'score':[dbscan_ideal_score]})
dataf = pd.concat([plot_kmeans_score, plot_single_score, plot_complete_score, plot_ward_score, plot_dbscan_score])
splot=sns.barplot(data=dataf, x="type", y="score", ci=None, palette=["blue", "orange", "green", "red", "pink"])
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.5f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha="center", va="center",
                   xytext=(0, 9),
                   textcoords = "offset points")
plt.title("DB scores (the lower the better)", fontsize=16)
plt.savefig("db_scores_comparison.png")
plt.close()