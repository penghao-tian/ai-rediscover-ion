"""
@Author: Penghao Tian <tph@ustc.edu.cn>
@Date: 2025/06/27 22:16
@Description: 
Perform clustering primarily using machine learning algorithms in sklearn

MiniBatch KMeans: 
Implement clustering models for s4 profiles
Include data preprocessing, model construction with different hyperparameters, and model optimization, 
and provide clustering results that can be inspected in real time

drop_small_clusters:
Some clusters contain obvious noise, which can be identified using this unsupervised method
and then removed to produce a cleaner dataset for subsequent training and analysis

# The main function runs two algorithms: minibatch and kmeans
"""


import os
import sys
import joblib
import scipy
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

from datetime import datetime
import time

import argparse


from pathlib import Path

# from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans, KMeans


class KMeans_Clustering_Pipeline():
    """
    Perform clustering analysis directly using Kmeans
    No longer use the cluster partitioning approach
    Kmeans has high computational complexity but remains effective for tens of millions of observations
    """

    def __init__(self, data,
                 n_clusters_final=25,
                 random_state=42,
                 output_dir="./clustering_model"):
        self.n_clusters_final = n_clusters_final
        self.random_state = random_state
        self.output_dir=output_dir
        self.X = None
        self.final_clustering = None
        self.final_labels = None
        # The random seed may affect other classifications
        # np.random.seed(self.random_state)
        
        self.load_data(data)
        pass
    
    def load_data(self, data):
        assert data.ndim == 2
        # data = data[:10000]
        self.X = data
        print(self.X.shape)
        print("Data loaded and reshaped.")
        pass
    
    def perform_clustering(self,):
        print("Performing Kmeans clustering...")
        self.final_labels = KMeans(n_clusters=self.n_clusters_final,
                                   n_init="auto",
                                   random_state=self.random_state).fit_predict(self.X)
        pass
    
    def save_model(self,):
        
        joblib.dump(self.final_labels, 
                    os.path.join(self.output_dir, 
                                 f"kmeans_final_labels_{self.n_clusters_final}_{self.random_state}.pkl"))
        
        print("Models and labels saved to disk.")
        pass


class MiniBatchKMeansClusteringPipeline():
    """
    Load the corresponding file contents and select all s4 index profile data
    COSMIC-1/2 contain approximately 14 million observations, making direct hierarchical clustering impractical
    First select 25,000 cluster centers and assign the 14 million observations to these 25,000 clusters,
    obtaining 25,000 cluster centers that represent most cluster characteristics; then perform hierarchical clustering on these 25,000 clusters
    Generate 25 labels, thereby mapping the 14 million observations to 25 classes
    
    data: Array to cluster (sample_size, feature_size)
    n_clusters_initial: Number of initial cluster centers, 25000
    n_clusters_final: Number of final labels, 25
    random_state: Random seed
    output_dir: Directory for saving the clustering models and final labels
    """
    def __init__(self, data, 
                 n_clusters_initial=25000, n_clusters_final=25, 
                 random_state=42,
                 output_dir="./clustering_model"):
        self.n_clusters_initial = n_clusters_initial
        self.n_clusters_final = n_clusters_final
        self.random_state = random_state
        self.output_dir=output_dir
        self.X = None
        self.initial_clustering = None
        self.final_clustering = None
        self.final_labels = None
        # The random seed may affect other classifications
        # np.random.seed(self.random_state)
        
        self.load_data(data)
        pass

    def load_data(self, data):
        # First select a limited subset of data for fitting
        # data = data[:100000, :]
        assert data.ndim == 2
        self.X = data
        print(self.X.shape)
        print("Data loaded and reshaped.")

    def perform_clustering(self):
        print("Performing MiniBatch clustering...")
        self.initial_clustering = MiniBatchKMeans(n_init="auto",
                                                  n_clusters=self.n_clusters_initial,
                                                  random_state=self.random_state,
                                                  )
        
        # Perform the initial fit with MiniBatchKMeans to obtain the cluster centers
        # breakpoint()
        initial_cluster_labels = self.initial_clustering.fit_predict(self.X)
        initial_cluster_centers = self.initial_clustering.cluster_centers_
        print("MiniBatch clustering completed.")

        # Use hierarchical clustering to classify the existing cluster centers without labels
        print("Performing final hierarchical clustering...")
        self.final_clustering = AgglomerativeClustering(n_clusters=self.n_clusters_final,
                                                        linkage="ward",
                                                        compute_distances=True)
        
        final_cluster_labels = self.final_clustering.fit_predict(initial_cluster_centers)
        self.final_labels = final_cluster_labels[initial_cluster_labels]
        
        print("Final clustering completed.")
        
    def save_model(self):
        """
        initial_clustering_*: Model obtained after the initial minibatchkmeans clustering of the data
        final_clustering_*: Hierarchical clustering model storing parameters that can be used to plot a dendrogram
        final_labels_*: Final labels, with the same size as the original input data, storing class labels from 0 to N
        
        """
        
        # joblib.dump(self.initial_clustering, os.path.join(self.output_dir, f"minibatch_initial_clustering_{self.n_clusters_initial}_{self.n_clusters_final}_{self.random_state}.pkl"))
        
        joblib.dump(self.final_clustering, os.path.join(self.output_dir, f"minibatch_final_clustering_{self.n_clusters_initial}_{self.n_clusters_final}_{self.random_state}.pkl"))
        
        joblib.dump(self.final_labels, os.path.join(self.output_dir, f"minibatch_final_labels_{self.n_clusters_initial}_{self.n_clusters_final}_{self.random_state}.pkl"))
        
        print("Models and labels saved to disk.")

    def visualize_cluster_distribution(self, output_path="./cluster_distribution.png"):
        print("Visualizing cluster distribution...")
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.countplot(x=self.final_labels)
        plt.title("Final Cluster Distribution")
        plt.xlabel("Cluster Label")
        plt.ylabel("Number of Samples")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Cluster distribution plot saved to {output_path}.")

        pass
    pass


    
def minibatchkmeans_functor(data, n_clusters_initial, n_clusters_final,
                            random_state, output_dir):
    
    # Use the minibatch algorithm
    pipeline = MiniBatchKMeansClusteringPipeline(data=data, 
                                                 n_clusters_initial=n_clusters_initial,
                                                 n_clusters_final=n_clusters_final, 
                                                 random_state=random_state,
                                                 output_dir=output_dir)
    
    pipeline.perform_clustering()
    pipeline.save_model()
    
    # Visualization functionality is currently unused
    # pipeline.visualize_cluster_distribution()
    
    
def subroutine_minibatchkmeans(data, output_dir):
    
    # Set the hyperparameter arrays and iterate over them
    
    n_clusters_initial_array = [20000, 25000, 30000, 40000, 50000]
    n_clusters_final_array = [20, 25, 30, 36, 50, 100]
    
    # n_clusters_initial_array = [50000]
    # n_clusters_final_array = [50,100]
    
    random_state_array = [42]
    
    for n_clusters_initial in n_clusters_initial_array:
        for n_clusters_final in n_clusters_final_array:
            for random_state in random_state_array:
                minibatchkmeans_functor(data,
                                        n_clusters_initial, 
                                        n_clusters_final, random_state,
                                        output_dir)
                pass
            pass
        pass
    pass


def kmeans_functor(data, n_clusters_final, random_state,
                   output_dir):
    # Run the Kmeans clustering algorithm
    # The cosmic1 data file contains no NaNs; cosmic2 contains NaNs; the processed ensemble contains no NaNs
    
    
    pipeline = KMeans_Clustering_Pipeline(data=data,
                                          n_clusters_final=n_clusters_final, 
                                          random_state=random_state,
                                          output_dir=output_dir)
    
    
    pipeline.perform_clustering()
    pipeline.save_model()
    
    pass

def subroutine_kmeans(data, output_dir):
    
    n_clusters_final_array = [20, 25, 30, 36, 50, 100]
    # n_clusters_final_array = [25]
    
    random_state_array = [42]
    
    
    for n_clusters_final in n_clusters_final_array:
        for random_state in random_state_array:
            kmeans_functor(data=data, n_clusters_final=n_clusters_final, 
                           random_state=random_state,
                           output_dir=output_dir)
            pass
        pass
    pass


def get_bash_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="cosmic_1", type=str, help="define the used data")
    parser.add_argument("--type", default="kmeans", type=str, help="define the basic routine")
    return parser.parse_args()


def main():
    
    # Retrieve bash arguments
    bash_args = get_bash_args()
    
    if bash_args.data_source == "cosmic_1":
        file_path = "./data/cosmic1_s4_2007_2019_process_snr.nc"
        output_dir="./clustering_model/cosmic_1"
        pass
    elif bash_args.data_source == "cosmic_2":
        file_path = "./data/cosmic2_s4_2019_2025_process_snr.nc"
        output_dir="./clustering_model/cosmic_2"
        pass
    elif bash_args.data_source == "cosmic_1_2":
        file_path = "./data/cosmic_1_2_s4_2007_2025_process_snr_filter_4.nc"
        output_dir="./clustering_model/cosmic_1_2"
        pass
    
    variable = "s4"
    ds = xr.open_dataset(file_path)
    ds.close()
    s4_data = ds[variable].to_numpy()
    
    if bash_args.type == "kmeans":
        subroutine_kmeans(s4_data, output_dir)
        pass
    elif bash_args.type == "minibatchkmeans":
        subroutine_minibatchkmeans(s4_data, output_dir)
    pass


def drop_small_clusters(final_labels, in_nc_path, out_nc_path, min_size=1000, time_dim="n_time"):
    """
    After initial processing, removing NaNs, and removing pairs of consecutive similar snr files, a small number of files with highly irregular s4 values remain
    Use clustering here as an initial step to identify and remove them 
    
    Remove data belonging to clusters in final_labels with sample counts < min_size, and save as a new NetCDF file.
    Only perform: find small clusters -> select complementary indices -> isel -> to_netcdf
    """
    labels = np.asarray(final_labels)
    with xr.open_dataset(Path(in_nc_path)) as ds:
        assert time_dim in ds.dims, f"Time dimension {time_dim} not found"
        assert ds.dims[time_dim] == labels.shape[0], "The length of labels does not match the time dimension"

        # Count cluster sizes and identify small clusters
        uniq, cnts = np.unique(labels, return_counts=True)
        small = set(uniq[cnts < min_size])

        # Select the complement: retain sample indices that do not belong to small clusters
        keep_idx = np.where(~np.isin(labels, list(small)))[0]

        # Select and save
        ds_sel = ds.isel({time_dim: keep_idx})
        ds_sel.to_netcdf(Path(out_nc_path))
        pass
    return ds_sel

    
if __name__=="__main__":
    print(f"Start script in {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("Script start")
    t1 = time.time()
    
    main()
    
    
    t2 = time.time()
    print(f"Data pre-process has completed in {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Wall time is {t2-t1}")
    pass

# Example usage:
# python s4_profile_clustering.py --data_source=cosmic_1 --type=kmeans

