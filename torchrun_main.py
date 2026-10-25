"""
@Author: Penghao Tian <tph@ustc.edu.cn>
@Date: 2025/07/27 21:52
@Description:
Implement the overall training workflow
train_encode_decode_routin: Train a DL model, not limited to an encoder-decoder architecture
evaluate_encode_decode_routine: Evaluate the trained DL model 
get_cluster_center_routine: Obtain intermediate compressed variables from the trained DL model and compute cluster centers 
train_encode_cluster_routine: Jointly train the DEC model using the computed cluster centers
evaluate_encode_cluster_routine: Evaluate the DEC model

This project no longer uses the DEC model; instead, it uses DL to compress variables, followed by minibatch kmeans + hierarchical clustering
Failed DEC training is one reason for this change

Use torchrun and python in auto_run.sh to launch multiple or single processes for computation
"""


import os
import sys
import time
import torch
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
import configparser
from pathlib import Path
from datetime import datetime
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

import matplotlib.pyplot as plt
import scipy.stats

import joblib

from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE

import train

# Too many objects are named model in the code; rename them to avoid confusion
import model as define_module
import evaluate
from s4_profile_clustering import MiniBatchKMeansClusteringPipeline, KMeans_Clustering_Pipeline



def global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    return

def ddp_setup_cpu():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12356"
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    init_process_group(backend="gloo")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )
    
    # init_process_group(backend="nccl")
                       # init_method="tcp://localhost:12355")
                       # init_method=f"file://{Path().cwd().joinpath('sharedfile').resolve()}")
    # print(f"Initialize process group complete # GPU::{rank} ")
    return

def ddp_setup():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12356"
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )
    
    # init_process_group(backend="nccl")
                       # init_method="tcp://localhost:12355")
                       # init_method=f"file://{Path().cwd().joinpath('sharedfile').resolve()}")
    # print(f"Initialize process group complete # GPU::{rank} ")
    return


def train_encode_decode_routine(args_config):
    # Train the corresponding encoder-decoder model
    
    # Create the communication group
    ddp_setup()
    # ddp_setup_cpu()
    
    # Start the training workflow
    # Define the dataset
    dataloader = train.PrepareDataLoader(dataset_path=args_config.get("train_params", "dataset_path"),
                                         process_method=args_config.getint("train_params", "process_method"),
                                         transform_method=args_config.getint("train_params", "transform_method"),
                                         return_type=args_config.get("train_params", "return_type"),
                                         split_method=args_config.getint("train_params", "split_method"),
                                         batch_size=args_config.getint("train_params", "batch_size"),
                                         num_workers=args_config.getint("train_params", "num_workers")).prepare_dataloader()
    
    # Define the complete model
    model = train.DefineModel(model_name=args_config.get("train_params", "model_name"), 
                              feature_size=args_config.getint("train_params", "feature_size"),
                              cluster_center=None,
                              load_model_params_file=args_config.get("train_params", "load_model_params_file")
                             ).model
    
    # Define the training class
    trainer = train.Trainer(dataloader=dataloader,
                            model_name=args_config.get("train_params", "model_name"),
                            model=model,
                            gpu_id=dist.get_rank(),
                            loss_func_name=args_config.get("train_params", "loss_func_name"),
                            optimizer_name=args_config.get("train_params", "optimizer_name"),
                            lr_scheduler_name=args_config.get("train_params", "lr_scheduler_name"),
                            lr=args_config.getfloat("train_params", "lr"),
                            max_epoch=args_config.getint("train_params", "max_epoch"),
                            save_model_dir=args_config.get("train_params", "save_model_dir"),
                            save_log_file=args_config.get("train_params", "save_log_file"),
                            args_config=args_config)
    
    trainer.train()
    
    # Destroy the communication group
    destroy_process_group()
    return 

def evaluate_encode_decode_routine(args_config):
    # Evaluate the effectiveness of the encoder-decoder in reconstructing the data
    
    # Define the dataset using the methods in evaluate to include all data
    dataloader = evaluate.PrepareDataLoader(return_type="all",
                                            dataset_path=args_config.get("train_params", "dataset_path"),
                                            process_method=args_config.getint("train_params", "process_method"),
                                            transform_method=None,
                                            batch_size=1024,
                                            num_workers=1).prepare_dataloader()
    
    # Define the model; training and evaluation use the same model, so the methods in train can be used directly
    # Also load the existing model parameter file
    def find_params_file():
        # Use the log file recorded during training to find the file with the best train_loss 
        log_file=args_config.get("train_params", "save_log_file")
        file_df = pd.read_csv(log_file)
        idx_min = file_df.loc[:, "train_loss"].idxmin()
        model_params_name = file_df.iloc[idx_min].loc["model_name"]
        model_params_name = Path(model_params_name)
        if not model_params_name.is_file():
            print(f'cannot find the best model params file in {args_config.get("train_params", "save_model_dir")}')
            raise
            pass
        return model_params_name
    
    model = train.DefineModel(model_name=args_config.get("train_params", "model_name"), 
                              feature_size=args_config.getint("train_params", "feature_size"),
                              cluster_center=None,
                              load_model_params_file=find_params_file()).model
    
    evaluator = evaluate.Evaluator(dataloader=dataloader,
                                   model=model,
                                   args_config=args_config)
    
    output_array, target_array = evaluator.evaluate()
    
    # Determine whether the two have consistent distributions
    metrics = evaluate_profiles(output_array, target_array)
    for key, val in metrics.items():
        print(f"{key}: {val}")
        pass
    
    pass

def evaluate_profiles(pred: np.ndarray, label: np.ndarray, plot_sample_index: int = 0):
    """
    Evaluate predicted and true profiles using regression metrics and visualisation.
    
    Args:
        pred (np.ndarray): shape (N, H) - predicted profiles
        label (np.ndarray): shape (N, H) - true profiles
        plot_sample_index (int): index of the sample to plot

    Returns:
        dict: evaluation results including global and per-sample metrics
    """
    assert pred.shape == label.shape, "Prediction and label must have the same shape"
    
    N, H = pred.shape

    # -------- Global Metrics --------
    mae_all = np.mean(np.abs(pred - label))
    rmse_all = np.sqrt(np.mean((pred - label) ** 2))
    # ss_res = np.sum((pred - label) ** 2)
    # ss_tot = np.sum((label - np.mean(label)) ** 2)
    # r2_all = 1 - ss_res / ss_tot

    # -------- Sample-wise Metrics --------
    mae_per_sample = np.mean(np.abs(pred - label), axis=1)
    rmse_per_sample = np.sqrt(np.mean((pred - label) ** 2, axis=1))
    # r2_per_sample = 1 - np.sum((pred - label) ** 2, axis=1) / np.sum((label - np.mean(label, axis=1, keepdims=True)) ** 2, axis=1)

    # -------- Spearman Correlation (per sample) --------
    corrs = []
    for i in range(N):
        corr, _ = scipy.stats.spearmanr(pred[i], label[i])
        corrs.append(corr)
        if i >= 1e7:
            break
            pass
        pass
    mean_corr = np.mean(corrs)

    # -------- Height-wise MAE --------
    mae_per_height = np.mean(np.abs(pred - label), axis=0)

    # -------- Plot 1: Sample Profile --------
    plt.figure()
    plt.plot(label[plot_sample_index], np.arange(H), label='True')
    plt.plot(pred[plot_sample_index], np.arange(H), label='Predicted')
    plt.gca().invert_yaxis()
    plt.xlabel("Signal")
    plt.ylabel("Height Index")
    plt.title(f"Sample #{plot_sample_index} Profile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"./test1_{plot_sample_index} profile")

    # -------- Plot 2: MAE vs Height --------
    plt.figure()
    plt.plot(mae_per_height, np.arange(H))
    plt.gca().invert_yaxis()
    plt.xlabel("MAE")
    plt.ylabel("Height Index")
    plt.title("Mean Absolute Error vs Height")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"./test2_{plot_sample_index} profile")

    # -------- Return Results --------
    return {
        "MAE_all": mae_all,
        "RMSE_all": rmse_all,
        # "R2_all": r2_all,
        "MAE_mean": np.mean(mae_per_sample),
        "RMSE_mean": np.mean(rmse_per_sample),
        # "R2_mean": np.mean(r2_per_sample),
        "Spearman_mean": mean_corr,
        "MAE_per_height": mae_per_height  # shape (H,)
    }

def get_cluster_center_routine(args_config):
    # Based on the existing encoder and kmeans, 
    # generate cluster centers and save them to the corresponding file
    # Most steps are the same as in evaluation
    
    # Define the dataset using the methods in evaluate to include all data
    dataloader = evaluate.PrepareDataLoader(return_type="all",
                                            dataset_path=args_config.get("train_params", "dataset_path"),
                                            process_method=args_config.getint("train_params", "process_method"),
                                            transform_method=None,
                                            batch_size=1024,
                                            num_workers=1).prepare_dataloader()
    
    # Define the model; training and evaluation use the same model, so the methods in train can be used directly
    # Also load the existing model parameter file
    def find_params_file():
        # Use the log file recorded during training to find the file with the best train_loss 
        log_file=args_config.get("train_params", "save_log_file")
        file_df = pd.read_csv(log_file)
        idx_min = file_df.loc[:, "train_loss"].idxmin()
        model_params_name = file_df.iloc[idx_min].loc["model_name"]
        model_params_name = Path(model_params_name)
        if not model_params_name.is_file():
            print(f'cannot find the best model params file in {args_config.get("train_params", "save_model_dir")}')
            raise
            pass
        return model_params_name
    
    # Define the corresponding model; the model loaded here is an encoder-decoder
    load_model_params_file = find_params_file()
    model = train.DefineModel(model_name=args_config.get("train_params", "model_name"), 
                              feature_size=args_config.getint("train_params", "feature_size"),
                              cluster_center=None,
                              load_model_params_file=load_model_params_file).model
    
    # Select the corresponding encoder module
    # model is encode_decode
    # model = model.encode
    
    # model is self_ann
    empty_model = nn.Sequential()
    model.decode = empty_model
    
    # Infer all corresponding latent features
    evaluator = evaluate.Evaluator(dataloader=dataloader,
                                   model=model,
                                   args_config=args_config)
    output_array, _ = evaluator.evaluate()
    
    # Perform MiniBatchKmeans clustering on the features in latent feature space
    # pipeline = MiniBatchKMeansClusteringPipeline(data=output_array, 
    #                                              n_clusters_initial=args_config.getint("cluster_params",
    #                                                                                    "n_clusters_initial"),
    #                                              n_clusters_final=args_config.getint("cluster_params", 
    #                                                                                  "n_clusters_final"),
    #                                              random_state=args_config.getint("train_params", "seed"),
    #                                              output_dir=args_config.get("train_params", "save_model_dir"))
    
    # Perform Kmeans clustering on the features in latent feature space
    pipeline = KMeans_Clustering_Pipeline(data=output_array,
                                          n_clusters_final=args_config.getint("cluster_params",
                                                                              "n_clusters_final"),
                                          random_state=args_config.getint("train_params",
                                                                          "seed"),
                                          output_dir=args_config.get("train_params",
                                                                     "save_model_dir"))
    pipeline.perform_clustering()
    pipeline.save_model()
    
#     kmeans = KMeans(n_clusters=args_config.getint("train_dec_params", "cluster_number"),
#                     n_init="auto",
#                     random_state=args_config.getint("train_params", "seed")).fit(output_array)
#     center = kmeans.cluster_centers_
#     label = kmeans.labels_
    
#     # Save the corresponding center file as pkl
#     cluster_center_name = "cluster_center_" + load_model_params_file.stem + ".pkl"
#     cluster_center_path = load_model_params_file.with_name(cluster_center_name)
#     joblib.dump(center, cluster_center_path)
    
#     # Save the corresponding final labels computed by kmeans
#     labels_name = "kmeans_labels_" + load_model_params_file.stem + ".pkl"
#     labels_path = load_model_params_file.with_name(labels_name)
#     joblib.dump(label, labels_path)
    pass

def train_encode_cluster_routine(args_config):
    # Read the corresponding cluster center pkl file, load the model, and fine-tune the corresponding DEC model
    # Train the corresponding encoder-decoder model
    
    # Create the communication group
    ddp_setup()
    # ddp_setup_cpu()
    
    # Start the training workflow
    # Define the dataset
    dataloader = train.PrepareDataLoader(dataset_path=args_config.get("train_params", "dataset_path"),
                                         process_method=args_config.getint("train_params", "process_method"),
                                         transform_method=args_config.getint("train_params", "transform_method"),
                                         return_type=args_config.get("train_params", "return_type"),
                                         split_method=args_config.getint("train_params", "split_method"),
                                         batch_size=args_config.getint("train_dec_params", "batch_size"),
                                         num_workers=args_config.getint("train_params", "num_workers")).prepare_dataloader()
    
    # Define the encoder-cluster model
    # Read the model loading file and the pkl file
    def find_params_file():
        # Use the log file recorded during training to find the file with the best train_loss 
        log_file=args_config.get("train_params", "save_log_file")
        file_df = pd.read_csv(log_file)
        idx_min = file_df.loc[:, "train_loss"].idxmin()
        model_params_name = file_df.iloc[idx_min].loc["model_name"]
        model_params_name = Path(model_params_name)
        if not model_params_name.is_file():
            print(f'cannot find the best model params file in {args_config.get("train_params", "save_model_dir")}')
            raise
            pass
        return model_params_name
    
    # Define the corresponding model; the model loaded here is an encoder-decoder
    load_model_params_file = find_params_file()
    cluster_center_name = "cluster_center_" + load_model_params_file.stem + ".pkl"
    cluster_center_path = load_model_params_file.with_name(cluster_center_name)
    cluster_center = joblib.load(cluster_center_path)
    
    raw_model = train.DefineModel(model_name=args_config.get("train_params", "model_name"), 
                              feature_size=args_config.getint("train_params", "feature_size"),
                              cluster_center=None,
                              load_model_params_file=load_model_params_file
                             ).model
    
    # Define the DEC model by extracting the encoder architecture from the raw model and appending the clustering architecture
    model = nn.Sequential(raw_model.encode, define_module.Cluster(center=cluster_center))
    
    # Define the training class
    trainer = train.Trainer(dataloader=dataloader,
                            model_name=args_config.get("train_dec_params", "model_name"),
                            model=model,
                            gpu_id=dist.get_rank(),
                            loss_func_name=args_config.get("train_dec_params", "loss_func_name"),
                            optimizer_name=args_config.get("train_dec_params", "optimizer_name"),
                            lr_scheduler_name=args_config.get("train_dec_params", "lr_scheduler_name"),
                            lr=args_config.getfloat("train_dec_params", "lr"),
                            max_epoch=args_config.getint("train_dec_params", "max_epoch"),
                            save_model_dir=args_config.get("train_params", "save_model_dir"),
                            save_log_file=args_config.get("train_dec_params", "save_log_file"),
                            args_config=args_config)
    
    trainer.train()
    
    # Destroy the communication group
    destroy_process_group()
    pass


def evaluate_encode_cluster_routine(args_config):
    # Evaluate the clustering results, directly generate the corresponding classifications, and save them to a pkl file
    
    # Define the dataset
    # Define the dataset using the methods in evaluate to include all data
    dataloader = evaluate.PrepareDataLoader(return_type="all",
                                            dataset_path=args_config.get("train_params", "dataset_path"),
                                            process_method=args_config.getint("train_params", "process_method"),
                                            transform_method=None,
                                            batch_size=1024,
                                            num_workers=1).prepare_dataloader()
    
    # Define the encoder-cluster model
    # Read the model loading file and the pkl file
    def find_params_file():
        # Use the log file recorded during training to find the file with the best train_loss 
        # The parameter here is train_dec_params; find the best clustering model
        log_file=args_config.get("train_dec_params", "save_log_file")
        file_df = pd.read_csv(log_file)
        idx_min = file_df.loc[:, "train_loss"].idxmin()
        model_params_name = file_df.iloc[idx_min].loc["model_name"]
        model_params_name = Path(model_params_name)
        if not model_params_name.is_file():
            print(f'cannot find the best model params file in {args_config.get("train_params", "save_model_dir")}')
            print(f"Use the final params file")
            model_params_name = Path(file_df.iloc[-1].loc["model_name"])
            pass
        
        # Find the cluster center pkl file to use as the initial model parameters
        folder_path = Path(args_config.get("train_params", "save_model_dir"))
        # Use glob to match the filename pattern; assume that there is only one optimal pkl file
        matching_files = list(folder_path.glob("cluster_center_*.pkl"))
        assert len(matching_files) == 1
        cluster_center_path = matching_files[0]
        cluster_center = joblib.load(cluster_center_path)
        return cluster_center, model_params_name
    
    cluster_center, load_model_params_file = find_params_file()
    
    # cluster_center is used for initialization
    # The trained cluster centers have already been loaded into the model through load_model_params
    model = train.DefineModel(model_name=args_config.get("train_dec_params", "model_name"), 
                              feature_size=args_config.getint("train_params", "feature_size"),
                              cluster_center=cluster_center,
                              load_model_params_file=load_model_params_file
                             ).model
    
    
    
    # Infer all corresponding latent features
    evaluator = evaluate.Evaluator(dataloader=dataloader,
                                   model=model,
                                   args_config=args_config)
    output_array, _ = evaluator.evaluate()
    
    # Save the corresponding final labels file as pkl
    final_labels = output_array.argmax(axis=1)
    final_labels_name = "final_labels_" + load_model_params_file.stem + ".pkl"
    final_labels_path = load_model_params_file.with_name(final_labels_name)
    joblib.dump(final_labels, final_labels_path)
    
    pass


def get_bash_args():
    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument("--type", default="train_encode_decode", type=str, help="define the basic routine")
    parser.add_argument("--config_file", default="./configs/config.ini", type=str, help="The file loaded hyper-parameters")
    return parser.parse_args()


if __name__ == "__main__":
    print(f"Script start in {datetime.now():%Y-%m-%d %H:%M:%S}")
    
    # Retrieve bash arguments
    bash_args = get_bash_args()
    
    # Retrieve ini file parameters
    args_config = configparser.ConfigParser()
    args_config.read(f"{bash_args.config_file}")
    
    # Fix the random seed
    global_seed(args_config.getint("train_params", "seed"))
    
    if bash_args.type == "train_encode_decode":
        # Train encode_decode
        # Launch with torch run in the bash startup file
        train_encode_decode_routine(args_config)
        pass
    elif bash_args.type == "eval_encode_decode":
        # Evaluate encode_decode
        # Run with ordinary python in bash; launching with torch run starts multiple processes
        evaluate_encode_decode_routine(args_config)
        pass
    elif bash_args.type == "get_cluster_center":
        # Compute the corresponding cluster centers using the trained encoder
        # Launch with python and save the cluster centers to a file
        # Read the cluster center data during subsequent DEC training
        get_cluster_center_routine(args_config)
        pass
    elif bash_args.type == "train_dec":
        # Read the encoder and cluster center files
        # Train dec
        train_encode_cluster_routine(args_config)
        pass
    elif bash_args.type == "eval_dec":
        # Compute the final labels output by dec
        evaluate_encode_cluster_routine(args_config)
    else:
        pass
    
    print(f"Script end in {datetime.now():%Y-%m-%d %H:%M:%S}")
    pass


