import numpy as np
import torch
import random
import time
import math
import pandas as pd
from time import perf_counter

from data_loader import Cross_Validation
from trainer import Trainer


def test_hyperparams(params, nepochs, num_folds, vae_test_loss, latent_size, base, Run, results_path, latent_vectors, device):
    Mal_NonMal = [0,0,0,0,0]
    Mal_Ben = [0,0,0,0,0]

    if not math.isnan(vae_test_loss):
        # 去除 ambiguous 样本
        ambiguous = np.load(r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/latent vectors/ambiguous.npy")
        latent_vectors2 = [torch.tensor(vec) for i, vec in enumerate(latent_vectors) if i not in ambiguous]
        latent_vectors = torch.stack(latent_vectors2)

        # 加载 meta 和标签
        meta = pd.read_csv(r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_ben.csv")
        labels = np.load(r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/latent vectors/labels3.npy")

        # 使用固定的超参数
        mlp_hyperparams = {
            "threshold": params["threshold"],
            "lr": params["lr"],
            "layer_sizes": params["layer_sizes"],
            "dropout": params["dropout"],
            "batch_size": params["batch_size"],
            "Depth": params["Depth"],
        }

        batch_size = mlp_hyperparams["batch_size"]

        CV_loss, CV_accuracy, CV_results, CV_auc = [], [], [], []
        for run in range(num_folds):
            train_loader, valid_loader, test_loader = Cross_Validation(run, num_folds, meta, latent_vectors, labels, batch_size)
            trainer = Trainer(params=params, device=device, results_path=results_path, latent_size=latent_size, base=base)
            loss, accuracy, results, auc = trainer.train_model(nepochs, train_loader, valid_loader, test_loader, mlp_hyperparams, run_index=Run, fold_index=run)
            CV_loss.append(loss)
            CV_accuracy.append(accuracy)
            CV_results.append(results)
            CV_auc.append(auc)

        # 结果平均与提取
        avg_auc = np.mean(CV_auc)
        avg_loss = np.mean(CV_loss)
        avg_accuracy = np.mean(CV_accuracy)
        avg_results = trainer.average_metrics(CV_results)
        
        Mal_Ben = [avg_auc, avg_loss, avg_accuracy, avg_results, mlp_hyperparams]
        
        print("Final Results Based on Fixed Hyperparameters:")
        print("AUC:", avg_auc)
        print("Loss:", avg_loss)
        print("Accuracy:", avg_accuracy)
        print("Performance Metrics:", avg_results)
        print("Hyperparameters:", mlp_hyperparams)

    return Mal_NonMal, Mal_Ben
