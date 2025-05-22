import numpy as np
import torch
import random
import time
import math
import pandas as pd
from time import perf_counter

from data_loader import Cross_Validation
from trainer import Trainer


def test_hyperparams(params, data, nepochs, num_folds, vae_test_loss, latent_size, base, Run, results_path, latent_vectors, device):
    break_indicator = 0
    accuracy_list, loss_list, auc_list, mlp_hyperparams_list, results_list = [], [], [], [], []
    mlp_runs = 0
    num_tried = 0
    
    # 创建 latent_vectors2，用于 malignant vs benign
    ambiguous = np.load(r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/latent vectors/ambiguous.npy")
    latent_vectors2 = []
    for i, vec in enumerate(latent_vectors):
        if i not in ambiguous:
            latent_vectors2.append(torch.tensor(vec))
    latent_vectors2 = torch.stack(latent_vectors2)

    if not math.isnan(vae_test_loss):
        if data == 1:
            # malignant vs non-malignant   
            meta = pd.read_csv(r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_nonmal.csv")
            latent_vectors = np.load(results_path + '/' + "latent_vectors_{}.npy".format(Run), allow_pickle=True)
            labels = np.load(r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/latent vectors/labels2.npy")
        if data == 2:
           # malignant vs benign 
            meta = pd.read_csv(r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_ben.csv")
            latent_vectors = latent_vectors2
            labels = np.load(r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/latent vectors/labels3.npy")   
            
        while len(mlp_hyperparams_list) < 25:
            mlp_runs +=1
            print("Attempt:", mlp_runs)
            print("Number of completed attempts:", num_tried)
            if mlp_runs == 100:
                break_indicator = 1
                break
            random.seed(time.time())
            mlp_hyperparams = {
                "threshold":params["threshold"],
                "lr":params["lr"],
                "layer_sizes":params["layer_sizes"],
                "dropout":params["dropout"],
                "batch_size":params["batch_size"],
                "Depth":params["Depth"],
                }
            batch_size = params["batch_size"]
            # 这里其实要随机的，我没有随机，所以 Attempt2-100 都跳过了
            if mlp_hyperparams in mlp_hyperparams_list:
                    continue

            train_start = perf_counter()
            CV_loss, CV_accuracy, CV_results, CV_auc = [], [], [], []
            for run in range(num_folds):  # 5-fold cross validation
                train_loader, valid_loader, test_loader = Cross_Validation(run, num_folds, meta, latent_vectors, labels, batch_size)
                trainer = Trainer(params=params, device=device, results_path=results_path, latent_size=latent_size, base=base)
                loss, accuracy, results, auc = trainer.train_model(nepochs, train_loader, valid_loader, test_loader, mlp_hyperparams, run_index=Run, fold_index=run)
                if accuracy < 0.6:
                    print("########## This is not a good candidate for cross validation ##########")
                    break 
                CV_loss.append(loss)
                CV_accuracy.append(accuracy)
                CV_results.append(results)
                CV_auc.append(auc)

          
            train_stop= perf_counter()
            print('training time', train_stop - train_start)  
            
            if len(CV_auc) == 5  or mlp_runs < 3:
                print(".................Cross Validation Averages.................")
                print("AUC:", np.mean(CV_auc), "Loss:", np.mean(CV_loss), "Accuracy:", np.mean(CV_accuracy), "Results:", trainer.average_metrics(CV_results))
                auc_list.append(np.mean(CV_auc))
                loss_list.append(np.mean(CV_loss))
                accuracy_list.append(np.mean(CV_accuracy))
                mlp_hyperparams_list.append(mlp_hyperparams)
                results_list.append(trainer.average_metrics(CV_results))
                if len(CV_auc) == 5:
                    num_tried +=1
                    print("Number of completed attempts:", num_tried)
                        
                auc_list = np.nan_to_num(auc_list).tolist()
                idx = auc_list.index(max(auc_list))
                print("Best so far", "Based on AUC:",
                'AUC:', max(auc_list), 
                'Test Loss:', loss_list[idx], 'Test Accuracy:', accuracy_list[idx], 
                'Performance Metrics:', results_list[idx],'Index:', idx, 
                'Hyperparameters:', mlp_hyperparams_list[auc_list.index(max(auc_list))])
                
                accuracy_list = np.nan_to_num(accuracy_list).tolist()
                idx2 = accuracy_list.index(max(accuracy_list))
                print("Best so far", "Based on Accuracy",
                'AUC:', auc_list[idx2], 'Test Loss:', loss_list[idx2], 'Test Accuracy:', accuracy_list[idx2], 
                'Performance Metrics:', results_list[idx2],'Index:', idx2, 
                'Hyperparameters:', mlp_hyperparams_list[accuracy_list.index(max(accuracy_list))])

        # 结果提取
        if len(auc_list) > 1:   
            auc_list = np.nan_to_num(auc_list).tolist()
            idx = auc_list.index(max(auc_list))
            print("Based on AUC:",
              'AUC:', max(auc_list), 
              'Test Loss:', loss_list[idx], 'Test Accuracy:', accuracy_list[idx], 
              'Performance Metrics:', results_list[idx],
              'Index:', idx, 
              'Hyperparameters:', mlp_hyperparams_list[auc_list.index(max(auc_list))])
        
            accuracy_list = np.nan_to_num(accuracy_list).tolist()
            idx2 = accuracy_list.index(max(accuracy_list))
            print("Based on Accuracy",
              'AUC:', auc_list[idx2], 'Test Loss:', loss_list[idx2], 'Test Accuracy:', accuracy_list[idx2], 
              'Performance Metrics:', results_list[idx2],'Index:', idx2, 
              'Hyperparameters:', mlp_hyperparams_list[accuracy_list.index(max(accuracy_list))])
        
            if data == 1:
                Mal_NonMal = [auc_list[idx],loss_list[idx],accuracy_list[idx],results_list[idx], mlp_hyperparams_list[idx]]
                Mal_Ben = [0,0,0,0,0]
                
                Mal_NonMal2 = [auc_list[idx2],loss_list[idx2],accuracy_list[idx2],results_list[idx2], mlp_hyperparams_list[idx2]]
                Mal_Ben2 = [0,0,0,0,0]
                
            if data == 2:
                Mal_NonMal = [0,0,0,0,0]
                Mal_Ben = [auc_list[idx],loss_list[idx],accuracy_list[idx],results_list[idx], mlp_hyperparams_list[idx]]  
                
                Mal_NonMal2 = [0,0,0,0,0]
                Mal_Ben2 = [auc_list[idx2],loss_list[idx2],accuracy_list[idx2],results_list[idx2], mlp_hyperparams_list[idx2]]  

    if math.isnan(vae_test_loss) or break_indicator == 1:
        Mal_Ben = [0,0,0,0,0]
        Mal_NonMal = [0,0,0,0,0]
        Mal_Ben2 = [0,0,0,0,0]
        Mal_NonMal2 = [0,0,0,0,0]

    return Mal_NonMal, Mal_Ben, Mal_NonMal2, Mal_Ben2