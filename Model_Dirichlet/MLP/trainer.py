import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# Trainer class handles model training, validation, early stopping, and result saving
# Trainer类负责模型的训练、验证、早停和结果保存
class Trainer:
    def __init__(self, params, device, results_path, latent_size, base):
        self.params = params    # Store hyperparameters 存储超参数
        self.device = device    # Set computation device 设置计算设备
        self.results_path = results_path    # Path to save results 保存结果路径
        self.latent_size = latent_size      # Latent variable dimensions 潜变量维度
        self.base = base        # Channel cardinality 通道基数
    
    
    # --------------- Main training function 主训练函数 --------------- 
    def train_model(self, model, epochs, train_loader, valid_loader):
        threshold = self.params['threshold']
        lr = self.params['lr']
        
        statsrec = np.zeros((4,epochs))
        
        loss_fn = nn.BCELoss()  # binary cross entropy
        optimiser = optimizer = optim.Adam(model.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=20, 
                                                                   threshold=0.001, threshold_mode='abs')
        
        # For early stopping 早停计数器
        counter = 0
        
        # Train model over several epochs 在多个epoch上训练模型
        for epoch in range(1,epochs+1):  # loop over the dataset multiple times
            correct = 0          # number of examples predicted correctly (for accuracy)
            total = 0            # number of examples
            running_loss = 0.0   # accumulated loss (for mean loss)
            n = 0                # number of minibatches
            model.train()
            for data in train_loader:
                inputs, labels = data
                inputs = inputs.float().to(self.device)
                labels = labels.float().to(self.device)
                if inputs.shape[0] == 1:
                    continue   
                # Zero the parameter gradients
                optimiser.zero_grad()
        
                # Forward
                outputs = model(inputs)
                # Backward, and update parameters
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
        
                # accumulate data for accuracy
                predicted = self.get_predictions(outputs.data, threshold)
                predicted = predicted.to(self.device)
                total += labels.size(0)    # add in the number of labels in this minibatch
                correct += (predicted == labels).sum().item()  # add in the number of correct labels
                
                # accumulate loss
                running_loss += loss.item()
                n += 1
        
            # collect together statistics for this epoch
            train_loss = running_loss/n
            train_accuracy = correct/total 
            
            val_loss, val_acc, val_outputs, val_labels = self.validate(valid_loader, model, threshold)
            
            statsrec[:,epoch-1] = (train_loss, train_accuracy, val_loss.item(), val_acc)
            
            if epoch % 75 == 0 or epoch == 1 or epoch == epochs - 1 or counter == 24:
                print(f"[MLP Trainer] epoch: {epoch} training loss: {train_loss: .3f} training accuracy: {train_accuracy: .1%}  validation loss: {val_loss: .3f} validation accuracy: {val_acc: .1%}")
            
            # Update LR based on train loss 根据训练损失更新学习率
            scheduler.step(train_loss)
            counter = self.early_stopping(counter, train_loss, val_loss, min_delta=0.25)

            if counter > 25:
                print("[MLP Trainer] At Epoch:", epoch)
                break
        
        val_cm = self.confusion_matrix(val_outputs, val_labels, threshold)
        
        auc = metrics.roc_auc_score(val_labels, val_outputs)
        
        results = self.evaluation_metrics(val_cm[1,1], val_cm[0,1], val_cm[0,0], val_cm[1,0])
        
        return statsrec, results, auc
    
    # Convert continuous prediction (probability) to binary label using threshold
    # 使用给定的阈值将概率输出转换为0或1标签
    def get_predictions(self, predictions, threshold):
        preds = []
        for pred in predictions:
            if pred >= threshold:
                preds.append([1])
            else:
                preds.append([0])
        return torch.Tensor(preds).to(self.device)

    # Compute confusion matrix between true labels and predicted probabilities
    # 根据标签和预测概率计算混淆矩阵（TP、FP、TN、FN）
    def confusion_matrix(self, outputs, train_labels, threshold):
        labels = np.squeeze(train_labels)
        labels = np.array([int(lab) for lab in labels])
        # convert outputs to numpy array 
        if type(outputs) == torch.Tensor:
            preds = np.array(outputs.detach())
        else:
            preds = outputs
        
        predictions = []
        for pred in preds:
            if pred >= threshold:
                predictions.append(1)  # p closer to 1
            if pred < threshold:
                predictions.append(0) # p close to 0
        predictions = np.array(predictions)   
        
        # Create 2x2 confusion matrix
        nclasses = 2
        cm = np.zeros((nclasses, nclasses)) # cm with counts
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(predictions == i, 1, 0) * np.where(labels == j, 1, 0))
        return cm

    # Calculate standard classification metrics: Precision, Recall, Specificity, F1
    # 计算分类任务中常见的评估指标
    def evaluation_metrics(self, tp, fp, tn, fn):
        if tp == 0 and fp == 0:
            fp = 1
        if tp == 0 and fn == 0:
            fn = 1
        if tn == 0 and fp == 0:
            fp = 1
        
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        specificity = tn/(tn+fp)
        f1 = 2*((precision*recall)/(precision+recall))
        
        results = [precision, recall, specificity, f1]
        
        return results
    
    # Evaluate model performance on a given dataset
    # 在验证集或测试集上评估模型损失、准确率，并记录输出/标签
    def validate(self, val_loader, model, threshold):
        correct = 0
        total = 0
        running_loss = 0
        n = 1    # counter for number of minibatches
        output_list = []
        label_list = []
        loss_fn = nn.BCELoss()
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images = images.float().to(self.device)
                labels = labels.float().to(self.device)
                model.eval()
                outputs = model(images)    

                # accumulate loss
                running_loss += loss_fn(outputs, labels)
                n += 1

                # accumulate data for accuracy
                #_, predicted = torch.max(outputs.data, 1)
                predicted = self.get_predictions(outputs.data, threshold)
                predicted = predicted.to(self.device)
                total += labels.size(0)    # add in the number of labels in this minibatch
                correct += (predicted == labels).sum().item()  # add in the number of correct labels
                output_list.append(outputs.cpu())
                label_list.append(labels.cpu())
            output_list = np.concatenate(output_list)
            label_list = np.concatenate(label_list)
            
            val_loss = running_loss.cpu()/n
            val_acc = correct/total
            val_outputs = output_list
            val_labels = label_list
            
        return val_loss, val_acc, val_outputs, val_labels

    # Early stopping logic 简单早停机制
    def early_stopping(self, counter, train_loss, validation_loss, min_delta):
        if (validation_loss - train_loss) > min_delta:
            counter += 1
            if counter % 10 == 0 or counter == 25:
                print('early stopping counter at:', counter)
        return counter
    
    # Plot training and test loss curves 绘制训练和测试损失曲线
    def plot_results(self, filename, model_path, epochs):
        # Skip if model file not found 如果模型文件不存在，则跳过绘图
        if not os.path.exists(model_path):
            print(f"[MLP Trainer] Data file {model_path} not found. Skipping plot generation.")
            return

        data = torch.load(model_path, weights_only=False)
        statsrec = data["stats"]
        fig, ax1 = plt.subplots()
        plt.plot(statsrec[0][:epochs], 'm', label = 'training loss', )
        plt.plot(statsrec[2][:epochs], 'g', label = 'validation loss' )
        plt.legend(loc='lower right')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training and validation loss, and validation accuracy')
        ax2=ax1.twinx()
        ax2.plot(statsrec[1][:epochs], 'b', label = 'training accuracy')
        ax2.plot(statsrec[3][:epochs], 'r', label = 'validation accuracy')
        ax2.set_ylabel('accuracy')
        plt.legend(loc='upper right')
        plt.show()
        plt.close()
        
        save_path = os.path.join(self.results_path, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

