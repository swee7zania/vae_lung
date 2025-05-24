import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from mlp_model import MLP

class Trainer:
    def __init__(self, params, device, results_path, latent_size, base):
        self.params = params
        self.device = device
        self.results_path = results_path
        self.latent_size = latent_size
        self.base = base

    def get_predictions(self, predictions, threshold):
        preds = []
        for pred in predictions:
            if pred >= threshold:
                preds.append([1])
            else:
                preds.append([0])
        return torch.Tensor(preds).to(self.device)

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
        nclasses = 2
        cm = np.zeros((nclasses, nclasses)) # cm with counts
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(predictions == i, 1, 0) * np.where(labels == j, 1, 0))
        return cm

    def evaluation_metrics(self, tp, fp, tn, fn):
        if tp == 0 and fp == 0:
            fp = 1
        if tp == 0 and fn == 0:
            fn = 1
        if tn == 0 and fp == 0:
            fp = 1
        precision = tp/(tp+fp)
        print('Precision,', 'proprotion of malignant predictions that are true:', precision,)
        recall = tp/(tp+fn)
        print('Recall,', 'proportion of tumours identified:', recall)
        specificity = tn/(tn+fp)
        print('Specificity,', 'proportion of non-cancerous lesions identified:', specificity)
        f1 = 2*((precision*recall)/(precision+recall))
        print('F1 score:', f1)
        
        results = [precision, recall, specificity, f1]
        return results
    
    def average_metrics(self, results_list):
        precision, recall, specificity, f1 = [], [], [], []
        for result in results_list:
            precision.append(result[0])
            recall.append(result[1])
            specificity.append(result[2])
            f1.append(result[3])
        print('Average Precision: {}, Recall: {}, Specificity: {}, and F1 Score: {}'.format(np.mean(precision), np.mean(recall),
                                                                                            np.mean(specificity), np.mean(f1)))
        average_results = [np.mean(precision), np.mean(recall), np.mean(specificity), np.mean(f1)]
        return average_results
    
    def stats(self, loader, model, threshold):
        '''
        function to calculate validation accuracy and loss
        '''
        correct = 0
        total = 0
        running_loss = 0
        n = 1    # counter for number of minibatches
        output_list = []
        label_list = []
        loss_fn = nn.BCELoss()
        with torch.no_grad():
            for data in loader:
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
        return running_loss.cpu()/n, correct/total, output_list, label_list

    def train_model(self, nepochs, train_loader, valid_loader, test_loader, mlp_hyperparams, run_index, fold_index):
        latent_size = self.latent_size
        base = self.base
        layer_sizes = self.params['layer_sizes']
        dropout = self.params['dropout']
        Depth = self.params['Depth']
        threshold = self.params['threshold']
        lr = self.params['lr']
        
        model = MLP(latent_size, base, layer_sizes, dropout, Depth).to(self.device)
        
        statsrec = np.zeros((4,nepochs))
        
        loss_fn = nn.BCELoss()  # binary cross entropy
        optimiser = optimizer = optim.Adam(model.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=20, 
                                                                   threshold=0.001, threshold_mode='abs')
        counter = 0
        for epoch in range(1,nepochs+1):  # loop over the dataset multiple times
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
            ltrn = running_loss/n
            atrn = correct/total 
            
            # Correc way to this part
            # valid_results = self.stats(test_loader, model, threshold)
            # valid_loss, valid_acc = valid_results[0], valid_results[1]
            # valid_outputs, valid_labels = valid_results[2], valid_results[3]
            # if epoch % 75 == 0 or epoch == nepochs - 1 or counter == 25:
            #     print('valid loss:', valid_loss.item(),'test accuracy:', valid_acc*100, '%')  
            # scheduler.step(valid_loss)
            
            results = self.stats(valid_loader, model, threshold)
            lval, aval = results[0], results[1]
            #val_outputs, val_labels = results[2], results[3]
            statsrec[:,epoch-1] = (ltrn, atrn, lval.item(), aval)
            if epoch % 75 == 0 or epoch == 1 or epoch == nepochs - 1 or counter == 24:
                print(f"epoch: {epoch} training loss: {ltrn: .3f} training accuracy: {atrn: .1%}  validation loss: {lval: .3f} validation accuracy: {aval: .1%}")
                
            test_results = self.stats(test_loader,model, threshold)
            test_loss, test_acc = test_results[0], test_results[1]
            test_outputs, test_labels = test_results[2], test_results[3]
            if epoch % 75 == 0 or epoch == nepochs - 1 or counter == 25:
                print('test loss:', test_loss.item(),'test accuracy:', test_acc*100, '%')  
            #incorrect
            scheduler.step(test_loss)  
            # correct
           # scheduler.step(lval)
            
            counter = self.early_stopping(counter, ltrn, lval, min_delta=0.25)
            if counter > 25:
                print("At Epoch:", epoch)
                break
        
        # save network parameters, losses and accuracy
        torch.save({"state_dict": model.state_dict(), "params": self.params, "stats": statsrec}, os.path.join(self.results_path, "MLP.pt"))
        self.plot_results(self.results_path, epoch, f"MLP_train_curve_run{run_index}_fold{fold_index}.png")
        
        test_cm = self.confusion_matrix(test_outputs, test_labels, threshold)
        auc = metrics.roc_auc_score(test_labels, test_outputs)
        print('AUC is:', auc)
        results = self.evaluation_metrics(test_cm[1,1], test_cm[0,1], test_cm[0,0], test_cm[1,0])
        
        return test_loss.item(), test_acc, results, auc
    
    def early_stopping(self, counter, train_loss, validation_loss, min_delta):
        if (validation_loss - train_loss) > min_delta:
            counter += 1
            if counter % 10 == 0 or counter == 25:
                print('early stopping counter at:', counter)
        return counter
    
    def plot_results(self, results_path, epoch, filename):
        '''
        method for plotting accuracy and loss graphs and saving file in given destination
        '''
        data_path = os.path.join(self.results_path, "MLP.pt")
        if not os.path.exists(data_path):
            print(f"Data file {data_path} not found. Skipping plot generation.")
            return
        data = torch.load(data_path, weights_only=False)
        statsrec = data["stats"]
        fig, ax1 = plt.subplots()
        plt.plot(statsrec[0][:epoch], 'm', label = 'training loss', )
        plt.plot(statsrec[2][:epoch], 'g', label = 'validation loss' )
        plt.legend(loc='lower right')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training and validation loss, and validation accuracy')
        ax2=ax1.twinx()
        ax2.plot(statsrec[1][:epoch], 'b', label = 'training accuracy')
        ax2.plot(statsrec[3][:epoch], 'r', label = 'validation accuracy')
        ax2.set_ylabel('accuracy')
        plt.legend(loc='upper right')
        #fig.savefig(filename)
        plt.show()
        plt.close()
        
        save_path = os.path.join(self.results_path, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

