import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support

class Trainer:
    def __init__(self, params, device, results_path="../VAE/results"):
        self.params = params
        self.device = device
        self.results_path = results_path

    def train(self, model, optimizer, criterion, train_loader):
        model.train()
        total_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        auc_score = roc_auc_score(all_labels, all_outputs)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, (np.array(all_outputs) >= self.params['threshold']).astype(int), average='binary')
        return total_loss / len(loader), auc_score, precision, recall, f1

    def train_model(self, model, train_loader, val_loader, run):
        optimizer = optim.Adam(model.parameters(), lr=self.params['lr'])
        criterion = nn.BCELoss()
        best_auc = 0
        for epoch in range(self.params['epochs']):
            train_loss = self.train(model, optimizer, criterion, train_loader)
            val_loss, auc, precision, recall, f1 = self.evaluate(model, val_loader, criterion)
            print(f"Epoch {epoch+1}/{self.params['epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - AUC: {auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), os.path.join(self.results_path, f"MLP_{run}.pt"))
                print(f"Model saved with AUC: {best_auc:.4f}")
