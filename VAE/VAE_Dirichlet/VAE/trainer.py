import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import math

class Trainer:
    def __init__(self, params, device, Run, results_path, model):
        self.params = params
        self.device = device
        self.Run = Run
        self.results_path = results_path
        self.model = model
    
    def train_model(self, model, lr, epochs, sample_shape, train_loader, test_loader):
        train_losses, test_losses, ssim_score_list = [], [], []
        optimiser = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=20, 
                                                               threshold=0.001, threshold_mode='abs')
        counter = 0
        for epoch in range(1, epochs + 1):
            train_loss, ssim_score, kld = self.train(model, epoch, epochs, optimiser, sample_shape, train_loader)
            test_loss, test_ssim = self.test(model, epoch, epochs, test_loader)
            scheduler.step(train_loss)
            counter = self.early_stopping(counter, train_loss, test_loss, min_delta=1)
            if counter > 25:
                print('Early stopping triggered at epoch:', epoch)
                break

            if math.isnan(train_loss):
                print('Training stopped due to infinite loss')
                break

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            ssim_score_list.append(ssim_score)
        
        torch.save({"state_dict": model.state_dict(), "train_losses": train_losses, "test_losses": test_losses, "params": self.params}, self.results_path + '/VAE_params.pt')
        return test_loss, test_ssim

    def train(self, model, epoch, epochs, optimiser, sample_shape, train_loader):
        model.train()
        train_loss, beta_train_loss = 0, 0
        ssim_list = []
        for batch_idx, data in enumerate(train_loader):
            data = data.float().to(self.device)
            optimiser.zero_grad()
            recon_batch, alpha, dirichlet_sample = model(data)
            loss, recon_loss, kld, ssim_score, pure_loss = model.loss_function(recon_batch, data, alpha, epoch, self.params)
            ssim_list.append(ssim_score.item())
            loss.backward()
            train_loss += pure_loss.item()
            beta_train_loss += loss.item()
            optimiser.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tPure Loss: {:.6f}, Beta Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    pure_loss.item(), loss.item()))
            if math.isnan(loss):
                break

        if((epoch%50==1) or (epoch < 5) or (epoch==epochs-1)):
            print('12 Real Images')
            img_grid = make_grid(data[:12], nrow=4, padding=12, pad_value=-1)
            plt.figure(figsize=(10,5))
            plt.imshow(img_grid[0].detach().cpu())
            plt.axis('off')
            plt.savefig(self.results_path + "/" + "visualise_real" + str(epoch) + '.png')
            plt.show()

            print('12 Reconstructed Images')
            img_grid = make_grid(recon_batch[:12], nrow=4, padding=12, pad_value=-1)
            plt.figure(figsize=(10,5))
            plt.imshow(img_grid[0].detach().cpu())
            plt.axis('off')
            plt.savefig(self.results_path + "/" + "visualise_reconstructed" + str(epoch) + '.png')
            plt.show()

            print('12 Synthetic Images')
            sample = torch.randn(sample_shape).to(self.device)
            recon_rand_sample = model.decode(sample)
            img_grid = make_grid(recon_rand_sample[:12], nrow=4, padding=12, pad_value=-1)
            plt.imshow(img_grid[0].detach().cpu())
            plt.axis('off')
            plt.savefig(self.results_path + "/" + "visualise_synthetic" + str(epoch) + '.png')
            plt.show()
        
        train_loss /= len(train_loader.dataset)
        beta_train_loss /= len(train_loader.dataset)
        print('====> Epoch {}: Average Train Loss: {:.4f}'.format(epoch, train_loss))
        print('====> Average Beta Train Loss: {:.4f}'.format(beta_train_loss))
        ssim_mean = np.mean(ssim_list)
        print('====> Average Train SSIM: {:.4f}'.format(ssim_mean))

        return train_loss, ssim_mean, kld
        
    def test(self, model, epoch, epochs, test_loader):
        model.eval()
        test_loss, beta_test_loss = 0, 0
        ssim_list = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                data = data.float().to(self.device)
                recon_batch, alpha, dirichlet_sample = model(data)
                testloss, recon_loss, kld, ssim_score, pure_loss = model.loss_function(recon_batch, data, alpha, epoch, self.params)
                test_loss += pure_loss.item()
                beta_test_loss += testloss.item()
                ssim_list.append(ssim_score.item())
                if math.isnan(testloss):
                    break

        test_loss /= len(test_loader.dataset)
        beta_test_loss /= len(test_loader.dataset)
        print('====> Pure Test Loss: {:.4f}'.format(test_loss))
        print('====> Beta Test Loss: {:.4f}'.format(beta_test_loss))
        ssim_mean = np.mean(ssim_list)
        print('====> Average Test SSIM: {:.4f}'.format(ssim_mean))
        return test_loss, ssim_mean

    def early_stopping(self, counter, train_loss, test_loss, min_delta):
        if (test_loss - train_loss) > min_delta:
            counter += 1
            if counter % 5 == 0:
                print('Early Stopping Counter At:', counter)  
        return counter
        
    def plot_results(self, filename):
        data_path = os.path.join(self.results_path, "VAE_params.pt")
        if not os.path.exists(data_path):
            print(f"Data file {data_path} not found. Skipping plot generation.")
            return

        data = torch.load(data_path)
        loss = data["train_losses"]
        val_loss = data["test_losses"]
        loss = loss[1:]
        val_loss = val_loss[1:]
        fig, ax1 = plt.subplots()
        plt.plot(loss, 'm', label = 'training loss')
        plt.plot(val_loss, 'g', label = 'test loss')
        plt.yscale("log")
        plt.legend(loc='lower right')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Training and validation loss')
        fig.savefig(os.path.join(self.results_path, filename))
        plt.show()
        plt.close()