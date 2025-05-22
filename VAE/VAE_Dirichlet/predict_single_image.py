import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from VAE.dirichlet_vae import DIR_VAE
from MLP.mlp_model import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Step 1: Custom Dataset (single image) ===
class SingleImageDataset(Dataset):
    def __init__(self, image_path):
        self.image = np.load(image_path)  # (H, W)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img_tensor = torch.tensor(self.image).unsqueeze(0).float()  # shape: (1, H, W)
        return img_tensor  # shape: (1, H, W)

# === Step 2: Loading the VAE model (with parameters) ===
def load_vae_model(vae_path):
    checkpoint = torch.load(vae_path, map_location=device, weights_only=False)
    params = checkpoint["params"]
    model = DIR_VAE(params["base"], params["latent_size"], params["alpha_fill_value"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, params

# === Step 3: Loading the MLP model (with parameters) ===
def load_mlp_model(mlp_path, latent_size, base):
    checkpoint = torch.load(mlp_path, map_location=device, weights_only=False)
    params = checkpoint["params"]
    model = MLP(latent_size, base, params["layer_sizes"], params["dropout"], params["Depth"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, params

# === Step 4: Predict ===
def classify_image(image_path, vae_path, mlp_path):
    # Build DataLoader (batch size = 1)
    dataset = SingleImageDataset(image_path)
    loader = DataLoader(dataset, batch_size=1)

    # Loading the VAE model
    vae, vae_params = load_vae_model(vae_path)

    # Read a image
    for img_batch in loader:
        img_batch = img_batch.to(device)  # shape: (1, 1, H, W)
        with torch.no_grad():
            _, alpha, _ = vae(img_batch)  # alpha: (1, latent_dim)
            latent_vector = alpha.view(1, -1)

    # Load MLP models and classify them
    mlp, mlp_params = load_mlp_model(mlp_path, vae_params["latent_size"], vae_params["base"])
    with torch.no_grad():
        output = mlp(latent_vector)
        prob = output.item()
        pred = int(prob >= mlp_params["threshold"])

    return pred, prob

