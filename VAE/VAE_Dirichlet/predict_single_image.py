import torch
import numpy as np
import torch.nn as nn
from dirichlet_vae import DIR_VAE
from mlp_model import MLP

# === Step 1: Load image and preprocess ===
def load_image(path):
    img = np.load(path)  # Shape: (H, W)
    tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()  # Shape: (1, 1, H, W)
    return tensor.to(device)

# === Step 2: Load VAE model ===
def load_vae_model(vae_path, base, latent_size, alpha_fill_value):
    model = DIR_VAE(base, latent_size, alpha_fill_value).to(device)
    checkpoint = torch.load(vae_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

# === Step 3: Load MLP classifier ===
def load_mlp_model(mlp_path, latent_size, base):
    checkpoint = torch.load(mlp_path)
    hyperparams = {
        "layer_sizes": checkpoint["layer_sizes"],
        "dropout": 0.5,  # 如有记录请替换
        "Depth": len(checkpoint["layer_sizes"]),
    }
    model = MLP(latent_size, base, **hyperparams).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

# === Step 4: 预测函数 ===
def classify_image(npy_path, vae_path, mlp_path, base, latent_size, alpha_fill_value, threshold=0.5):
    image_tensor = load_image(npy_path)  # (1, 1, H, W)

    vae = load_vae_model(vae_path, base, latent_size, alpha_fill_value)
    mlp = load_mlp_model(mlp_path, latent_size, base)

    with torch.no_grad():
        _, alpha, _ = vae(image_tensor)  # alpha shape: (1, latent_dim)
        latent_vector = alpha.view(1, -1).to(device)  # flatten to (1, latent_dim)
        output = mlp(latent_vector)
        prob = output.item()
        prediction = int(prob >= threshold)

    return prediction, prob

# === Usage example ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fill in these paths and parameters based on your setup
    
    image_path = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Images/0001_NI000_slice000.npy"
    vae_path = "VAE/results/VAE_params.pt"
    mlp_path = "VAE/results/MLP.pt"

    latent_size = 16  # 替换为你训练时用的值
    base = 32         # 替换为你训练时用的值
    alpha_fill_value = 0.1
    threshold = 0.5

    prediction, prob = classify_image(
        image_path, vae_path, mlp_path, base, latent_size, alpha_fill_value, threshold
    )

    cls = "MALIGNANT" if prediction == 1 else "NON-MALIGNANT"
    print(f"✅ Prediction: {cls} (Probability: {prob:.4f})")
