import os
import pandas as pd
import torch
import numpy as np
from VAE.dirichlet_vae import DIR_VAE
from MLP.mlp_model import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Images"
meta = r"D:/aMaster/github_code/VAE_lung_lesion_BMVC/Data/Meta/meta_mal_ben.csv"
vae_path = 'results/VAE_params.pt'
mlp_path = 'results/MLP.pt'

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


# === Step 1: Custom Dataset (single image) ===
def classify_patient(patient_id, meta, vae_path, mlp_path, IMAGE_DIR, method='vote'):
    # 读取CSV
    df = pd.read_csv(meta)
    
    # 过滤出目标病人的测试切片
    patient_slices = df[df['patient_id'] == patient_id]
    
    if patient_slices.empty:
        raise ValueError(f"No Test slices found for patient_id: {patient_id}")
    
    # 加载模型
    vae, vae_params = load_vae_model(vae_path)
    mlp, mlp_params = load_mlp_model(mlp_path, vae_params["latent_size"], vae_params["base"])

    predictions = []
    probabilities = []
    
    print(f"\n--- Predicting patient {patient_id} ---")
    
    for _, row in patient_slices.iterrows():
        image_filename = row["original_image"] + ".npy"
        image_path = os.path.join(IMAGE_DIR, image_filename)
        
        image = np.load(image_path)  # shape: (H, W)
        img_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)  # shape: (1, 1, H, W)
        
        with torch.no_grad():
            _, alpha, _ = vae(img_tensor)
            latent_vector = alpha.view(1, -1)
            output = mlp(latent_vector)
            prob = output.item()
            pred = int(prob >= mlp_params["threshold"])
        
        predictions.append(pred)
        probabilities.append(prob)
        
        print(f"Slice: {image_filename} → Prediction: {pred}, Probability: {prob:.4f}")

    # 聚合预测结果
    # 多数投票策略（majority voting）
    if method == 'vote':
        final_pred = int(sum(predictions) > len(predictions) / 2)
    # 平均概率策略。
    elif method == 'avg':
        final_pred = int(np.mean(probabilities) >= mlp_params["threshold"])
    else:
        raise ValueError("Invalid method. Choose 'vote' or 'avg'.")
        
    return final_pred, predictions, probabilities

# === Step 4: Predict ===
if __name__ == "__main__":
    patient_id = 21
    final_result, slice_predictions, slice_probs = classify_patient(
        patient_id, meta, vae_path, mlp_path, IMAGE_DIR, method='vote')
    
    label = "No Cancer" if final_result == 1 else "Cancer"
    
    print("\n--- Final Prediction ---")
    print(f"Patient: {patient_id}")
    print(f"Result: {label}")
    # print(f"Slice-level predictions: {slice_predictions}")
    # print(f"Slice-level probabilities: {slice_probs}")

