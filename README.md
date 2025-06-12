## Lung Lesion Representation Learning and Classification with VAE and MLP

This project implements a complete deep learning pipeline for **lung lesion image representation and classification** based on CT data. It combines a **Variational Autoencoder (VAE)** for unsupervised feature learning and a **multi-layer perceptron (MLP)** classifier for supervised diagnosis.

------

### Project Pipeline

1. **Preprocessing**:
   - Converts raw DICOM-format CT scans to NumPy arrays.
   - Applies lung-window normalization using Hounsfield Units (HU).
   - Organizes metadata and performs patient-level training/testing splits.
2. **Representation Learning**:
   - Trains a convolutional VAE to encode lesion images into compact latent vectors.
   - Supports hybrid loss with L1 + SSIM metrics.
   - Includes support for KL annealing and reconstruction loss weighting.
3. **Classification**:
   - Uses learned VAE latent features to train an MLP classifier.
   - Performs 5-fold cross-validation with patient-level separation.
   - Computes AUC, accuracy, precision, recall, specificity, and F1 score.

```py
/Data/
├── /Images/             # Save the processed image
├── /Mask/               # Save the processed image mask
├── /Meta/               # Save the processed metadata

/Preprocessing/
├── LIDC_DICOM_to_Numpy.ipynb    # Processing of CT images and mask images
├── Train_Test_Split.ipynb       # Data partitioning and metadata processing

/Model_Dirichlet/
├── /VAE/                # Code for training the VAE model
├── /MLP/                # Code for training the MLP classifier
├── /utils/              # As the execution entry for the project
├── /results/            # Save all results
```

------

### Technology Stack

| Component      | Technology / Library                           |
| -------------- | ---------------------------------------------- |
| Preprocessing  | NumPy, Pandas, PyTorch Dataset API             |
| Model Backbone | Convolutional VAE with flexible latent size    |
| Loss Functions | L1, SSIM (`pytorch-msssim`), KL Divergence     |
| Optimization   | Adam optimizer with learning rate scheduler    |
| Classifier     | Configurable MLP (PyTorch)                     |
| Evaluation     | sklearn (AUC), custom metrics                  |
| CV Strategy    | Patient-wise stratified 5-fold split           |
| Visualization  | Matplotlib (loss plots, image reconstructions) |