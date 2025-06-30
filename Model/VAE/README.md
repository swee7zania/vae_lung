# Hyperparameter Search

- To perform hyperparameter tuning, run `vae_hyperparam_search.py`. 

- All intermediate search results will be saved in the `vae_search_results` directory.

- The final summary of all experiments is consolidated in `vae_hyperparam_results.xlsx`.

  - The optimal hyperparameter combination was selected based on a custom scoring formula:
    $$
    Score = 0.7 * (1 - ValLoss) + 0.2 * SSIM + 0.1 * (1 - MSE)
    $$

  - This formula prioritizes validation loss, while also accounting for perceptual similarity (SSIM) and robustness to pixel-level noise (MSE).

- Best Hyperparameter Combination (Score = 0.862):

  | `HU_UpperBound` | `HU_LowerBound` | `base` | `latent_size` | `annealing` | `alpha` | `beta` | `lr`   | `batch_size` |
  | --------------- | --------------- | ------ | ------------- | ----------- | ------- | ------ | ------ | ------------ |
  | 600             | 1000            | 64     | 4             | 0           | 0.8     | 5      | 0.0001 | 128          |

### Field Explanation

| Field Name      | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `mse`           | Mean Squared Error – the lower, the better reconstruction accuracy |
| `l1`            | Mean Absolute Error (L1 Loss / MAE) – also better when lower |
| `ssim`          | Structural Similarity Index (SSIM), measures structural similarity of images; higher is better (range: 0~1) |
| `train_loss`    | Total loss on the training set (could be a weighted loss like L1+SSIM+KL) |
| `val_loss`      | Total loss on the validation set, same as above              |
| `params`        | Hyperparameter configuration used in this experiment, stored as a dictionary |
| `HU_UpperBound` | Upper bound of HU window, used for lung CT image normalization |
| `HU_LowerBound` | Lower bound of HU window                                     |
| `base`          | Base multiplier for channel expansion in the network         |
| `latent_size`   | Dimensionality of the latent variable                        |
| `annealing`     | Whether KL annealing is enabled (0 = No, 1 = Yes)            |
| `alpha`         | Weight of L1 in the L1 and SSIM combination (`recon_mix = α·L1 + (1-α)·SSIM`) |
| `beta`          | Scaling factor before KL divergence, controls bottleneck strength |
| `lr`            | Learning rate                                                |
| `batch_size`    | Batch size – larger batches improve training stability but use more memory |
| `combo_id`      | ID number of the current hyperparameter combination          |

### Metric Interpretation

| **Metric**   | **Important?** | **Purpose/Explanation**                                      |
| ------------ | -------------- | ------------------------------------------------------------ |
| `val_loss`   | ✅ Pivotal      | Total reconstruction loss on the validation set; reflects generalization |
| `train_loss` |                | Total reconstruction loss on the training set; used to monitor overfitting |
| `ssim`       | ⚠️ Minor        | Perceptual reconstruction quality; more aligned with human vision |
| `mse`        | ⚠️ Minor        | Mean squared error; traditional metric, sensitive to pixel noise |
| `l1`         |                | MAE (Mean Absolute Error), similar to MSE, less perceptually meaningful |