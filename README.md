# Comparative Study of Generative Models & Empirical Proof of Classification Model Limitations for Image Generation

## Repository Overview

This repository contains the complete empirical implementation for the comparative study of generative models and the mathematical-empirical verification of discriminative classification limitations in generative tasks:

1. **`2- MLP Classifie.ipynb` (Point 2)**: Theoretical and empirical proof demonstrating why the weights and latent feature representations of a Multi-Layer Perceptron (MLP) Classifier cannot be directly utilized for image reconstruction or synthesis, conducted through five structured experimental stages.
2. **`3_Generative_Models.ipynb` (Point 3)**: Full-scale standardized implementation, training, quantitative evaluation, and comparative analysis of six major generative architectures, addressing core research inquiries Q1 through Q6.

---

## Repository Structure

```text
├── 2- MLP Classifie.ipynb      # Point 2: MLP Classifier Limitations for Image Generation (Stages 1-5)
├── 3_Generative_Models.ipynb   # Point 3: Comparative Study of 6 Generative Models (Q1 - Q6 Analysis)
├── requirements.txt            # Python dependencies and deep learning libraries
├── .gitignore                  # Exclusion rules for datasets, cache, and model binaries
└── README.md                   # Project documentation and execution instructions
```

---

## 🔬 Experimental Scope & Methodology

### 1. Proof of MLP Classifier Limitations (`2- MLP Classifie.ipynb`)
Empirically demonstrates three fundamental theoretical bottlenecks: the classification **information bottleneck** (loss of intra-class spatial variance), **algebraic non-invertibility** ($W^T \ne W^{-1}$ with irreversible ReLU clipping), and **absence of latent density regularization** ($P(z)$).

* **Stage 1: Hierarchical MLP Classifier Training**:
  Trains an MLP Classifier ($784 \to 256 \to 64 \to 10$) using Cross-Entropy Loss to reach $\ge 88\%$ test accuracy. Extracts 64-dimensional bottleneck latent features $z_{\text{MLP}} \in \mathbb{R}^{64}$.
* **Stage 2: Top-Down Weight Transposition from Class Vectors**:
  Direct backward projection of one-hot class vectors ($y_c \in \mathbb{R}^{10}$) to pixel space via weight transposition:
  $$\hat{z} = \text{ReLU}(W_3^T y_c) \longrightarrow \hat{h}_1 = \text{ReLU}(W_2^T \hat{z}) \longrightarrow \hat{x} = \text{Sigmoid}(W_1^T \hat{h}_1)$$
  Visualizes resulting coarse weight template superpositions caused by non-orthogonal forward weights.
* **Stage 3: Latent Transpose Projection from Real Test Images**:
  Directly projects real image latents $z_{\text{MLP}} \in \mathbb{R}^{64}$ back to pixel space:
  $$\hat{x} = \text{Sigmoid}(W_1^T \text{ReLU}(W_2^T z_{\text{MLP}}))$$
  Demonstrates that forward classification weights cannot function as an invertible reconstructive decoder.
* **Stage 4: Frozen Latent Decoding Evaluation**:
  Freezes classifier weights (`requires_grad = False`) and trains an independent MLP Decoder ($64 \to 256 \to 784$) with MSE Loss. Evaluates reconstruction metrics (Test MSE, PSNR, SSIM) against an equivalent Vanilla Autoencoder ($z_{\text{AE}} \in \mathbb{R}^{64}$).
* **Stage 5: Generative Prior Sampling in MLP Latent Space**:
  Computes empirical test set latent statistics ($\mu_z, \sigma_z$) and samples Gaussian latents:
  $$z_{\text{rand}} \sim \mathcal{N}(\mu_z, \text{diag}(\sigma_z^2))$$
  Demonstrates that the classifier latent space lacks a continuous, regular generative manifold.

---

### 2. Comparative Study of 6 Generative Models (`3_Generative_Models.ipynb`)
Standardized implementation and benchmark evaluation across six model families:

1. **Vanilla Autoencoder (Vanilla AE)**: Deterministic dimensionality reduction with pixel-wise MSE reconstruction loss.
2. **Sparse Autoencoder (Sparse AE)**: L1 sparsity regularization penalty applied to latent activations.
3. **Variational Autoencoder (VAE)**: Continuous probabilistic latent space modeling using the Reparameterization Trick and KL-Divergence regularization against an isotropic Gaussian prior.
4. **Vanilla GAN**: Minimax adversarial game with fully connected (MLP) Generator and Discriminator networks.
5. **Deep Convolutional GAN (DCGAN)**: Convolutional adversarial network using transposed convolutions, strided convolutions, batch normalization, and LeakyReLU activations.
6. **Denoising Diffusion Probabilistic Models (DDPM)**: Forward discrete-time Gaussian noise injection schedule with a reverse denoising trajectory network.

---

## Evaluation Metrics & Quantitative Framework

* **Reconstruction Fidelity**: Evaluated on the unseen Test Set using Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR in dB), and Structural Similarity Index Measure (SSIM).
* **Generative Sample Quality**: Fréchet Inception Distance (FID) to assess the distribution distance between synthesized samples and real ground-truth images.
* **Inference Latency & Efficiency**: Measured per batch of 64 images to examine the trade-offs of the *Generative Trilemma* (Sample Quality vs. Sampling Speed vs. Mode Diversity).
* **Latent Space Continuity**: Evaluated via linear latent interpolation:
  $$z(\alpha) = (1 - \alpha)z_A + \alpha z_B, \quad \alpha \in [0, 1]$$

---

## Installation & Execution Guide

### 1. Environment Setup
Python 3.10 or 3.11 with an isolated virtual environment is recommended:

```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
Install all required libraries specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Running the Notebooks
Launch Jupyter Notebook or open the notebooks directly in VS Code:

```bash
jupyter notebook
```

* Execute **`2- MLP Classifie.ipynb`** to run the 5-stage empirical proof of classification model limitations.
* Execute **`3_Generative_Models.ipynb`** to run the 6-model generative comparison and metric evaluations.

