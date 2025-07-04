import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange, repeat

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split into train, validation, test (80%, 10%, 10%)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

# Test dataset (official test set)
official_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
official_test_loader = DataLoader(official_test_dataset, batch_size=batch_size, shuffle=False)

# Parameters

patch_size = 7

embed_dim = 128
num_heads = 4
num_layers = 3

# Enter the size of images  . 
image_size = st.sidebar.number_input("Images Size ", min_value=1, max_value=1000, value=28)

num_patches = (image_size // patch_size) ** 2
# Streamlit app
st.title("Masked Autoencoder Vision Transformer (MAE ViT)")

# Sidebar for parameters
st.sidebar.header("Model Parameters")

# Manually enter epochs, batch index, and sample indices
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=1000, value=100)

# Slider for masking ratio
mask_ratio = st.sidebar.slider("Masking Ratio", min_value=0.1, max_value=0.9, value=0.75, step=0.05)

# Input for batch index
batch_idx = st.sidebar.number_input(
    "Enter Batch Index",
    min_value=0,
    max_value=len(train_loader)-1,

    value=0
)

# Input for sample indices
sample_indices_input = st.sidebar.text_input(
    "Enter Sample Indices (comma-separated)",
    value="0,1,2,3,4"
)


# Convert the comma-separated string to a list of integers
sample_indices = [int(idx.strip()) for idx in sample_indices_input.split(",") if idx.strip().isdigit()]

# Ensure sample indices are within the valid range
sample_indices = [idx for idx in sample_indices if 0 <= idx < batch_size]

# Helper functions
def show_images(images, titles=None, num_images=5):
    """Display a grid of images"""
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.axis('off')
        if titles is not None:
            ax.set_title(titles[i])
    st.pyplot(fig)

# Function to select batch and samples, apply patching and masking
def process_batch(batch_idx, sample_indices, loader=train_loader, mask_ratio=0.75):
    """Select a batch and samples, apply patching and masking"""
    # Get the selected batch
    for i, (images, labels) in enumerate(loader):
        if i == batch_idx:
            selected_batch = images
            selected_labels = labels
            break

    # Select specific samples from the batch
    selected_images = selected_batch[sample_indices]
    selected_labels = selected_labels[sample_indices]

    # Apply patching
    patches = rearrange(selected_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

    # Create masking
    batch_size, num_patches, _ = patches.shape
    num_masked = int(mask_ratio * num_patches)

    # Create random noise and sort for masking
    noise = torch.rand(batch_size, num_patches, device=patches.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Split into masked and unmasked
    ids_keep = ids_shuffle[:, :num_patches - num_masked]
    ids_mask = ids_shuffle[:, num_patches - num_masked:]

    # Gather patches
    patches_keep = torch.gather(patches, 1, ids_keep.unsqueeze(-1).repeat(1, 1, patches.shape[-1]))
    patches_mask = torch.gather(patches, 1, ids_mask.unsqueeze(-1).repeat(1, 1, patches.shape[-1]))

    # Create binary mask (0=keep, 1=mask)
    mask = torch.ones(batch_size, num_patches, device=patches.device)
    mask[:, :num_patches - num_masked] = 0
    mask = torch.gather(mask, 1, ids_restore)

    return selected_images, patches, patches_keep, patches_mask, mask, ids_restore, selected_labels

# Masked Autoencoder Transformer
class PatchEmbedding(nn.Module):
    """Turn image into patches and project to embedding dimension"""
    def __init__(self, img_size=28, patch_size=7, in_chans=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Linear(patch_size * patch_size * in_chans, embed_dim)

    def forward(self, x):
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=self.patch_size, p2=self.patch_size)
        x = self.proj(x)
        return x

class PositionalEncoding(nn.Module):
    """Add learnable positional encoding to patch embeddings"""
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def forward(self, x):
        return x + self.pos_embed

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_chans=1, embed_dim=128,
                 num_heads=4, num_layers=3, mask_ratio=0.75):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = PositionalEncoding(self.num_patches, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=4*embed_dim,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=4*embed_dim,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=1)

        # Prediction head
        self.head = nn.Linear(embed_dim, patch_size * patch_size * in_chans)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x):
        """Randomly mask patches"""
        B, N, D = x.shape
        num_masked = int(self.mask_ratio * N)

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :N - num_masked]
        ids_mask = ids_shuffle[:, N - num_masked:]

        # Keep unmasked patches
        x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate mask tokens for masked patches
        mask_tokens = self.mask_token.repeat(B, num_masked, 1)

        # Combine unmasked and mask tokens
        x_masked = torch.cat([x_keep, mask_tokens], dim=1)

        # Restore original order
        x_masked = torch.gather(x_masked, 1, ids_restore.unsqueeze(-1).repeat(1, 1, D))

        # Create binary mask (0=keep, 1=mask)
        mask = torch.ones(B, N, device=x.device)
        mask[:, :N - num_masked] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        # Embed patches
        x = self.patch_embed(x)
        x = self.pos_embed(x)

        # Mask patches
        x, mask, ids_restore = self.random_masking(x)

        # Apply transformer encoder
        x = self.encoder(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # Apply transformer decoder
        x = self.decoder(x)

        # Predict pixel values for all patches
        pred = self.head(x)

        # Restore original patch order
        pred = torch.gather(pred, 1, ids_restore.unsqueeze(-1).repeat(1, 1, pred.shape[-1]))
        return pred

    def forward(self, x):
        # Encoder
        latent, mask, ids_restore = self.forward_encoder(x)

        # Decoder
        pred = self.forward_decoder(latent, ids_restore)

        return pred, mask

    def reconstruct(self, x):
        """Reconstruct the image from patches"""
        pred, _ = self.forward(x)

        # Reshape predictions to image
        pred = rearrange(pred, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                         h=self.img_size//self.patch_size,
                         p1=self.patch_size, p2=self.patch_size)
        return pred

# Set model parameters and create model
model = MaskedAutoencoderViT(
    img_size=image_size,
    patch_size=patch_size,
    in_chans=1,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    mask_ratio=mask_ratio
).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Training function
def train_model(model, train_loader, val_loader, epochs):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            # Forward pass
            pred, mask = model(images)

            # Get original patches
            patches = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                               p1=patch_size, p2=patch_size)

            # Calculate loss only on masked patches
            loss = criterion(pred[mask.bool()], patches[mask.bool()])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)

                pred, mask = model(images)
                patches = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                   p1=patch_size, p2=patch_size)

                loss = criterion(pred[mask.bool()], patches[mask.bool()])
                val_loss += loss.item() * images.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        st.write(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return train_losses, val_losses

# Visualization functions
def visualize_patching_masking(batch_idx, sample_indices, mask_ratio=0.75):
    """Visualize original images and masked images"""
    images, patches, patches_keep, patches_mask, mask, ids_restore, labels = process_batch(
        batch_idx, sample_indices, mask_ratio=mask_ratio
    )

    # Convert tensors to numpy for visualization
    images = images.cpu().numpy()
    patches = patches.cpu().numpy()
    mask = mask.cpu().numpy()

    # Select first few samples
    num_samples = min(5, len(sample_indices))

    # Show original images
    st.write("Original Images:")
    show_images(images[:num_samples], [f"Label: {labels[i]}" for i in range(num_samples)], num_samples)

    # Create masked images
    masked_patches = patches.copy()
    for i in range(num_samples):
        masked_patches[i, mask[i].astype(bool)] = 0

    masked_recon = rearrange(masked_patches[:num_samples], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                           h=image_size//patch_size, p1=patch_size, p2=patch_size)
    st.write(f"Masked Images ({int(mask_ratio*100)}% patches masked):")
    show_images(masked_recon[:num_samples], [f"Label: {labels[i]}" for i in range(num_samples)], num_samples)

# Call the visualization function at the beginning of your script


def visualize_reconstructed_images(model, batch_idx, sample_indices):
    """Visualize only the reconstructed images"""
    model.eval()

    # Get the selected batch
    for i, (images, labels) in enumerate(train_loader):
        if i == batch_idx:
            selected_batch = images.to(device)
            selected_labels = labels
            break

    # Select specific samples from the batch
    images = selected_batch[sample_indices]
    labels = selected_labels[sample_indices]

    # Get model predictions
    with torch.no_grad():
        pred, mask = model(images)

    # Reconstruct images from predictions
    recon = model.reconstruct(images)

    # Convert to numpy for visualization
    recon_np = recon.detach().cpu().numpy()

    # Select first few samples
    num_samples = min(5, len(sample_indices))

    # Plot reconstructed images
    st.write("Reconstructed Images:")
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for i in range(num_samples):
        axes[i].imshow(recon_np[i].squeeze(), cmap='gray')
        axes[i].set_title("Reconstructed")
        axes[i].axis('off')

    plt.tight_layout()
    st.pyplot(fig)


# Evaluation function
def evaluate_model(model, loader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            pred, mask = model(images)
            patches = rearrange(images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                               p1=patch_size, p2=patch_size)

            loss = criterion(pred[mask.bool()], patches[mask.bool()])
            total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

if __name__ == "__main__":
    st.title("Masked Autoencoder Vision Transformer (MAE ViT)")

    # Visualize patching and masking at the top
    st.write("Visualizing patching and masking process...")
    visualize_patching_masking(batch_idx, sample_indices, mask_ratio)

    # Train the model
    st.write("Training the model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs)

    # Plot training curves
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    st.pyplot(fig)

    # Evaluate on test set
    test_loss = evaluate_model(model, test_loader)
    st.write(f"Test Loss: {test_loss:.4f}")

    # Evaluate on official test set
    official_test_loss = evaluate_model(model, official_test_loader)
    st.write(f"Official Test Set Loss: {official_test_loss:.4f}")

    # Visualize only reconstructed images
    st.write("Visualizing reconstructed images:")
    visualize_reconstructed_images(model, batch_idx, sample_indices)
