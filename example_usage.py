import torch
import torch.nn as nn
from autoencoders import DeepAutoencoder, AbstractAutoencoder
from typing import Sequence

# Custom DeepAutoencoder for 3D inputs that preserves sequence structure
class SequentialDeepAutoencoder(AbstractAutoencoder):
    """ Deep AE that processes each frame independently while preserving sequence structure """
    def __init__(self, dims: Sequence[int], use_bias=True):
        """
        :param dims: seq of integers specifying the dimensions of the layers 
                    (first dim should be feature_size, last dim is the compressed size)
        :param use_bias: if False, don't use bias
        """
        super().__init__()
        assert len(dims) > 0 and all(d > 0 for d in dims)
        self.type = "sequentialDeepAE"
        self.use_bias = use_bias
        
        # Build encoder layers
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1], bias=use_bias))
            if i < len(dims) - 2:  # Don't add ReLU after the last layer
                enc_layers.append(nn.ReLU(inplace=True))
        
        # Build decoder layers
        dec_layers = []
        for i in reversed(range(1, len(dims))):
            dec_layers.append(nn.Linear(dims[i], dims[i - 1], bias=use_bias))
            if i > 1:  # Don't add activation after the last layer
                dec_layers.append(nn.ReLU(inplace=True))
        
        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, feature_size)
        batch_size, seq_len, feature_size = x.shape
        
        # Reshape to process all frames at once: (batch_size * seq_len, feature_size)
        x_reshaped = x.view(-1, feature_size)
        
        # Encode
        encoded_reshaped = self.encoder(x_reshaped)
        latent_dim = encoded_reshaped.shape[-1]
        
        # Reshape back to sequence format: (batch_size, seq_len, latent_dim)
        encoded = encoded_reshaped.view(batch_size, seq_len, latent_dim)
        
        # Decode
        decoded_reshaped = self.decoder(encoded_reshaped)
        decoded = decoded_reshaped.view(batch_size, seq_len, feature_size)
        
        return encoded, decoded


def main():
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define your input dimensions
    batch_size = 32
    sequence_length = 766  # number of frames
    input_feature_size = 1024
    output_feature_size = 768
    
    # Create sample input data
    input_data = torch.randn(batch_size, sequence_length, input_feature_size).to(device)
    print(f"Input shape: {input_data.shape}")
    
    # Method 1: Using the original DeepAutoencoder (flattens everything)
    print("\n=== Method 1: Original DeepAutoencoder (Not Recommended for your use case) ===")
    
    # Calculate total flattened size
    flattened_input_size = sequence_length * input_feature_size  # 766 * 1024 = 784,384
    flattened_output_size = sequence_length * output_feature_size  # 766 * 768 = 588,288
    
    # Define architecture: input -> hidden layers -> bottleneck -> hidden layers -> output
    dims_original = [flattened_input_size, 10000, 5000, 2000, flattened_output_size]
    
    original_ae = DeepAutoencoder(dims_original).to(device)
    
    # Flatten input for original autoencoder
    input_flattened = input_data.view(batch_size, -1)  # (32, 784384)
    print(f"Flattened input shape: {input_flattened.shape}")
    
    with torch.no_grad():
        encoded_orig, decoded_orig = original_ae(input_flattened)
        # Reshape back to 3D
        decoded_orig_3d = decoded_orig.view(batch_size, sequence_length, output_feature_size)
        print(f"Original AE - Encoded shape: {encoded_orig.shape}")
        print(f"Original AE - Decoded shape (reshaped): {decoded_orig_3d.shape}")
    
    # Method 2: Using the Sequential DeepAutoencoder (Recommended)
    print("\n=== Method 2: Sequential DeepAutoencoder (Recommended) ===")
    
    # Define architecture for frame-wise processing
    dims_sequential = [input_feature_size, 896, 832, output_feature_size]  # 1024 -> 896 -> 832 -> 768
    
    sequential_ae = SequentialDeepAutoencoder(dims_sequential).to(device)
    
    with torch.no_grad():
        encoded_seq, decoded_seq = sequential_ae(input_data)
        print(f"Sequential AE - Input shape: {input_data.shape}")
        print(f"Sequential AE - Encoded shape: {encoded_seq.shape}")
        print(f"Sequential AE - Decoded shape: {decoded_seq.shape}")
    
    # Method 3: Using PyTorch's built-in approach with Linear layers
    print("\n=== Method 3: Simple Linear Transformation ===")
    
    # For your specific use case, you might just need a simple linear transformation
    simple_encoder = nn.Linear(input_feature_size, output_feature_size).to(device)
    
    with torch.no_grad():
        # Apply the transformation to each frame
        simple_output = simple_encoder(input_data)  # Works directly on (32, 766, 1024)
        print(f"Simple Linear - Input shape: {input_data.shape}")
        print(f"Simple Linear - Output shape: {simple_output.shape}")
    
    # Training example for Sequential DeepAutoencoder
    print("\n=== Training Example ===")
    
    # Create model and optimizer
    model = SequentialDeepAutoencoder(dims_sequential).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop example
    model.train()
    for epoch in range(3):  # Just 3 epochs for demo
        optimizer.zero_grad()
        
        # Forward pass
        encoded, decoded = model(input_data)
        
        # For autoencoder, we want to reconstruct the input
        # But since you want different output size, you'd need target data
        # Here's an example assuming you have target data
        target_data = torch.randn(batch_size, sequence_length, output_feature_size).to(device)
        
        # Loss between decoded output and target
        loss = criterion(decoded, target_data)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/3, Loss: {loss.item():.6f}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main() 