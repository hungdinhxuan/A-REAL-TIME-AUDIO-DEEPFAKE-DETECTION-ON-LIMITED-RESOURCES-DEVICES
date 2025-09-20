import torch
from autoencoders import DeepAutoencoder

def compare_frame_strategies():
    """
    Compare different strategies for handling frame information:
    1. Keep frames separate (current approach)
    2. Merge frames (pool/average)
    3. Use temporal modeling
    """
    
    # Setup
    device = torch.device('cpu')
    batch_size = 1
    sequence_length = 3
    input_feature_size = 1024
    output_feature_size = 768
    
    # Create sample input data with some temporal pattern
    # Frame 1: mostly positive values
    # Frame 2: mixed values  
    # Frame 3: mostly negative values
    input_data = torch.stack([
        torch.randn(batch_size, input_feature_size) + 1.0,  # Frame 1: positive bias
        torch.randn(batch_size, input_feature_size),         # Frame 2: neutral
        torch.randn(batch_size, input_feature_size) - 1.0,  # Frame 3: negative bias
    ], dim=1)  # Shape: (1, 3, 1024)
    
    print("=== INPUT DATA ANALYSIS ===")
    print(f"Input shape: {input_data.shape}")
    print(f"Frame 1 mean: {input_data[0, 0].mean():.3f}")
    print(f"Frame 2 mean: {input_data[0, 1].mean():.3f}")
    print(f"Frame 3 mean: {input_data[0, 2].mean():.3f}")
    print(f"Temporal variance: {input_data[0].mean(dim=1).var():.3f}")
    
    # Create autoencoder
    dims = [input_feature_size, 32, 64, output_feature_size]
    autoencoder = DeepAutoencoder(dims=dims, use_bias=True)
    
    print("\n=== STRATEGY 1: KEEP FRAMES SEPARATE (Current) ===")
    # Process each frame independently
    input_frames = input_data.view(-1, input_feature_size)  # (3, 1024)
    
    with torch.no_grad():
        encoded_frames = autoencoder.encoder(input_frames)  # (3, 768)
        output_separate = encoded_frames.view(batch_size, sequence_length, output_feature_size)  # (1, 3, 768)
    
    print(f"Output shape: {output_separate.shape}")
    print(f"Frame 1 encoded mean: {output_separate[0, 0].mean():.3f}")
    print(f"Frame 2 encoded mean: {output_separate[0, 1].mean():.3f}")
    print(f"Frame 3 encoded mean: {output_separate[0, 2].mean():.3f}")
    print(f"Temporal info preserved: {output_separate[0].mean(dim=1).var():.3f}")
    
    print("\n=== STRATEGY 2A: MERGE BY AVERAGING ===")
    # Average all frames first, then encode
    averaged_input = input_data.mean(dim=1)  # (1, 1024)
    
    with torch.no_grad():
        encoded_avg = autoencoder.encoder(averaged_input)  # (1, 768)
    
    print(f"Output shape: {encoded_avg.shape}")
    print(f"Averaged encoded mean: {encoded_avg[0].mean():.3f}")
    print("Temporal info lost: All frame information merged into single representation")
    
    print("\n=== STRATEGY 2B: MERGE BY MAX POOLING ===")
    # Max pool across frames, then encode
    maxpooled_input, _ = input_data.max(dim=1)  # (1, 1024)
    
    with torch.no_grad():
        encoded_max = autoencoder.encoder(maxpooled_input)  # (1, 768)
    
    print(f"Output shape: {encoded_max.shape}")
    print(f"Max pooled encoded mean: {encoded_max[0].mean():.3f}")
    print("Temporal info partially lost: Only maximum activations preserved")
    
    print("\n=== STRATEGY 3: ENCODE THEN AGGREGATE ===")
    # Encode each frame, then aggregate the encoded representations
    with torch.no_grad():
        # Same as Strategy 1 but with different aggregation options
        encoded_frames = autoencoder.encoder(input_frames)  # (3, 768)
        
        # Option 3a: Average encoded frames
        encoded_avg_post = encoded_frames.mean(dim=0, keepdim=True)  # (1, 768)
        
        # Option 3b: Max pool encoded frames  
        encoded_max_post, _ = encoded_frames.max(dim=0, keepdim=True)  # (1, 768)
        
        # Option 3c: Weighted average (attention-like)
        weights = torch.softmax(torch.randn(3), dim=0)
        encoded_weighted = (encoded_frames * weights.unsqueeze(1)).sum(dim=0, keepdim=True)  # (1, 768)
    
    print(f"Post-encoding average shape: {encoded_avg_post.shape}")
    print(f"Post-encoding max shape: {encoded_max_post.shape}")
    print(f"Post-encoding weighted shape: {encoded_weighted.shape}")
    print("Temporal info: Partially preserved through learned aggregation")
    
    print("\n=== RECOMMENDATIONS ===")
    print("🔹 KEEP FRAMES SEPARATE if:")
    print("  - Temporal dynamics are important (e.g., speech recognition, video analysis)")
    print("  - You have downstream models that can handle sequences")
    print("  - You want maximum information preservation")
    print("  - Your task benefits from frame-level predictions")
    
    print("\n🔹 MERGE FRAMES if:")
    print("  - You need a fixed-size representation regardless of sequence length")
    print("  - Temporal order is less important than overall content")
    print("  - You have computational/memory constraints")
    print("  - Your downstream model expects single vectors")
    
    print("\n🔹 FOR YOUR AUDIO SPOOFING DETECTION:")
    print("  - Audio spoofing often has temporal artifacts")
    print("  - Frame-level analysis can catch local inconsistencies")
    print("  - BUT: Your current models (AASIST, Conformer) expect sequence input")
    print("  - RECOMMENDATION: Keep frames separate for now!")
    
    return {
        'separate_frames': output_separate,
        'averaged': encoded_avg,
        'maxpooled': encoded_max,
        'post_avg': encoded_avg_post,
        'post_max': encoded_max_post,
        'weighted': encoded_weighted
    }

def analyze_temporal_importance():
    """
    Analyze how much temporal information matters for your specific use case
    """
    print("\n" + "="*60)
    print("TEMPORAL INFORMATION ANALYSIS FOR AUDIO SPOOFING")
    print("="*60)
    
    print("\n📊 Why temporal info matters in audio spoofing detection:")
    print("1. Artifacts often appear as temporal inconsistencies")
    print("2. Natural speech has specific temporal patterns")
    print("3. Synthetic speech may have periodic artifacts")
    print("4. Transitions between phonemes contain important cues")
    
    print("\n📊 Evidence from your codebase:")
    print("- AASIST uses both spectral AND temporal GAT layers")
    print("- Conformer uses positional embeddings for sequence modeling")
    print("- Your models expect (batch, time, features) input")
    
    print("\n📊 Current approach assessment:")
    print("✅ GOOD: Preserves all temporal information")
    print("✅ GOOD: Compatible with existing models")
    print("✅ GOOD: Allows frame-level analysis")
    print("⚠️  CONSIDER: Computational overhead")
    print("⚠️  CONSIDER: Memory usage for long sequences")

if __name__ == "__main__":
    results = compare_frame_strategies()
    analyze_temporal_importance()
    
    print("\n" + "="*60)
    print("FINAL RECOMMENDATION")
    print("="*60)
    print("Based on your audio spoofing detection task:")
    print("👉 KEEP FRAMES SEPARATE (current approach)")
    print("👉 Your models are designed for temporal data")
    print("👉 Temporal artifacts are crucial for spoofing detection")
    print("👉 You can always add pooling layers later if needed") 