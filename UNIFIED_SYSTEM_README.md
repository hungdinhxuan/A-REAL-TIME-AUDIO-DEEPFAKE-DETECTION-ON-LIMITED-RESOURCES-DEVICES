# Unified Knowledge Distillation System

This system provides a flexible and configurable approach to knowledge distillation with unified loss functions. It allows you to easily experiment with different projection layers, pooling methods, and loss combinations through YAML configuration.

## Files Overview

- `unified_main.py`: Main training script that integrates the unified loss system
- `unified_utils.py`: Utility functions for training and validation with unified loss support
- `unified_loss.py`: The core unified loss implementation with flexible configurations
- `unified_config_example.yaml`: Example configuration file showing all available options

## Key Features

### 1. Flexible Projection Layers
- **Linear**: Simple linear transformation
- **Shallow Autoencoder**: Single-layer encoder-decoder with reconstruction loss
- **Deep Autoencoder**: Multi-layer encoder-decoder with reconstruction loss
- **MLP**: 3-layer MLP with ReLU activations
- **None**: No projection (direct feature matching)

### 2. Configurable Pooling Methods
- **mean**: Average pooling across layers
- **sum**: Sum pooling across layers
- **max**: Max pooling across layers
- **min**: Min pooling across layers
- **weighted_sum**: Weighted sum using learnable layer weights
- **last**: Use only the last layer
- **first**: Use only the first layer

### 3. Multiple Loss Types
- **MSE**: Mean Squared Error loss
- **Cosine**: Cosine Embedding loss
- **Reconstruction**: Reconstruction loss (for autoencoder projections)

### 4. Processing Modes
- **legacy**: Original format with manual feature extraction
- **new**: New format with direct feature tensor input

## Usage

### Basic Usage

1. **Prepare your configuration file** (see `unified_config_example.yaml`):

```yaml
unified_loss:
  projection:
    type: "linear"
    target: "teacher"
    input_dim: 1024
    output_dim: 768
  loss:
    enabled: ["mse", "cosine"]
    weights:
      mse: 0.0001
      cosine: 1.0
  pooling:
    method: "mean"
  processing_mode: "legacy"
```

2. **Run the training**:

```bash
python unified_main.py --yaml your_config.yaml
```

### Configuration Examples

#### Example 1: Linear Projection with Mean Pooling
```yaml
unified_loss:
  projection:
    type: "linear"
    target: "teacher"
    input_dim: 1024
    output_dim: 768
  loss:
    enabled: ["mse", "cosine"]
    weights:
      mse: 0.0001
      cosine: 1.0
  pooling:
    method: "mean"
```

#### Example 2: Shallow Autoencoder with Reconstruction Loss
```yaml
unified_loss:
  projection:
    type: "shallow_ae"
    target: "teacher"
    input_dim: 1024
    output_dim: 768
    kwargs:
      use_bias: true
  loss:
    enabled: ["mse", "cosine", "recon"]
    weights:
      mse: 0.0001
      cosine: 1.0
      recon: 0.0001
  pooling:
    method: "weighted_sum"
```

#### Example 3: MLP with Sum Pooling
```yaml
unified_loss:
  projection:
    type: "mlp"
    target: "student"
    input_dim: 768
    output_dim: 1024
    kwargs:
      hidden_dim: 1024
  loss:
    enabled: ["mse", "cosine"]
    weights:
      mse: 0.0001
      cosine: 1.0
  pooling:
    method: "sum"
```

#### Example 4: Deep Autoencoder with Max Pooling
```yaml
unified_loss:
  projection:
    type: "deep_ae"
    target: "teacher"
    input_dim: 1024
    output_dim: 768
    kwargs:
      dims: [1024, 1024, 768]
      use_bias: true
  loss:
    enabled: ["mse", "cosine", "recon"]
    weights:
      mse: 0.0001
      cosine: 1.0
      recon: 0.0001
  pooling:
    method: "max"
```

## Configuration Parameters

### Projection Configuration
- `type`: Type of projection layer ("linear", "shallow_ae", "deep_ae", "mlp", null)
- `target`: Which model to project ("teacher" or "student")
- `input_dim`: Input feature dimension
- `output_dim`: Output feature dimension
- `kwargs`: Additional parameters for specific projection types

### Loss Configuration
- `enabled`: List of enabled loss types (["mse", "cosine", "recon"])
- `weights`: Dictionary of loss weights

### Pooling Configuration
- `method`: Pooling method across layers ("mean", "sum", "max", "min", "weighted_sum", "last", "first")

### Processing Mode
- `processing_mode`: Input processing format ("legacy" or "new")

## Monitoring and Logging

The system provides comprehensive logging of all loss components:

- `unified_loss`: Total unified loss
- `unified_loss_mse`: MSE component
- `unified_loss_cosine`: Cosine component
- `unified_loss_recon`: Reconstruction component (if enabled)
- `ce_loss`: Cross-entropy loss

All metrics are logged to both Wandb and TensorBoard for easy monitoring.

## Advanced Features

### Layer Weighting
When using `weighted_sum` pooling, the system learns layer-specific weights automatically through learnable parameters.

### Mixed Precision Training
Full support for automatic mixed precision (AMP) training for improved performance and memory efficiency.

### Model Averaging
Automatic saving and averaging of top-N best models for improved final performance.

### Early Stopping
Configurable early stopping based on validation loss improvement.

## Migration from Legacy System

To migrate from the existing system:

1. Replace `StandardMidLoss` with `UnifiedMidLoss` in your configuration
2. Add the `unified_loss` section to your YAML configuration
3. Update your training script to use `unified_main.py` instead of the original main script
4. Update your training/validation functions to use the unified versions

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**: Ensure `input_dim` and `output_dim` match your model's feature dimensions
2. **Memory Issues**: Reduce batch size or use gradient accumulation
3. **Convergence Issues**: Try different loss weights or pooling methods

### Performance Tips

1. Use `weighted_sum` pooling for better layer importance learning
2. Enable reconstruction loss with autoencoder projections for richer feature learning
3. Experiment with different projection targets (teacher vs student)
4. Use `new` processing mode for better performance with compatible models

## Examples

See the `unified_config_example.yaml` file for comprehensive examples of different configurations and their use cases.

