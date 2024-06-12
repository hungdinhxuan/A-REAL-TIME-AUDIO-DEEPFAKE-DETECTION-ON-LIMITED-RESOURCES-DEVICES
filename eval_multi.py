import time
import torch
import argparse
# Set the number of threads for intra-op and inter-op parallelism
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


parser = argparse.ArgumentParser(
        description='Eval multi metrics')

parser.add_argument('--flops', default=False, action='store_true', help='Calculate FLOPs')
parser.add_argument('--rtf', default=False, action='store_true', help='Calculate Real-Time Factor')

# Example model and dummy input for tracing
class MyModel(torch.nn.Module):
    def forward(self, x):
        return x * 2  # Simple operation for illustration

model = MyModel()
dummy_input = torch.randn(1, 3, 224, 224)  # Replace with appropriate dummy input

# Trace the model
traced_model = torch.jit.trace(model, dummy_input)

# Your data loader setup
data_loader = MyDataLoader()  # Replace with your data loader
device = torch.device('cpu')  # Ensure you are using the CPU

def process_signal(model, signal):
    with torch.no_grad():
        output = model(signal)
    return output

def measure_rtf(model, data_loader, device):
    """
    Measure the Real-Time Factor (RTF) for a given model and data loader.

    Args:
    model (torch.nn.Module): The PyTorch model.
    data_loader (torch.utils.data.DataLoader): DataLoader providing input signals.
    device (torch.device): Device to run the model on (CPU).

    Returns:
    float: The Real-Time Factor (RTF).
    """
    model.to(device)
    model.eval()
    
    total_processing_time = 0
    total_signal_duration = 0
    
    for batch in data_loader:
        signals = batch['signal'].to(device)
        signal_durations = batch['duration']  # Assuming each batch has signal durations

        for signal, duration in zip(signals, signal_durations):
            start_time = time.time()
            process_signal(model, signal)
            end_time = time.time()
            
            processing_time = end_time - start_time
            total_processing_time += processing_time
            total_signal_duration += duration.item()

    if total_signal_duration == 0:
        raise ValueError("Total signal duration cannot be zero.")
        
    rtf = total_processing_time / total_signal_duration
    return rtf

# Measure RTF using the traced model
rtf = measure_rtf(traced_model, data_loader, device)
print(f"The Real-Time Factor (RTF) is: {rtf}")
