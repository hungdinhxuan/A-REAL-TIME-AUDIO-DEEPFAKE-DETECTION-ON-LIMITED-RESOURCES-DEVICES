import torch
import torch.nn as nn
import yaml
from torchdistill.models.registry import get_model
from main import W2V2_TA
from torchaudio.models.wav2vec2.utils import import_fairseq_model
#from export_v2 import WrapperModel, pad
PADDING_SIZE = 10000
def pad(x, max_len: int = PADDING_SIZE):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = x.repeat((1, num_repeats))[:, :max_len][0]
    return padded_x
device = "cuda" if torch.cuda.is_available() else "cpu"
class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = nn.Softmax(dim=1)
        print('WrapperModel')

    def forward(self, x):
        wav_padded = pad(x).unsqueeze(0)
        output = self.model(wav_padded)
        return self.softmax(output)[0][0]


        # Update for kaist
        output = torch.argmax(output, dim=1)
        # If output is 1, then it is bonafide (real) else it is spoofed
        # Swap the output
        if output == 0:
            return 1
        else:
            return 0
def test_jit_component(component, name, input_data=None):
    """Test if a component can be JIT compiled"""
    try:
        component.eval()
        if input_data is not None:
            # Test with actual data first
            with torch.no_grad():
                output = component(input_data)
                print(f"✓ {name} - Forward pass successful, output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        
        jit_model = torch.jit.script(component)
        print(f"✓ {name} - JIT compilation successful")
        
        if input_data is not None:
            with torch.no_grad():
                jit_output = jit_model(input_data)
                print(f"✓ {name} - JIT inference successful")
        
        return True
    except Exception as e:
        print(f"✗ {name} - Failed: {str(e)[:200]}...")
        return False

def debug_full_pipeline():
    """Debug the full model pipeline step by step"""
    
    # Create test input
    test_input = torch.randn(1, 64600)  # Typical audio input
    print(f"Test input shape: {test_input.shape}")
    
    # Load model config (use a simple one first)
    try:
        config_path = "/nvme1/hungdx/KDW2V-AASISTL/configs/MDT/2025_July_25/xlsr_conformertcm_large_corpus_nov_conf-1-2.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded config: {config_path}")
    except:
        print("Could not load config file. Exiting...")
        return
    
    print("\n" + "="*50)
    print("STEP 1: Testing Base Model Components")
    print("="*50)
    
    # Create the base model
    student_model_name = config['model']['student']['name']
    model = get_model(student_model_name, device=device, **config['model']['student']['kwargs']).to(device)
    
    # Test individual components first
    print("\n1. Testing front_end (original):")
    if hasattr(model, 'front_end'):
        test_input_fe = test_input.squeeze(-1) if test_input.dim() == 3 else test_input
        test_jit_component(model.front_end, "Original front_end", test_input_fe)
    
    print("\n2. Testing other major components:")
    if hasattr(model, 'LL'):
        # Create appropriate input for LL layer
        try:
            with torch.no_grad():
                fe_output = model.front_end.extract_feat(test_input_fe)
                test_jit_component(model.LL, "Linear Layer (LL)", fe_output)
        except:
            print("Could not test LL layer - front_end extraction failed")
    
    if hasattr(model, 'first_bn'):
        test_jit_component(model.first_bn, "First BatchNorm")
    
    if hasattr(model, 'backend'):
        print("\n3. Testing backend (MyConformer):")
        test_jit_component(model.backend, "Backend (MyConformer)")
    
    print("\n" + "="*50)
    print("STEP 2: Testing Model Replacement")
    print("="*50)
    
    # Test the replacement process
    print("\n1. Testing import_fairseq_model conversion:")
    try:
        original_fairseq_model = model.front_end.model
        converted_model = import_fairseq_model(original_fairseq_model)
        test_jit_component(converted_model, "Converted Fairseq model", test_input_fe)
    except Exception as e:
        print(f"✗ Fairseq conversion failed: {str(e)[:200]}...")
    
    print("\n2. Testing W2V2_TA wrapper:")
    try:
        converted_model = import_fairseq_model(model.front_end.model)
        w2v2_ta = W2V2_TA(converted_model)
        test_jit_component(w2v2_ta, "W2V2_TA wrapper", test_input_fe)
    except Exception as e:
        print(f"✗ W2V2_TA wrapper failed: {str(e)[:200]}...")
    
    print("\n" + "="*50)
    print("STEP 3: Testing Complete Modified Model")
    print("="*50)
    
    # Apply the same modifications as export_v2.py
    try:
        model.front_end = W2V2_TA(import_fairseq_model(model.front_end.model)).to(device)
        print("✓ Successfully replaced front_end")
        
        # Test the modified complete model
        model.eval()
        padded_input = pad(test_input).unsqueeze(0)
        
        print("\n1. Testing complete modified model (forward pass):")
        with torch.no_grad():
            output = model(padded_input)
            print(f"✓ Complete model forward pass successful, output shape: {output.shape}")
        
        print("\n2. Testing complete modified model (JIT compilation):")
        test_jit_component(model, "Complete modified model", padded_input)
        
    except Exception as e:
        print(f"✗ Model modification failed: {str(e)[:200]}...")
    
    print("\n" + "="*50)
    print("STEP 4: Testing Wrapper Models")
    print("="*50)
    
    try:
        wrapper_model = WrapperModel(model).to(device)
        print("\n1. Testing WrapperModel:")
        test_jit_component(wrapper_model, "WrapperModel", test_input)
        
    except Exception as e:
        print(f"✗ WrapperModel failed: {str(e)[:200]}...")

if __name__ == "__main__":
    debug_full_pipeline() 