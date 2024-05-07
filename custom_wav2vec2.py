from torchaudio.models import wav2vec2_model
import torch
from models import Custom_Wav2Vec2_Fe, My_XLSR_FE
from student import Distil_Wav2vec2_Custom_VIB
from torchinfo import summary
from mamba_ssm import Mamba


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = Distil_Wav2vec2_Custom_VIB(device=device).to(device=device)

# # print(model(torch.randn(1, 64600).to(device=device)))
# # print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
# summary(model, input_size=(1, 64600), device=device)
# # torch.jit.script(model)
# # print(model)

batch, length, dim = 2, 201, 128
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim,  # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=16,    # Block expansion factor
).to("cuda")
# y = model(x)
print(summary(model, input_size=(batch, length, dim)))
