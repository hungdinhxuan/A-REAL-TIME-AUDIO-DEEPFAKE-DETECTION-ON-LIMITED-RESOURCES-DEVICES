import torch
from student import Self_Distil_XLSR_N_Trans_Layer_VIB
from torchdistill.core.forward_hook import ForwardHookManager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
forward_hook_manager = ForwardHookManager(device)

x = torch.randn(1, 16000).to(device)

model = Self_Distil_XLSR_N_Trans_Layer_VIB(device, num_layers=5).to(device)


forward_hook_manager.add_hook(
    model, 'LL', requires_input=False, requires_output=True)

forward_hook_manager.add_hook(
    model, 'VIB', requires_input=False, requires_output=True)

# forward
y = model(x)


io_dict = forward_hook_manager.pop_io_dict()
print(io_dict.keys())

print(len(io_dict['VIB']['output']))
# print(io_dict['LL'])
