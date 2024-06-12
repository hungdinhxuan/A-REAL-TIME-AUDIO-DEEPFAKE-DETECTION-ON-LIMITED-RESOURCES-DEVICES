import torch
from wav2vec2_vib import Model as Wav2Vec2VIB
from student import Distil_XLSR_N_Trans_Layer_VIB
import torch
from torchinfo import summary
import torch_pruning as tp
from main import produce_evaluation_file, W2V2_TA
from torchaudio.models.wav2vec2.utils import import_fairseq_model
from torch import nn


device, ssl_cpkt_path = "cuda", "/datab/hungdx/KDW2V-AASISTL/xlsr2_300m.pt"
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 1]).to(device))

model = Distil_XLSR_N_Trans_Layer_VIB(
    device, ssl_cpkt_path, num_layers=5).to(device)

model.ssl_model = W2V2_TA(import_fairseq_model(
    model.ssl_model.model
)).to(device)

# model.eval()

example_inputs = torch.randn(1, 64600).to(device)

# print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


# or GroupNormImportance(p=2), GroupHessianImportance(), etc.
imp = tp.importance.GroupTaylorImportance()

# 2. Initialize a pruner with the model and the importance criterion
ignored_layers = []
for m in model.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 2:
        ignored_layers.append(m)  # DO NOT prune the final classifier!

pruner = tp.pruner.MetaPruner(  # We can always choose MetaPruner if sparse training is not required.
    model,
    example_inputs,
    importance=imp,
    # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    pruning_ratio=0.5,
    # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
    ignored_layers=ignored_layers,
)

base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)

print(f"<Before Pruning> MACs: {base_macs}, #Params: {base_nparams}")
pruner.update_regularizer()  # <== initialize regularizer
if isinstance(imp, tp.importance.GroupTaylorImportance):
    # Taylor expansion requires gradients for importance estimation
    # A dummy loss, please replace this line with your loss function and data!
    print("Calculating importance...")
    output = model(example_inputs)
    print(f"Output: {output}")
    loss = criterion(output, torch.randint(0, 2, (1,)).to(device))
    loss.backward()  # before pruner.step()
    pruner.regularize(model, loss)
# pruner.step()
macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)

print(f"<After Pruning> MACs: {macs}, #Params: {nparams}")

# # 4. Save & Load
# model.zero_grad()  # clear gradients
# # We can not use .state_dict as the model structure is changed.
# torch.save(model, 'model.pth')
# model = torch.load('model.pth')  # load the pruned model
