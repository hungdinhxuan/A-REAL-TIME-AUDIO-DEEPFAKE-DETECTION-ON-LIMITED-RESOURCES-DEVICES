from teacher import W2V2_AASIST_Self_KD
from student import Distil_W2V2BASE_AASISTL_Self_KD
import torch
import torch.nn as nn

device = "cuda:0"
model = Distil_W2V2BASE_AASISTL_Self_KD(device)
nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])

# Origianl 95235402, 95244630
print("model nb_params: ", nb_params)
model = nn.DataParallel(model).to(device)
# model.load_state_dict(torch.load('/datab/hungdx/KDW2V-AASISTL/W2V2-AASIST-model.pth', map_location=device), strict=False)
print("Done!")

with torch.no_grad():
    output, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T = model(torch.randn(1, 16000).to(device))
    # Softmax for classification
    output = torch.nn.functional.softmax(output, dim=1)
    spectral_output = torch.nn.functional.softmax(spectral_output, dim=1)
    temporal_output = torch.nn.functional.softmax(temporal_output, dim=1)
    graph_output_S = torch.nn.functional.softmax(graph_output_S, dim=1)
    graph_output_T = torch.nn.functional.softmax(graph_output_T, dim=1)
    hs_gal_output_S = torch.nn.functional.softmax(hs_gal_output_S, dim=1)
    hs_gal_output_T = torch.nn.functional.softmax(hs_gal_output_T, dim=1)

    print("output: ", output)
    print("spectral_output: ", spectral_output)
    print("temporal_output: ", temporal_output)
    print("graph_output_S: ", graph_output_S)
    print("graph_output_T: ", graph_output_T)
    print("hs_gal_output_S: ", hs_gal_output_S)
    print("hs_gal_output_T: ", hs_gal_output_T)

    print("Done! Test OK!")
