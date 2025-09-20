from torchdistill.models.registry import register_model
from conformer_tcm.model import MyConformer
from torch import nn
import torch
from transformers import Wav2Vec2Model
def middle_indices(array_length, number_of_middle_elements):
    # Calculate the start index
    start_index = (array_length - number_of_middle_elements) // 2
    # Calculate the end index
    end_index = start_index + number_of_middle_elements
    # Create a list of the middle indices
    middle_indices = list(range(start_index, end_index))
    return middle_indices

class HF_Wav2vec2(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.num_layers = kwargs.get('num_layers', 12)
        self.order = kwargs.get('order', 'first')
        self.custom_order = kwargs.get('custom_order', None)
        self.out_dim = 768
        if self.order == 'last':
            # Get the last n layers
            self.model.encoder.layers = self.model.encoder.layers[-self.num_layers:]
        elif self.order == 'first':
            # Get the first n layers
            self.model.encoder.layers = self.model.encoder.layers[:self.num_layers]
        elif self.order == 'middle':
            indices = middle_indices(12, self.num_layers)

            self.model.encoder.layers = nn.ModuleList([
                self.model.encoder.layers[i] for i in indices])
        else:
            if self.custom_order is None:
                raise ValueError(
                    "Custom order must be provided as a list of integers (0-23).")

            # Check if the custom order is valid
            if type(self.custom_order) != list:
                raise ValueError("Custom order must be a list of integers.")

            # if len(self.custom_order) != self.num_layers:
            #     raise ValueError(
            #         "Length of custom order must be less than or equal to the number of layers.")
            self.model.encoder.layers = nn.ModuleList([
                self.model.encoder.layers[i] for i in self.custom_order])


    def forward(self, x, layerwise=False):
        x = self.model(x, output_hidden_states=True)
        layer_results, last_hidden_state = x.hidden_states, x.last_hidden_state
        if layerwise:
            # ignore the first layer because it is the input layer from cov
            layer_results = layer_results[1:]
            layer_results = torch.stack(layer_results, dim=1)
            return last_hidden_state, layer_results
        else:
            return last_hidden_state

@register_model(key='Distil_Wav2vec2_HF_N_Trans_Layer_ConformerTCM')
class Distil_Wav2vec2_HF_N_Trans_Layer_ConformerTCM(nn.Module):
    def __init__(self, device,  args, **kwargs):
        super().__init__()
        self.front_end = HF_Wav2vec2(device, **kwargs)
        self.LL = nn.Linear(768, args['emb_size'])
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.backend=MyConformer(**args)
    
    def get_front_end_hidden_states(self, x):
        x_ssl_feat = self.front_end(x, output_hidden_states=True)
        x_ssl_feat = x_ssl_feat.hidden_states
        return x_ssl_feat # (n_layers, bs, frame_number, 768)
    
    def forward(self, x, layerwise=False):
        if layerwise:
            x_ssl_feat, layer_results = self.front_end(x, layerwise=layerwise)
        else:
            x_ssl_feat = self.front_end(x)
        x = self.LL(x_ssl_feat)
        x = x.unsqueeze(dim=1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out = self.backend(x)
        if layerwise:
            return out, layer_results
        else:
            return out


if __name__ == "__main__":
    model = HF_Wav2vec2(device='cuda', args={'emb_size': 128})
    print(model)