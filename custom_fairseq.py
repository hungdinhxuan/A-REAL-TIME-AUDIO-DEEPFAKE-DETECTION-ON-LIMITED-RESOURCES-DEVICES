from torchaudio.models.wav2vec2 import wav2vec2_base
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
from models import SSLModelBase
import torchaudio
from torchaudio.models.wav2vec2.utils import import_fairseq_model
import torch
import fairseq

# bundle = WAV2VEC2_ASR_BASE_960H

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = bundle.get_model().to(device)

# ssl_model = SSLModelBase(device)
# ssl_model = ssl_model.to(device)

# with torch.inference_mode():
#     waveform = torch.randn(1, 16000).to(device)
#     features, _ = model.extract_features(waveform)
#     features = features[0]
#     feat2 = ssl_model(waveform)

#     # torch.jit.script(model)
#     print(features)
#     print(feat2)
model_file = './wav2vec_small.pt'
model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
device = "cuda" if torch.cuda.is_available() else "cpu"
original = model[0].to(device)
imported = import_fairseq_model(original).to(device)
waveform = torch.randn(1,16000).to(device)
features, _ = imported.extract_features(waveform)
features = features[0]

print(features.shape)

# reference = original.feature_extractor(waveform).transpose(1, 2)

# print(reference.shape)
reference2 = original(waveform, mask=False, features_only=True)['x']
torch.testing.assert_close(features, reference2)
# torch.testing.assert_close(reference, reference2)

# torch.jit.script(imported)