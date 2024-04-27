from torchaudio.models.wav2vec2.utils import import_fairseq_model
import fairseq


# Load XLSR Wav2Vec2 model

# Load the model
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    ['/datab/hungdx/KDW2V-AASISTL/xlsr2_300m.pt'])
model = model[0]


torch_audio_model = import_fairseq_model(model)
