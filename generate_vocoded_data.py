# Add the downloaded github repo to PYTHONPATH
import sys
import torch
sys.path.insert(1, "./project-NN-Pytorch-scripts")

import os
import sys
import pyworld as pw
import numpy as np
from core_scripts.data_io import io_tools
from core_scripts.data_io import wav_tools
from core_scripts.data_io import dsp_tools

from pretrained_voxceleb2.hifigan import model as hifigan
from pretrained_voxceleb2.hn_sinc_nsf import model as hn_sinc_nsf
from pretrained_voxceleb2.hn_sinc_nsf_hifi import model as hn_sinc_nsf_hifi
from pretrained_voxceleb2.waveglow import model as waveglow
import logging

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

# configurations fixed and used by the pre-trained models
# Don't change these settings

# sampling rate
wav_sampling_rate = 16000
# acoustic feature upsampling rate,
# this is equal to wav_sampling_rate * frame_shift
# here we use 10ms shift, thu 10ms * 16 kHz = 160
feat_upsampling_rate = 160
# FFT size for mel-spectra extraction
feat_mel_fft_size = 1024
# Frame length for Mel-spectra extraction
feat_mel_frame_length = 400
# Frame length for Pyworld (in ms), here is 10ms
feat_f0_frame_period = feat_upsampling_rate * 1000 // wav_sampling_rate

# input feature dimension of F0 and Mel (per frame)
input_f0_dim = 1
input_mel_dim = 80
# output dimension = 1 for waveform
output_dim = 1

def extract_f0(wav_data, sr=wav_sampling_rate, frame_period=feat_f0_frame_period):
    """f0 = extract_f0(wav_data, sr, frame_period)

    input: wav_data, np.array, in shape (length, 1), waveform data
    input: sr, int, sampling rate in Hz
    input: frame_period, float, frame length in ms
    output: f0, np.array, in shape (length, 1), f0 dat
    """
    x = np.array(wav_data, dtype=np.float64)


    _f0, t = pw.dio(x, sr, frame_period=frame_period)
    f0 = pw.stonemask(x, _f0, t, sr)  # pitch refinement
    return np.array(f0, dtype=np.float32)

g_mel_extractor = dsp_tools.Melspec(sf=wav_sampling_rate,
                                    fftl=feat_mel_fft_size,
                                    fl=feat_mel_frame_length,
                                    fs=feat_upsampling_rate,
                                    ver=2)
def extract_mel(wav_data, mel_extractor=g_mel_extractor):
    """mel_spec = extract_mel(wav_data, mel_extractor)

    input: wav_data, np.array, in shape (length, 1), waveform data
    input: mel_extractor, mel-spectra extractor
    output: mel_spec, np.array, in shape (length, dim), mel-spectra data
    """
    return mel_extractor.analyze(wav_data)


def extract_mel_f0_core(f_mel_extractor, f_f0_extractor, input_wav):
    """mel, f0 = extract_mel_f0_core(f_mel_extractor, f_f0_extractor, input_wav)

    This function calls f_mel_extractor and f_f0_extractor to extract Mel-spectra
    and F0 from input wav.

    input: f_mel_extractor, Mel-spectra extractor
    input: f_f0_extractor, F0 extractor,
    input: wav_data, np.array, in shape (length, 1), waveform data
    output: mel_spec, np.array, in shape (length, dim), mel-spectra data
    output: f0, np.array, in shape (length, 1), f0 dat
    """
    # extract mel or other spectral featues
    mel = f_mel_extractor(input_wav)
    # f0
    f0 = f_f0_extractor(input_wav)

    # change shape of F0 data and assign 0 to unvoiced frames
    if f0.ndim > 1:
        f0 = f0[:, 0]
    f0[np.isnan(f0)] = 0

    # adjust length (number of frames) of two features
    mel_frame = mel.shape[0]
    if f0.shape[0] < mel_frame:
        f0 = np.concatenate(
            [f0, np.zeros([mel.shape[0] - f0.shape[0]])], axis=0)
    else:
        f0 = f0[:mel_frame]
    return mel, f0

# This is the configuration in config.py
# For convenience, put it here
# Don't change these settings
class PrjConfig_hifigan:
    def __init__(self, wav_samp_rate, feat_upsamp_rate):
        self.input_dim = input_mel_dim
        self.output_dim = output_dim
        self.wav_samp_rate = wav_samp_rate
        self.input_reso = [feat_upsamp_rate]
        self.options = {'hifigan_config':
           {
               'upsample_rates': [8, 5, 2, 2],
               'upsample_kernel_sizes': [16, 9, 4, 4],
               'upsample_initial_channel': 512,
               'resblock_kernel_sizes': [3, 7, 11],
               'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
           }}
        return


class PrjConfig_hn_sinc_nsf:
    def __init__(self, wav_samp_rate, feat_upsamp_rate):
        self.input_dim = input_mel_dim + input_f0_dim
        self.output_dim = output_dim
        self.wav_samp_rate = wav_samp_rate
        self.input_reso = [feat_upsamp_rate]
        return

class PrjConfig_hn_sinc_nsf_hifi:
    def __init__(self, wav_samp_rate, feat_upsamp_rate):
        self.input_dim = input_mel_dim + input_f0_dim
        self.output_dim = output_dim
        self.wav_samp_rate = wav_samp_rate
        self.input_reso = [feat_upsamp_rate]
        return

class PrjConfig_waveglow:
    def __init__(self, wav_samp_rate, feat_upsamp_rate):
        self.input_dim = input_mel_dim
        self.output_dim = output_dim
        self.wav_samp_rate = wav_samp_rate
        self.input_reso = [feat_upsamp_rate]
        return


pretrained_version = 'pretrained_voxceleb2'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_cpu = 'cpu'

# path to the pre-trained models
#
vocoder_pretrained_path = {
    'hifigan': './{:s}/hifigan/trained_network_G.pt'.format(pretrained_version),
    'hn_sinc_nsf': './{:s}/hn_sinc_nsf/trained_network.pt'.format(pretrained_version),
    'hn_sinc_nsf_hifi': './{:s}/hn_sinc_nsf_hifi/trained_network_G.pt'.format(pretrained_version),
    'waveglow': './{:s}/waveglow/trained_network.pt'.format(pretrained_version)
    }

# configuration files
vocoder_config = {
    'hifigan': PrjConfig_hifigan(wav_sampling_rate, feat_upsampling_rate),
    'hn_sinc_nsf': PrjConfig_hn_sinc_nsf(wav_sampling_rate, feat_upsampling_rate),
    'hn_sinc_nsf_hifi': PrjConfig_hn_sinc_nsf_hifi(wav_sampling_rate, feat_upsampling_rate),
    'waveglow': PrjConfig_waveglow(wav_sampling_rate, feat_upsampling_rate),
    }




# HIFI-GAN
model_hifigan_config = vocoder_config['hifigan']
pretrained_path = vocoder_pretrained_path['hifigan']
m_hifigan_vocoder = hifigan.ModelGenerator(model_hifigan_config.input_dim, model_hifigan_config.output_dim, None, model_hifigan_config)

m_hifigan_vocoder.to(device, dtype=torch.float32)

# load pretrained model
checkpoint = torch.load(pretrained_path, map_location=device)
m_hifigan_vocoder.load_state_dict(checkpoint)
m_hifigan_vocoder.eval()
logging.info('Load HIFI-GAN vocoder model from %s', pretrained_path)



# HN-SINC-NSF
hn_sinc_nsf_config = vocoder_config['hn_sinc_nsf']
pretrained_path = vocoder_pretrained_path['hn_sinc_nsf']

hn_sinc_nsf_vocoder = hn_sinc_nsf.Model(hn_sinc_nsf_config.input_dim, hn_sinc_nsf_config.output_dim, None, hn_sinc_nsf_config)
hn_sinc_nsf_vocoder.to(device, dtype=torch.float32)

# load pretrained model
checkpoint = torch.load(pretrained_path, map_location=device)
hn_sinc_nsf_vocoder.load_state_dict(checkpoint)
hn_sinc_nsf_vocoder.eval()
logging.info('Load HN-SINC-NSF vocoder model from %s', pretrained_path)

# HN-SINC-NSF-HIFI
hn_sinc_nsf_hifi_config = vocoder_config['hn_sinc_nsf_hifi']
pretrained_path = vocoder_pretrained_path['hn_sinc_nsf_hifi']

hn_sinc_nsf_hifi_vocoder = hn_sinc_nsf_hifi.ModelGenerator(hn_sinc_nsf_hifi_config.input_dim, hn_sinc_nsf_hifi_config.output_dim, None, hn_sinc_nsf_hifi_config)

hn_sinc_nsf_hifi_vocoder.to(device, dtype=torch.float32)
checkpoint = torch.load(pretrained_path, map_location=device)
hn_sinc_nsf_hifi_vocoder.load_state_dict(checkpoint)
hn_sinc_nsf_hifi_vocoder.eval()
logging.info('Load HN-SINC-NSF-HIFI vocoder model from %s', pretrained_path)

# WAVEGLOW
waveglow_config = vocoder_config['waveglow']
pretrained_path = vocoder_pretrained_path['waveglow']

m_waveglow_vocoder = waveglow.Model(waveglow_config.input_dim, waveglow_config.output_dim, None, waveglow_config)
m_waveglow_vocoder.to(device_cpu, dtype=torch.float32)
checkpoint = torch.load(pretrained_path, map_location=device_cpu)
m_waveglow_vocoder.load_state_dict(checkpoint)
m_waveglow_vocoder.eval()
logging.info('Load WAVEGLOW vocoder model from %s', pretrained_path)


VOXCELEB2_DATASET_PATH = '/datab/hungdx/KDW2V-AASISTL/data/voxceleb2'
VOCODED_DATASET_PATH = '/datab/hungdx/KDW2V-AASISTL/data/voxceleb2_vocoded'

# Create the directory if it does not exist
if not os.path.exists(VOCODED_DATASET_PATH):
    os.makedirs(VOCODED_DATASET_PATH)

## VOXCELEB2 dataset have structure:
# /datab/hungdx/KDW2V-AASISTL/data/voxceleb2
# ├── id00012
# │   ├── 5ycNmG-uYgQ
# │   │   ├── 00001.wav
# │   │   ├── 00002.wav
# │   │   ├── 00003.wav
# │   │   ├── 00004.wav
# │   │   ├── .....
# ├── id00013
# │   ├── 5ycNmG-uYgQ
# │   │   ├── 00001.wav
# │   │   ├── 00002.wav
# │   │   ├── 00003.wav
# │   │   ├── 00004.wav
# │   │   ├── .....
# ├── .....

# We will generate the vocoded dataset with the same structure
# /datab/hungdx/KDW2V-AASISTL/data/voxceleb2_vocoded
# ├── id00012
# │   ├── 5ycNmG-uYgQ
# │   │   ├── 00001_hifigan.wav
# │   │   ├── 00001_hn_sinc_nsf.wav
# │   │   ├── 00001_hn_sinc_nsf_hifi.wav
# │   │   ├── 00001_waveglow.wav
# │   │   ├── .....
# ├── id00013
# │   ├── 5ycNmG-uY2gQ
# │   │   ├── 00001_hifigan.wav
# │   │   ├── 00001_hn_sinc_nsf.wav
# │   │   ├── 00001_hn_sinc_nsf_hifi.wav
# │   │   ├── 00001_waveglow.wav
# │   │   ├── .....
# ├── .....

for root, dirs, files in os.walk(VOXCELEB2_DATASET_PATH):
    for file in files:
        if file.endswith('.wav'):
            logging.info('Processing file: %s', file)
            wav_path = os.path.join(root, file)
            sr, input_wav = wav_tools.waveReadAsFloat(wav_path)

            # Extract Mel-spectra and F0
            mel, f0 = extract_mel_f0_core(lambda x: extract_mel(x),
                                        lambda x: extract_f0(x),
                                        input_wav)

            assert input_mel_dim == mel.shape[1], "FAIL to extract valid Mel spectrogram"

            # Generate vocoded data
            with torch.no_grad():
                # Convert to tensor
                input_feat_hifigan = torch.tensor(mel, dtype=torch.float32, device=device).unsqueeze(0)
                input_feat_wav_hn_sinc_nsf = torch.tensor(np.concatenate([mel, np.expand_dims(f0, axis=1)], axis=1), 
                          dtype=torch.float32, device=device).unsqueeze(0)
                input_feat_wav_hn_sinc_nsf_hifi = torch.tensor(np.concatenate([mel, np.expand_dims(f0, axis=1)], axis=1), 
                          dtype=torch.float32, device=device).unsqueeze(0)
                input_feat_waveglow = torch.tensor(mel, dtype=torch.float32, device=device_cpu).unsqueeze(0)

                # Generate vocoded data
                
                wav_hifigan = m_hifigan_vocoder(input_feat_hifigan)
                wav_hn_sinc_nsf = hn_sinc_nsf_vocoder(input_feat_wav_hn_sinc_nsf)
                wav_hn_sinc_nsf_hifi = hn_sinc_nsf_hifi_vocoder(input_feat_wav_hn_sinc_nsf_hifi)
                wav_waveglow = m_waveglow_vocoder.inference(input_feat_waveglow)
                
                # Convert to numpy array
                wav_hifigan = wav_hifigan[0, :, 0].cpu().numpy()

                wav_hn_sinc_nsf = wav_hn_sinc_nsf[0].cpu().numpy()

                wav_hn_sinc_nsf_hifi = wav_hn_sinc_nsf_hifi[0, :, 0].cpu().numpy()

                wav_waveglow = wav_waveglow[0, :, 0].numpy()

            # Save the vocoded data
            wav_hifigan_save_path = os.path.join(VOCODED_DATASET_PATH, root, file.replace('.wav', '_hifigan.wav')).replace('voxceleb2', 'voxceleb2_vocoded')
            wav_hn_sinc_nsf_save_path = os.path.join(VOCODED_DATASET_PATH, root, file.replace('.wav', '_hn_sinc_nsf.wav')).replace('voxceleb2', 'voxceleb2_vocoded')
            wav_hn_sinc_nsf_hifi_save_path = os.path.join(VOCODED_DATASET_PATH, root, file.replace('.wav', '_hn_sinc_nsf_hifi.wav')).replace('voxceleb2', 'voxceleb2_vocoded')
            wav_waveglow_save_path = os.path.join(VOCODED_DATASET_PATH, root, file.replace('.wav', '_waveglow.wav')).replace('voxceleb2', 'voxceleb2_vocoded')

            # Generate the directory if it does not exist
            # split last wav file name to get the directory
            if not os.path.exists(os.path.dirname(wav_hifigan_save_path)):
                os.makedirs(os.path.dirname(wav_hifigan_save_path))
                

            wav_tools.waveFloatToPCMFile(wav_hifigan, wav_hifigan_save_path)
            wav_tools.waveFloatToPCMFile(wav_hn_sinc_nsf, wav_hn_sinc_nsf_save_path)
            wav_tools.waveFloatToPCMFile(wav_hn_sinc_nsf_hifi, wav_hn_sinc_nsf_hifi_save_path)
            wav_tools.waveFloatToPCMFile(wav_waveglow, wav_waveglow_save_path)

            # wav_tools.waveFloatToPCMFile(wav_hifigan, os.path.join(VOCODED_DATASET_PATH, root, file))
            # wav_tools.waveFloatToPCMFile(wav_hn_sinc_nsf, os.path.join(VOCODED_DATASET_PATH, root, file))
            # wav_tools.waveFloatToPCMFile(wav_hn_sinc_nsf_hifi, os.path.join(VOCODED_DATASET_PATH, root, file))
            # wav_tools.waveFloatToPCMFile(wav_waveglow, os.path.join(VOCODED_DATASET_PATH, root, file))
            print('Vocoded file: ', file)