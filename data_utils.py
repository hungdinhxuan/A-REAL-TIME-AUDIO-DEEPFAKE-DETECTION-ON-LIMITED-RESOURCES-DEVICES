import os
import numpy as np
import torch
import torchaudio
from torchaudio.utils import download_asset
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav
from torch.utils.data import DataLoader, WeightedRandomSampler
import audiomentations as aa
import logging

import core_scripts.data_io.wav_tools as nii_wav_tools
from core_scripts.data_io import wav_augmentation as nii_wav_aug

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

SAMPLE_RATE = 16000
PADDING_SIZE = 64600  # 4 seconds of audio

""" backup
def genSpoof_list(dir_meta, is_train=False, is_eval=False, tts_only=True):
    
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            _, key, _, att, label = line.strip().split(' ')
            if (tts_only):
                if((att=="A05") or (att=="A06")):
                    continue
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

    elif(is_eval):
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, att, label = line.strip().split(' ')
            if (tts_only):
                if((att=="A05") or (att=="A06")):
                    continue
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list
"""


class bio_emb(nn.Module):
    def __init__(self, device):
        super(bio_emb, self).__init__()
        self.device = device

    def get_output(self, out_file):
        # load teacher score
        emb = torch.load(out_file)
        # emb is an array (list). we need to convert it to tensor
        emb = torch.tensor(emb, dtype=torch.float32, device=self.device)
        return emb

# def genSpoof_list(dir_meta, is_train=False, is_eval=False, tts_only=False):

#     d_meta = {}
#     file_list = []
#     with open(dir_meta, 'r') as f:
#         l_meta = f.readlines()

#     if (is_train):
#         for line in l_meta:
#             _, key, _, att, label = line.strip().split(' ')
#             if (tts_only):
#                 if((att=="A05") or (att=="A06")):
#                     continue
#             file_list.append(key)
#             d_meta[key] = 1 if label == 'bonafide' else 0
#         return d_meta, file_list

#     elif(is_eval):
#         for line in l_meta:
#             key = line.strip()
#             file_list.append(key)
#         return file_list
#     else:
#         for line in l_meta:
#             _, key, _, att, label = line.strip().split(' ')
#             if (tts_only):
#                 if((att=="A05") or (att=="A06")):
#                     continue
#             file_list.append(key)
#             d_meta[key] = 1 if label == 'bonafide' else 0
#         return d_meta, file_list


def genSpoof_list(dir_meta, is_train=False, is_eval=False, num_eval_samples=60000):

    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            _, key, _, _, label = line.strip().split()

            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

    elif (is_eval):
        # Randomly num_eval_samples  samples from eval set

        if num_eval_samples > len(l_meta) or num_eval_samples < 0:
            num_eval_samples = len(l_meta)

        np.random.seed(0)
        np.random.shuffle(l_meta)
        l_meta = l_meta[:num_eval_samples]

        for line in l_meta:
            key = line.strip()
            # _,key,_,_,label = line.strip().split()
            file_list.append(key)
            # d_meta[key] = 1 if label == 'bonafide' else 0
        return None, file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split()

            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list


def genSpoof_in_the_wild_list(dir_meta):

    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        key, label = line.strip().split()
        file_list.append(key)
    return file_list


def genSpoof_list_v2(dir_meta, is_train=False, is_dev=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            key, subset, _, label = line.strip().split()
            if subset == "train":
                file_list.append(key)
                d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list
    if (is_dev):
        for line in l_meta:
            key, subset, _, label = line.strip().split()
            if subset == "dev":
                file_list.append(key)
                d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

    elif (is_eval):
        for line in l_meta:
            key, subset, _, label = line.strip().split()
            if subset == "eval":
                file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            key, subset, _, label = line.strip().split()
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list
# ------------------------------------------


def genList(dir_meta, is_train=False, is_eval=False, is_dev=False):
    # bonafide: 1, spoof: 0
    d_meta = {}
    file_list = []
    # get dir of metafile only
    dir_meta = os.path.dirname(dir_meta)
    if is_train:
        metafile = os.path.join(dir_meta, 'scp/train_bonafide.lst')
    elif is_dev:
        metafile = os.path.join(dir_meta, 'scp/dev_bonafide.lst')
    elif is_eval:
        metafile = os.path.join(dir_meta, 'scp/test.lst')

    with open(metafile, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            key = line.strip().split()
            file_list.append(key[0])
        return [], file_list

    if (is_dev):
        for line in l_meta:
            key = line.strip().split()
            file_list.append(key[0])
        return [], file_list

    elif (is_eval):
        for line in l_meta:
            key = line.strip().split()
            file_list.append(key[0])

        return [], file_list
# ------------------------------------------


def pad(x, max_len=PADDING_SIZE):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo = algo
        self.args = args
        self.cut = PADDING_SIZE  # take ~4 sec audio (PADDING_SIZE samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+utt_id+'.flac', sr=SAMPLE_RATE)
        Y = process_Rawboost_feature(X, fs, self.args, self.algo)
        X_pad = pad(Y, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]

        return x_inp, target


class Dataset_ASVspoof2019_train_augment2(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = PADDING_SIZE  # take ~4 sec audio (PADDING_SIZE samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+utt_id+'.flac', sr=SAMPLE_RATE)
        Y = process_audiomentations(X, fs)
        X_pad = pad(Y, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target


class Dataset_ASVspoof2019_train_emb(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo, is_train=True):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo = algo
        self.args = args
        self.cut = PADDING_SIZE  # take ~4 sec audio (PADDING_SIZE samples)
        self.is_train = is_train

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+utt_id+'.flac', sr=SAMPLE_RATE)
        Y = process_Rawboost_feature(X, fs, self.args, self.algo)
        X_pad = pad(Y, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]

        # Load teacher emb
        if self.is_train:
            folder_to_extract = './teacher_cosine_emb'
            save_path = os.path.join(folder_to_extract, utt_id+".pt")
            emb = torch.load(save_path, map_location="cpu")

            return x_inp, target, emb
        else:
            return x_inp, target


class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
            '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = PADDING_SIZE  # take ~4 sec audio (PADDING_SIZE samples)
        key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+key+'.flac', sr=SAMPLE_RATE)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key


class Dataset_ASVspoof2021_with_labels_eval(Dataset):
    def __init__(self, list_IDs, base_dir, labels):
        '''self.list_IDs	: list of strings (each string: utt key),
            '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.labels = labels

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = PADDING_SIZE  # take ~4 sec audio (PADDING_SIZE samples)
        key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+key+'.flac', sr=SAMPLE_RATE)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key, self.labels[key]


class Dataset_ASVspoof2019_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = PADDING_SIZE  # take ~4 sec audio (PADDING_SIZE samples)
        keys = self.list_IDs[index].strip().split(' ')
        # key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+keys[1]+'.flac', sr=SAMPLE_RATE)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)

        return x_inp, self.list_IDs[index]


def pad_v2(x, utt_id, max_len=PADDING_SIZE):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    if x_len < 1:
        logging.error("file {} has 0 size".format(utt_id))
        exit(0)

    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_cnsl(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        '''self.list_IDs	: list of strings (each string: utt key),
            self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo = algo
        self.args = args
        self.cut = PADDING_SIZE  # take ~4 sec audio (PADDING_SIZE samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + "/" + utt_id, sr=16000)
        Y = process_Rawboost_feature(X, fs, self.args, self.algo)
        # Y=process_audiomentations(X,fs)
        X_pad = pad_v2(Y, utt_id, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]

        return x_inp, target


class Dataset_cnsl_augment(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
            self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.args = args
        self.cut = PADDING_SIZE  # take ~4 sec audio (PADDING_SIZE samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + "/" + utt_id, sr=16000)
        Y = process_audiomentations(X, fs)
        X_pad = pad_v2(Y, utt_id, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]

        return x_inp, target


class Dataset_cnsl_augment_v2(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
            self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.args = args
        self.cut = PADDING_SIZE  # take ~4 sec audio (PADDING_SIZE samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + "/" + utt_id, sr=16000)
        Y = process_audiomentations_v2(X, fs)
        X_pad = pad_v2(Y, utt_id, self.cut)
        x_inp = Tensor(X_pad.copy())
        target = self.labels[utt_id]

        return x_inp, target


class Dataset_cnsl_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
            '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = PADDING_SIZE  # take ~4 sec audio (PADDING_SIZE samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + "/" + utt_id, sr=16000)
        X_pad = pad_v2(X, utt_id, self.cut)

        x_inp = Tensor(X_pad)
        return x_inp, utt_id


class Dataset_in_the_wild_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
            '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = PADDING_SIZE  # take ~4 sec audio (PADDING_SIZE samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        utt_id = self.list_IDs[index]
        X, fs = librosa.load(
            self.base_dir + "/in_the_wild/" + utt_id, sr=16000)
        X_pad = pad_v2(X, utt_id, self.cut)

        x_inp = Tensor(X_pad)
        return x_inp, utt_id


class Dataset_cnsl_augment_contrastive(Dataset):
    """
    Dataset class for contrastive learning
    """

    def __init__(self, list_ids, labels, base_dir):
        """
        list_ids: list of strings (each string: utt key),
        labels: dictionary (key: utt key, value: label integer)
        """
        self.list_ids = list_ids
        self.labels = labels
        self.base_dir = base_dir
        self.cut = PADDING_SIZE  # take ~4 sec audio (PADDING_SIZE samples)

        # Calculate weights for WeightedRandomSampler
        class_counts = np.bincount(list(self.labels.values()))
        class_weights = 1. / class_counts
        self.weights = [class_weights[self.labels[utt_id]]
                        for utt_id in self.list_ids]

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        utt_id = self.list_ids[index]
        audio, fs = librosa.load(f"{self.base_dir}/{utt_id}", sr=16000)
        processed_audio = process_audiomentations(audio, fs)
        padded_audio = pad_v2(processed_audio, utt_id, self.cut)
        audio_tensor = Tensor(padded_audio)
        target = self.labels[utt_id]

        return audio_tensor, target

    def get_sampler(self):
        return WeightedRandomSampler(self.weights, len(self.weights))

# --------------Audiomentations---------------------------#


def process_audiomentations(feature, sr):
    """ DA using audiomentations library    
    """
    # aa.ApplyImpulseResponse(ir_path="/path/to/sound_folder", p=1.0)

    augment = aa.Compose([
        aa.AddBackgroundNoise(
            sounds_path="/nfs/datab/longnv/musan/mix", p=0.75),
        aa.AdjustDuration(duration_seconds=4, p=1.0, padding_mode="wrap"),
        aa.TimeStretch(min_rate=0.8, max_rate=1.2,
                       leave_length_unchanged=True, p=0.75),
        aa.Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.75),
        aa.AirAbsorption(min_distance=1.0, max_distance=20.0, p=0.75),
        aa.TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=0.5),
        aa.Mp3Compression(min_bitrate=96, max_bitrate=320, p=0.3)
    ])
    return augment(samples=feature, sample_rate=sr)


def process_audiomentations_v2(feature, sr):
    """ DA using audiomentations library    
    """
    # aa.ApplyImpulseResponse(ir_path="/path/to/sound_folder", p=1.0)
    augment = aa.Compose([
        aa.AddGaussianSNR(
            min_snr_db=5.0,
            max_snr_db=40.0,
            p=0.5
        ),
        aa.PitchShift(
            min_semitones=-12,
            max_semitones=12,
            p=0.5
        ),
        aa.Reverse(p=0.75),
        # aa.RepeatPart(mode="replace", p=0.75)
    ])
    return augment(samples=feature, sample_rate=sr)


def process_torchaudio_augment(feature, sr):
    """ DA using torchaudio library """
    SAMPLE_WAV = download_asset(
        "tutorial-assets/steam-train-whistle-daniel_simon.wav")
    SAMPLE_RIR = download_asset(
        "tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
    SAMPLE_SPEECH = download_asset(
        "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav")
    SAMPLE_NOISE = download_asset(
        "tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")


def process_Rawboost_feature(feature, sr, args, algo):

    # Data process by Convolutive noise (1st algo)
    if algo == 1:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)

    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by coloured additive noise (3rd algo)
    elif algo == 3:

        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
                                     args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by all 3 algo. together in series (1+2+3)
    elif algo == 4:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
                                     args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by 1st two algo. together in series (1+2)
    elif algo == 5:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo == 6:

        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                        args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
                                     args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo == 7:

        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands, args.minF,
                                     args.maxF, args.minBW, args.maxBW, args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)

    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo == 8:

        feature1 = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF, args.minBW, args.maxBW,
                                         args.minCoeff, args.maxCoeff, args.minG, args.maxG, args.minBiasLinNonLin, args.maxBiasLinNonLin, sr)
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para = feature1+feature2
        feature = normWav(feature_para, 0)  # normalized resultant waveform

    # original data without Rawboost processing
    else:

        feature = feature

    return feature


def get_train_dev_dataloader_v2(args):
    # define train dataloader
    d_label_trn, file_train = genSpoof_list(dir_meta=os.path.join(
        args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'), is_train=True, is_eval=False)

    print('no. of training trials', len(file_train))

    train_set = Dataset_ASVspoof2019_train_emb(args, list_IDs=file_train, labels=d_label_trn, base_dir=os.path.join(
        args.database_path+'ASVspoof2019_LA_train/'), algo=args.algo)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=True)

    del train_set, d_label_trn

    # define dev (validation) dataloader

    d_label_dev, file_dev = genSpoof_list(dir_meta=os.path.join(
        args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'), is_train=False, is_eval=False)

    print('no. of validation trials', len(file_dev))

    dev_set = Dataset_ASVspoof2019_train_emb(args, list_IDs=file_dev, labels=d_label_dev, base_dir=os.path.join(
        args.database_path+'ASVspoof2019_LA_dev/'), algo=args.algo, is_train=False)

    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, num_workers=8, shuffle=False)

    del dev_set, d_label_dev

    return train_loader, dev_loader


def get_train_dev_dataloader(args, augment='rawboost', dataset='LA19'):

    # define train dataloader
    if dataset == 'LA19':
        logging.info("Using LA19 dataset")
        d_label_trn, file_train = genSpoof_list(dir_meta=os.path.join(
            args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'), is_train=True, is_eval=False)

        print('no. of training trials', len(file_train))

        if augment == 'rawboost':
            train_set = Dataset_ASVspoof2019_train(args, list_IDs=file_train, labels=d_label_trn, base_dir=os.path.join(
                args.database_path+'ASVspoof2019_LA_train/'), algo=args.algo)
        elif augment == 'audiomentations':
            train_set = Dataset_ASVspoof2019_train_augment2(
                list_IDs=file_train, labels=d_label_trn, base_dir=os.path.join(args.database_path+'ASVspoof2019_LA_train/'))

        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  num_workers=args.workers, shuffle=True, drop_last=True)

        del train_set, d_label_trn

        # define dev (validation) dataloader

        d_label_dev, file_dev = genSpoof_list(dir_meta=os.path.join(
            args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'), is_train=False, is_eval=False)

        print('no. of validation trials', len(file_dev))

        dev_set = Dataset_ASVspoof2019_train(args, list_IDs=file_dev, labels=d_label_dev, base_dir=os.path.join(
            args.database_path+'ASVspoof2019_LA_dev/'), algo=args.algo)

        dev_loader = DataLoader(
            dev_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)

        del dev_set, d_label_dev

        return train_loader, dev_loader

    else:
        logging.info("Using CNSL dataset")
        # define train dataloader
        d_label_trn, file_train = genSpoof_list_v2(dir_meta=os.path.join(args.database_path, args.protocols_path),
                                                   is_train=True, is_dev=False, is_eval=False)

        print('no. of training trials', len(file_train))

        if augment == 'rawboost':
            logging.info(
                "Using Rawboost for data augmentation with algo: {}".format(args.algo))
            train_set = Dataset_cnsl(
                args, list_IDs=file_train, labels=d_label_trn, base_dir=args.database_path+'/', algo=args.algo)
        elif augment == 'audiomentations':
            train_set = Dataset_cnsl_augment(
                args, list_IDs=file_train, labels=d_label_trn, base_dir=args.database_path+'/')
        elif augment == 'audiomentations_v2':
            train_set = Dataset_cnsl_augment_v2(
                args, list_IDs=file_train, labels=d_label_trn, base_dir=args.database_path+'/')
        else:
            logging.error("Invalid augment type: {}".format(augment))
            exit(0)

        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  num_workers=args.workers, shuffle=True, drop_last=True)

        del train_set, d_label_trn

        # define validation dataloader
        d_label_dev, file_dev = genSpoof_list_v2(dir_meta=os.path.join(args.database_path, args.protocols_path),
                                                 is_train=False, is_dev=True, is_eval=False)

        print('no. of validation trials', len(file_dev))

        dev_set = Dataset_cnsl(args, list_IDs=file_dev, labels=d_label_dev,
                               base_dir=args.database_path+'/', algo=args.algo)
        dev_loader = DataLoader(dev_set, batch_size=100,
                                num_workers=args.workers, shuffle=False)
        del dev_set, d_label_dev
        return train_loader, dev_loader


def get_train_dev_dataloader_contrastive(args):
    logging.info("Using CNSL dataset with contrastive learning")
    # define train dataloader
    d_label_trn, file_train = genList(dir_meta=os.path.join(args.database_path, args.protocols_path),
                                      is_train=True, is_dev=False, is_eval=False)

    print('no. of training trials', len(file_train))

    # train_set = Dataset_cnsl_augment_contrastive(
    #     file_train, d_label_trn, base_dir=args.database_path+'/')

    train_set = Dataset_for(args, list_IDs=file_train, labels=d_label_trn,
                            base_dir=args.database_path+'/', vocoders=['hifigan', 'hn-sinc-nsf-hifi', 'waveglow'])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=True, drop_last=True)

    d_label_dev, file_dev = genList(dir_meta=os.path.join(args.database_path, args.protocols_path),
                                    is_train=False, is_dev=True, is_eval=False)

    print('no. of validation trials', len(file_dev))

    dev_set = Dataset_for(args, list_IDs=file_dev, labels=d_label_dev,
                          base_dir=args.database_path+'/', vocoders=['hifigan', 'hn-sinc-nsf-hifi', 'waveglow'])
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    del dev_set, d_label_dev
    return train_loader, dev_loader


# -----------------------------Contrastive learning---------------------------------#

def RawBoost12(x, args, sr=16000, audio_path=None):
    """ RawBoost12: RawBoost with 12dB gain
    """
    return process_Rawboost_feature(x, sr, args, 3)


class Dataset_for(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo=5, vocoders=[],
                 augmentation_methods=[], num_additional_real=2, num_additional_spoof=2,
                 trim_length=64000, wav_samp_rate=16000, noise_path=None, rir_path=None,
                 aug_dir=None, online_aug=False, repeat_pad=True):
        """
        Args:
            list_IDs (string): Path to the .lst file with real audio filenames.
            vocoders (list): list of vocoder names.
            augmentation_methods (list): List of augmentation methods to apply.
        """
        self.args = args
        self.args.noise_path = noise_path
        self.args.rir_path = rir_path
        self.args.aug_dir = aug_dir
        self.args.online_aug = online_aug
        self.list_IDs = list_IDs
        self.bonafide_dir = os.path.join(base_dir, 'bonafide')
        self.vocoded_dir = os.path.join(base_dir, 'vocoded')
        # list available spoof samples (only .wav files)
        self.spoof_dir = os.path.join(base_dir, 'spoof')
        self.spoof_list = [f for f in os.listdir(self.spoof_dir) if os.path.isfile(
            os.path.join(self.spoof_dir, f)) and (f.endswith('.wav') or f.endswith('.flac'))]
        self.repeat_pad = repeat_pad
        self.trim_length = trim_length
        self.sample_rate = wav_samp_rate

        self.vocoders = vocoders
        print("vocoders:", vocoders)
        self.num_additional_spoof = num_additional_spoof
        self.num_additional_real = num_additional_real
        self.augmentation_methods = augmentation_methods

        if len(augmentation_methods) < 1:
            # using default augmentation method RawBoostWrapper12
            self.augmentation_methods = ["RawBoost12"]

    def load_audio(self, file_path):
        waveform, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        # _, waveform = nii_wav_tools.waveReadAsFloat(file_path)
        return waveform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        # Anchor real audio sample
        real_audio_file = os.path.join(self.bonafide_dir, self.list_IDs[idx])
        real_audio = self.load_audio(real_audio_file)

        # Vocoded audio samples as negative data
        vocoded_audio_files = [os.path.join(
            self.vocoded_dir, vf + "_" + self.list_IDs[idx]) for vf in self.vocoders]
        vocoded_audios = []
        augmented_vocoded_audios = []
        for vf in vocoded_audio_files:
            vocoded_audio = self.load_audio(vf)
            vocoded_audios.append(np.expand_dims(vocoded_audio, axis=1))
            # Augmented vocoded samples as negative data with first augmentation method
            augmented_vocoded_audio = globals()[self.augmentation_methods[0]](
                vocoded_audio, self.args, self.sample_rate, audio_path=vf)
            augmented_vocoded_audios.append(
                np.expand_dims(augmented_vocoded_audio, axis=1))

        # Augmented real samples as positive data
        augmented_audios = []
        for augment in self.augmentation_methods:
            augmented_audio = globals()[augment](
                real_audio, self.args, self.sample_rate, audio_path=real_audio_file)
            # print("aug audio shape",augmented_audio.shape)
            augmented_audios.append(np.expand_dims(augmented_audio, axis=1))

        # Additional real audio samples as positive data
        idxs = list(range(len(self.list_IDs)))
        idxs.remove(idx)  # remove the current audio index
        additional_idxs = np.random.choice(
            idxs, self.num_additional_real, replace=False)
        additional_audios = [np.expand_dims(self.load_audio(os.path.join(
            self.bonafide_dir, self.list_IDs[i])), axis=1) for i in additional_idxs]

        # Additional spoof audio samples as negative data
        additional_spoof_idxs = np.random.choice(
            self.spoof_list, self.num_additional_spoof, replace=False)
        additional_spoofs = [np.expand_dims(self.load_audio(os.path.join(
            self.spoof_dir, i)), axis=1) for i in additional_spoof_idxs]

        # merge all the data
        batch_data = [np.expand_dims(real_audio, axis=1)] + augmented_audios + \
            additional_audios + vocoded_audios + augmented_vocoded_audios + additional_spoofs
        batch_data = nii_wav_aug.batch_pad_for_multiview(
            batch_data, self.sample_rate, self.trim_length, random_trim_nosil=True, repeat_pad=self.repeat_pad)
        batch_data = np.concatenate(batch_data, axis=1)
        # print("batch_data.shape", batch_data.shape)

        # return will be anchor ID, batch data and label
        batch_data = Tensor(batch_data)
        # label is 1 for anchor and positive, 0 for vocoded
        label = [1] * (len(augmented_audios) + len(additional_audios) + 1) + \
            [0] * (len(self.vocoders*2) + len(additional_spoofs))
        # print("label", label)
        return batch_data, Tensor(label)
