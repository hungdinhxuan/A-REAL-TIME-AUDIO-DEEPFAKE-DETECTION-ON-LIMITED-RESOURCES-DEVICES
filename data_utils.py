import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from torch.utils.data import DataLoader
import audiomentations as aa
import logging

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

SAMPLE_RATE = 16000

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
def genSpoof_list( dir_meta,is_train=False,is_eval=False, num_eval_samples=60000):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    
    elif(is_eval):
        # Randomly num_eval_samples  samples from eval set

        if num_eval_samples > len(l_meta) or num_eval_samples < 0:
            num_eval_samples = len(l_meta)
        
        np.random.seed(0)
        np.random.shuffle(l_meta)
        l_meta = l_meta[:num_eval_samples]
        
        for line in l_meta:
            key= line.strip()
            # _,key,_,_,label = line.strip().split()
            file_list.append(key)
            # d_meta[key] = 1 if label == 'bonafide' else 0
        return None,file_list
    else:
        for line in l_meta:
             _,key,_,_,label = line.strip().split()
             
             file_list.append(key)
             d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list

def genSpoof_list_v2(dir_meta, is_train=False, is_dev=False, is_eval=False):
    
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
             key, subset, _ ,label = line.strip().split()
             if subset == "train":
                 file_list.append(key)
                 d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    if (is_dev):
        for line in l_meta:
             key, subset, _,label = line.strip().split()
             if subset == "dev":
                 file_list.append(key)
                 d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list
    
    elif(is_eval):
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
        return d_meta,file_list

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	

class Dataset_ASVspoof2019_train(Dataset):
	def __init__(self,args,list_IDs, labels, base_dir,algo):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir = base_dir
            self.algo=algo
            self.args=args
            self.cut=64600 # take ~4 sec audio (64600 samples)

	def __len__(self):
           return len(self.list_IDs)


	def __getitem__(self, index):
            
            utt_id = self.list_IDs[index]
            X,fs = librosa.load(self.base_dir+utt_id+'.flac', sr=SAMPLE_RATE) 
            Y=process_Rawboost_feature(X,fs,self.args,self.algo)
            X_pad= pad(Y,self.cut)
            x_inp= Tensor(X_pad)
            target = self.labels[utt_id]
            
            return x_inp, target            

class Dataset_ASVspoof2019_train_augment2(Dataset):
	def __init__(self,list_IDs, labels, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir = base_dir
            self.cut=64600 # take ~4 sec audio (64600 samples)

	def __len__(self):
           return len(self.list_IDs)

	def __getitem__(self, index):
            
            utt_id = self.list_IDs[index]
            X,fs = librosa.load(self.base_dir+utt_id+'.flac', sr=SAMPLE_RATE) 
            Y=process_audiomentations(X,fs)
            X_pad= pad(Y,self.cut)
            x_inp= Tensor(X_pad)
            target = self.labels[utt_id]
            return x_inp, target 

class Dataset_ASVspoof2019_train_emb(Dataset):
	def __init__(self,args,list_IDs, labels, base_dir,algo, is_train=True):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir = base_dir
            self.algo=algo
            self.args=args
            self.cut=64600 # take ~4 sec audio (64600 samples)
            self.is_train = is_train

	def __len__(self):
           return len(self.list_IDs)


	def __getitem__(self, index):
            
            utt_id = self.list_IDs[index]
            X,fs = librosa.load(self.base_dir+utt_id+'.flac', sr=SAMPLE_RATE) 
            Y=process_Rawboost_feature(X,fs,self.args,self.algo)
            X_pad= pad(Y,self.cut)
            x_inp= Tensor(X_pad)
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
        self.cut=64600 # take ~4 sec audio (64600 samples)
        key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+key+'.flac', sr=SAMPLE_RATE)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp,key 

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
        self.cut=64600 # take ~4 sec audio (64600 samples)
        key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+key+'.flac', sr=SAMPLE_RATE)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp,key, self.labels[key]           
        
            
class Dataset_ASVspoof2019_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        keys = self.list_IDs[index].strip().split(' ')
        # key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+keys[1]+'.flac', sr=SAMPLE_RATE)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        
        return x_inp, self.list_IDs[index]        



def pad_v2(x, utt_id, max_len=64600):
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
    def __init__(self,args,list_IDs, labels, base_dir, algo):
        '''self.list_IDs	: list of strings (each string: utt key),
            self.labels      : dictionary (key: utt key, value: label integer)'''
            
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo=algo
        self.args=args
        self.cut=64600 # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):
            
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + "/" + utt_id, sr=16000)
        Y=process_Rawboost_feature(X,fs,self.args,self.algo)
        # Y=process_audiomentations(X,fs)
        X_pad= pad_v2(Y,utt_id,self.cut)
        x_inp= Tensor(X_pad)
        target = self.labels[utt_id]
        
        return x_inp, target

class Dataset_cnsl_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
            '''
            
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut=64600 # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)


    def __getitem__(self, index):
            
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir + "/" + utt_id, sr=16000)
        X_pad = pad(X,utt_id,self.cut)
        
        x_inp = Tensor(X_pad)
        return x_inp, utt_id


#--------------Audiomentations---------------------------#
def process_audiomentations(feature, sr):
    """ DA using audiomentations library    
    """
    # aa.ApplyImpulseResponse(ir_path="/path/to/sound_folder", p=1.0)

    augment = aa.Compose([
        aa.AddBackgroundNoise(sounds_path="/nfs/datab/longnv/musan/mix", p=0.75),
        aa.AdjustDuration(duration_seconds=4, p=1.0, padding_mode="wrap"), 
        aa.TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=True, p=0.75),
        aa.Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.75),
        aa.AirAbsorption(min_distance=1.0, max_distance=20.0, p=0.75),
        aa.TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=0.5),
        aa.Mp3Compression(min_bitrate=96, max_bitrate=320, p=0.3)
        ])
    return augment(samples=feature, sample_rate=sr)

def process_Rawboost_feature(feature, sr,args,algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo==1:

        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo==2:
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature


def get_train_dev_dataloader_v2(args):
       # define train dataloader
    d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),is_train=True,is_eval=False)
    
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_ASVspoof2019_train_emb(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_train/'),algo=args.algo)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn
    

    # define dev (validation) dataloader

    d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),is_train=False,is_eval=False)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_ASVspoof2019_train_emb(args,list_IDs = file_dev,labels = d_label_dev,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_dev/'),algo=args.algo, is_train=False)

    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)

    del dev_set,d_label_dev

    return train_loader, dev_loader

def get_train_dev_dataloader(args, augment='rawboost', dataset='LA19'):

    # define train dataloader
    if dataset == 'LA19':
        logging.info("Using LA19 dataset")
        d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),is_train=True,is_eval=False)
        
        print('no. of training trials',len(file_train))
        
        if augment == 'rawboost':
            train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_train/'),algo=args.algo)
        elif augment == 'audiomentations':
            train_set=Dataset_ASVspoof2019_train_augment2(list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_train/'))
        
        train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=args.workers, shuffle=True,drop_last = True)
        
        del train_set,d_label_trn
        

        # define dev (validation) dataloader

        d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),is_train=False,is_eval=False)
        
        print('no. of validation trials',len(file_dev))
        
        dev_set = Dataset_ASVspoof2019_train(args,list_IDs = file_dev,labels = d_label_dev,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_dev/'),algo=args.algo)

        dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=args.workers, shuffle=False)

        del dev_set,d_label_dev

        return train_loader, dev_loader
    
    else:
        logging.info("Using CNSL dataset")
        # define train dataloader
        d_label_trn, file_train = genSpoof_list_v2(dir_meta = os.path.join(args.database_path, args.protocols_path), 
                                            is_train=True, is_dev=False, is_eval=False)
        
        print('no. of training trials',len(file_train))
        
        train_set = Dataset_cnsl(args, list_IDs = file_train, labels = d_label_trn, base_dir = args.database_path+'/', algo=args.algo)
        
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, drop_last = True)
        
        del train_set,d_label_trn
        

        # define validation dataloader
        d_label_dev, file_dev = genSpoof_list_v2(dir_meta = os.path.join(args.database_path, args.protocols_path), 
                                            is_train=False, is_dev=True, is_eval=False)
        
        print('no. of validation trials',len(file_dev))
        
        dev_set = Dataset_cnsl(args, list_IDs = file_dev, labels = d_label_dev, base_dir = args.database_path+'/', algo=args.algo)
        dev_loader = DataLoader(dev_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
        del dev_set, d_label_dev
        return train_loader, dev_loader
    