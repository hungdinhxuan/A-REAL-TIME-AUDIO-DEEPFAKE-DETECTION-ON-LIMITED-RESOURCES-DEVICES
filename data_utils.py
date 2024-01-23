import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

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
            X,fs = librosa.load(self.base_dir+utt_id+'.flac', sr=16000) 
            Y=process_Rawboost_feature(X,fs,self.args,self.algo)
            X_pad= pad(Y,self.cut)
            x_inp= Tensor(X_pad)
            target = self.labels[utt_id]
            
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
        X, fs = librosa.load(self.base_dir+key+'.flac', sr=16000)
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
        X, fs = librosa.load(self.base_dir+key+'.flac', sr=16000)
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
        X, fs = librosa.load(self.base_dir+keys[1]+'.flac', sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        
        return x_inp, self.list_IDs[index]        

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