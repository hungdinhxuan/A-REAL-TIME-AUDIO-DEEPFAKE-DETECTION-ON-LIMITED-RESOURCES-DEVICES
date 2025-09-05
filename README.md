# Model and code for _A REAL-TIME AUDIO DEEPFAKE DETECTION ON LIMITED RESOURCES DEVICES_

# Clone the repository and submodules

```
git clone --recursive https://github.com/hungdinhxuan/A-REAL-TIME-AUDIO-DEEPFAKE-DETECTION-ON-LIMITED-RESOURCES-DEVICES
```

### Pre-trained models:

| Models        | Link                                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------------------- |
| KDW2V-AASISTL | [google drive](https://drive.google.com/drive/folders/1-wm-Xt-Ek-GuYgWIcPtA8mHufKJAJD77?usp=sharing) |

## Command to restore last step of training in KD cosine mode

```
CUDA_VISIBLE_DEVICES=0 python main.py --KD_cosine --comment "KD_cosine" --batch_size=30 --student_restore
```

## Command to restore last step of training in KD logits mode

```
CUDA_VISIBLE_DEVICES=0 python main.py --KD_logits --comment "KD_logits" --batch_size=30 --student_restore
```

## Command to restore last step of training in KD mse mode

```
CUDA_VISIBLE_DEVICES=0 python main.py --KD_mse --comment "KD_mse" --batch_size=30 --student_restore
```

## Evaluation

### Evaluation with teacher

```
CUDA_VISIBLE_DEVICES=0 python main.py --is_eval_teacher --batch_size_eval=32 --num_eval_samples=150000 --eval_output "./teacher_test_150k.txt" --eval --KD_logits

```

### Evaluation with KD mse mode

```
CUDA_VISIBLE_DEVICES=0 python main.py --comment "KD_mse" --student_restore --model_path='' --eval --KD_mse --batch_size=30 --eval_output "./KD_mse_test_150k.txt" --batch_size_eval=100 --num_eval_samples=150000
```

### Evaluation with KD cosine mode

```
CUDA_VISIBLE_DEVICES=0 python main.py --comment "KD_cosine" --student_restore --model_path='' --eval --KD_cosine --batch_size=30 --eval_output "./KD_cosine_test_150k.txt" --batch_size_eval=64 --num_eval_samples=150000
```

### Evaluation with KD logits mode

```
CUDA_VISIBLE_DEVICES=2 python main.py --comment "KD_logits" --student_restore --model_path='' --eval --KD_logits --batch_size=30 --eval_output "./KD_logits_test_150k.txt" --batch_size_eval=64 --num_eval_samples=150000
```

### Evaluation with Self KD + Teacher Cosine (16bit half precision)

```
CUDA_VISIBLE_DEVICES=1 python main.py --student_ckpt="/datab/hungdx/KDW2V-AASISTL/models/model_DF_weighted_CCE_100_40_1e-06_self_KD_teacher_W2VBaseHG/best_checkpoint_42.pth" --self_KD_type="self_KD_Teacher_HG" --eval --half --eval_output="./self_KD_teacher_cosine_half_test_150k.txt" --self_KD --batch_size_eval=200 --num_eval_samples=150000
```

## Lastest command to train and evaluate

### Train

```
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python torchdistill_main.py --database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22' --protocols_path='protocol.txt' --batch_size=64 --yaml ''
```

### FT teacher

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python torchdistill_main.py --database_path='/datab/hungdx/KDW2V-AASISTL/databases/' --protocols_path='/datab/hungdx/KDW2V-AASISTL/protocols/' --yaml '/datab/hungdx/KDW2V-AASISTL/ft_teacher_config/trial5.yaml'
```

```
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python torchdistill_main.py --database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22' --protocols_path='protocol.txt' --yaml '/datab/hungdx/KDW2V-AASISTL/ft_teacher_config/trial3.yaml'
```

### Eval

```
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datad/hungdx/KDW2V-AASISTL/runs/Distil_XLSR_5_Custom_Trans_Layer_VIB_bestconf46_noaudioaug_b16_randomstart_large_corpus/best_checkpoint_0.pth" --eval_output="./Distil_XLSR_5_Custom_Trans_Layer_VIB_bestconf46_noaudioaug_b16_randomstart_large_corpus_best0_large_copus_intern_4s.txt" --batch_size_eval=200 --student_model_type='Distil_XLSR_N_Trans_Layer_Linear' --yaml='/datad/hungdx/KDW2V-AASISTL/distill-config/trial128.yaml' --dataset='feb07' --database_path='/home/hungdx/Datasets/supcon_cnsl_feb07' --protocols_path='protocol.txt'
```

### CNSL dataset

```
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datad/hungdx/KDW2V-AASISTL/fairseq python torchdistill_main.py --yaml 'configs/ICONIP2024/AASIST_best_conf_on_LA19_1s_updated_ce_weight_frombestr3_w_o_mse.yaml' --database_path='/datab/Dataset/cnsl_real_fake_audio/0_large-corpus' --protocols_path='protocol.txt'
```

### LA19 1s

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datad/hungdx/KDW2V-AASISTL/fairseq python torchdistill_main.py --yaml '/home/hungdx/code/KDW2V-AASISTL/kd/train/6Layer/XLSR-6-AASIST_LA19-1s_conf-2-2.yaml'

###

### MoreKor

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datad/hungdx/KDW2V-AASISTL/fairseq python torchdistill_main.py  --yaml '/datad/hungdx/KDW2V-AASISTL/distill-config/trial122.yaml' --database_path='/datad/Datasets/moreko/wavs' --protocols_path='../protocol.txt'
```

### VoxCeleb2 dataset

```
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python torchdistill_main.py  --yaml '/datab/hungdx/KDW2V-AASISTL/distill-config/trial54.yaml' --database_path='/datab/hungdx/KDW2V-AASISTL/data' --protocols_path='/datab/hungdx/KDW2V-AASISTL/data/protocol_file.txt'
```

### Eval CNSL

#### PKTLoss

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_AASISTL_PKTLoss_cnsl/best_checkpoint_42.pth" --eval_output="./W2V2BASE_AASISTL_PKTLoss_cnsl_best42.txt" --batch_size_eval=300 --wrapper_ssl --dataset='cnsl' --database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22' --protocols_path='protocol.txt'
```

#### KD Loss

```
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_AASISTL_KDLoss_cnsl_audiomentations/best_checkpoint_78.pth" --eval_output="./W2V2BASE_AASISTL_KDLoss_cnsl_audiomentations_best78.txt" --batch_size_eval=300 --wrapper_ssl --dataset='cnsl' --database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22' --protocols_path='protocol.txt'
```

#### DIST Loss

```
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_AASISTL_DISTLoss_cnsl_audiomentations/best_checkpoint_32.pth" --eval_output="./W2V2BASE_AASISTL_DISTLoss_cnsl_audiomentations_best32.txt" --batch_size_eval=300 --wrapper_ssl --dataset='cnsl' --database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22' --protocols_path='protocol.txt'
```

#### DKD Loss

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_5_v4/best_checkpoint_32.pth" --eval_output="./W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_5_v4_best32_jan22.txt" --batch_size_eval=300 --wrapper_ssl --dataset='cnsl' --database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22' --protocols_path='protocol.txt'
```

#### Self KD

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_5/best_checkpoint_11.pth" --eval_output="./W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_5_best11_jan22_1s.txt" --batch_size_eval=600 --wrapper_ssl --dataset='cnsl' --database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22' --protocols_path='protocol.txt' --student_model_type='sss'
```

### Eval VoxCeleb2

### VoxCeleb2 dataset

```
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_3_v10/best_checkpoint_63.pth" --database_path='/datab/hungdx/KDW2V-AASISTL/data' --protocols_path='/datab/hungdx/KDW2V-AASISTL/data/protocol_file.txt' --student_model_type='self_KD' --batch_size_eval=100 --eval_output="./W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_3_v10_best63_voxcleb2.txt"
```

## Calculate EER

```
python3 score_file_to_eer.py ./results/Distil_XLSR_5_Custom_Trans_Layer_VIB_noaudioaug_randomstart_large_corpus_on_2s_best124_intern_2s.txt '/datab/Dataset/intern_2024_jan31/protocol.txt' 'all'
```

```
python score_file_to_eer.py /datad/hungdx/KDW2V-AASISTL/models/Distil_XLSR_N_Trans_Layer_Linear_DKDLoss_noaudioaug_b16_randomstart_MultiStepLR_feb07/best_checkpoint_27.pth.txt '/home/hungdx/Datasets/supcon_cnsl_feb07/protocol.txt' 'eval'
```

```
python score_file_to_eer.py /datad/hungdx/KDW2V-AASISTL/results/W2V2BASE_Linear_DKDLoss_noaudioaug_b16_randomstart_MultiStepLR_feb07_best_checkpoint_41_wrapper2.5s.txt   '/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_feb07/protocol.txt' 'eval'
```

# Fairseq Error:

1. ImportError: cannot import name 'metrics' from 'fairseq' (unknown location)
   Please refer to [fairseq issues](https://github.com/facebookresearch/av_hubert/issues/70#issuecomment-1646736723) for more details.
   In short you can fix it by adding the following line to your command:

```
PYTHONPATH=$PYTHONPATH:<your absolute path to fairseq submodule> python <your comand here>
```

PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python torchdistill_main.py --yaml '/datad/hungdx/KDW2V-AASISTL/lst_configs_custom/tmp.yaml' --database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22' --protocols_path='protocol.txt'

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python selfdistill_main.py --yaml '/datab/hungdx/KDW2V-AASISTL/self-kd-config/trial5.yaml' --database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22' --protocols_path='protocol.txt'

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python torchdistill_main.py --yaml '/datab/hungdx/KDW2V-AASISTL/distill-config/trial61.yaml' --database_path='/datab/hungdx/KDW2V-AASISTL/data' --protocols_path='protocol_file_2.txt'

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python torchdistill_main.py --yaml '/datab/hungdx/KDW2V-AASISTL/distill-config/trial102.yaml' --database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_feb07' --protocols_path='protocol.txt'

#### Train with LA19

```
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python torchdistill_main.py  --yaml '/datad/hungdx/KDW2V-AASISTL/configs/dev2.yaml' --database_path='/home/hungdx/Datasets/' --protocols_path='/datad/hungdx/KDW2V-AASISTL/protocols/'
```

### Eval Feb07

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_Linear_DKDLoss_cnsl_noaudiomentations_v2/best_checkpoint_30.pth" --eval_output="./W2V2BASE_Linear_DKDLoss_cnsl_noaudiomentations_v2_best30_feb07.txt" --batch_size_eval=300 --wrapper_ssl --dataset='cnsl' --database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_feb07' --protocols_path='protocol.txt' --student_model_type='Distil_W2V2BASE_Linear'

### Eval in the wild

```
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_5/best_checkpoint_11.pth" --eval_output="./W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_5_best11_in_the_wild.txt" --batch_size_eval=300 --wrapper_ssl --dataset='in_the_wild' --database_path='/datab/Dataset/cnsl_real_fake_audio' --protocols_path='in_the_wild.txt'
```

```
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=5 python eval.py --student_model_path "/datad/pretrained/best_checkpoint_124.pth" --eval_output="./results/Distil_XLSR_5_Custom_Trans_Layer_VIB_noaudioaug_randomstart_large_corpus_on_2s_best124_eval_itw.txt" --batch_size_eval=400 --wrapper_ssl --database_path='/datab/Dataset/cnsl_real_fake_audio' --protocols_path='in_the_wild.txt' --yaml='configs/best_conf_on_large_corpus_2s.yaml' --dataset='in_the_wild' --padding_size=32000
```

### Eval Feb07 from VIB MoreKor

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datad/hungdx/KDW2V-AASISTL/models/W2V2BASE_VIB_DKDLoss_noaudioaug_b16_randomstart_CosineAnnealingWarmRestarts_acmccs_apr3/best_checkpoint_31.pth" --eval_output="./W2V2BASE_VIB_DKDLoss_noaudioaug_b16_randomstart_CosineAnnealingWarmRestarts_acmccs_apr3_best31_feb07.txt" --batch_size_eval=400 --wrapper_ssl --database_path='/home/hungdx/Datasets/supcon_cnsl_feb07' --protocols_path='protocol.txt' --yaml='/datad/hungdx/KDW2V-AASISTL/distill-config/trial127.yaml' --dataset='feb07' --student_model_type='Distil_W2V2BASE_VIB'
```

### Eval Feb07 Distil_XLSR_N_Trans_Layer_VIB

```
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path /home/hungdx/code/KDW2V-AASISTL/runs/Distil_XLSR_5_Custom_Trans_Layer_VIB_noaudioaug_randomstart_large_corpus_on_2s/best_checkpoint_124.pth --eval_output="./results/Distil_XLSR_5_Custom_Trans_Layer_VIB_noaudioaug_randomstart_large_corpus_on_2s_best124_intern_1s.txt" --batch_size_eval=400 --wrapper_ssl --database_path='/datab/Dataset/intern_2024_jan31' --protocols_path='protocol.txt' --yaml='/home/hungdx/code/KDW2V-AASISTL/configs/best_conf_on_large_corpus_2s.yaml' --dataset='feb07' --student_model_type='Distil_XLSR_N_Trans_Layer_VIB' --padding_size=16000
```

### Eval DF21 Distil_XLSR_N_Trans_Layer_VIB

```
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datad/hungdx/KDW2V-AASISTL/runs/Distil_XLSR_5_Custom_Trans_Layer_VIB_DKDLoss_MSE_noaudioaug_b16_randomstart_feb07_1_2_3_4_5th_new_hyp_copy_weights_tuning_MSE/best_checkpoint_19.pth" --eval_output="./Distil_XLSR_5_Custom_Trans_Layer_VIB_DKDLoss_MSE_noaudioaug_b16_randomstart_feb07_1_2_3_4_5th_new_hyp_copy_weights_tuning_MSE_4s_torchaudio.txt" --batch_size_eval=100 --wrapper_ssl --database_path='/home/hungdx/Datasets/supcon_cnsl_feb07' --protocols_path='protocol.txt' --yaml='/datad/hungdx/KDW2V-AASISTL/best_configs/trial46.yaml' --dataset='feb07' --student_model_type='Distil_XLSR_N_Trans_Layer_VIB' --padding_size=64600
```

### Eval MOreKor from VIB MoreKor

```
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datad/hungdx/KDW2V-AASISTL/models/W2V2BASE_VIB_DKDLoss_noaudioaug_b16_randomstart_CosineAnnealingWarmRestarts_acmccs_apr3_feb07_from_W2V2BASE_VIB_DKDLoss_noaudioaug_b16_randomstart_CosineAnnealingWarmRestarts_acmccs_apr3_moreko/best_checkpoint_2.pth" --eval_output="./W2V2BASE_VIB_DKDLoss_noaudioaug_b16_randomstart_CosineAnnealingWarmRestarts_acmccs_apr3_feb07_from_W2V2BASE_VIB_DKDLoss_noaudioaug_b16_randomstart_CosineAnnealingWarmRestarts_acmccs_apr3_moreko_best2_eval_moreko.txt" --batch_size_eval=400 --wrapper_ssl --database_path='/datad/Datasets/moreko/wavs' --protocols_path='../protocol.txt' --yaml='/datad/hungdx/KDW2V-AASISTL/distill-config/trial125.yaml' --dataset='moreko' --student_model_type='Distil_W2V2BASE_VIB'
```

## Train with large corpus

```
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/home/hungdx/code/fairseq python torchdistill_main.py --yaml 'configs/best_conf_on_large_corpus_2s-v1.yaml' --padding_size=32000 --database_path='/datab/Dataset/cnsl_real_fake_audio/0_large-corpus/' --protocols_path='protocol.txt'
```

## Eval with large corpus

```
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datad/pretrained/best_checkpoint_124.pth" --eval_output="./results/Distil_XLSR_5_Custom_Trans_Layer_VIB_noaudioaug_randomstart_large_corpus_on_2s_best124_eval_largecopus_detail.txt" --batch_size_eval=400 --wrapper_ssl --database_path='/datab/Dataset/cnsl_real_fake_audio/0_large-corpus' --protocols_path='protocol.txt' --yaml='configs/best_conf_on_large_corpus_2s.yaml' --dataset='largecorpus' --student_model_type='Distil_W2V2BASE_VIB' --padding_size=32000
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datad/pretrained/best_checkpoint_124.pth" --eval_output="./results/Distil_XLSR_5_Custom_Trans_Layer_VIB_noaudioaug_randomstart_large_corpus_on_2s_best124_eval_largecopus.txt" --batch_size_eval=400 --wrapper_ssl --database_path='/datab/Dataset/cnsl_real_fake_audio/0_large-corpus' --protocols_path='protocol.txt' --yaml='configs/best_conf_on_large_corpus_2s.yaml' --dataset='largecorpus' --student_model_type='Distil_W2V2BASE_VIB' --padding_size=32000
```

# Export models

```
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python export.py --student_model_path="/home/hungdx/code/KDW2V-AASISTL/runs/Distil_XLSR_5_Custom_Trans_Layer_VIB_noaudioaug_randomstart_large_corpus_on_2s/best_checkpoint_124.pth" --comment="wrapper_2s" --padding=2 --yaml="/datad/hungdx/KDW2V-AASISTL/best_configs/trial46.yaml"
```

```
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --is_eval_teacher --database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_feb07' --protocols_path='protocol.txt' --eval_output="./supcon_jan22_longR3_epoch_43_feb07.txt" --student_model_path="/datab/hungdx/KDW2V-AASISTL/supcon_jan22_longR3_epoch_43.pth" --batch_size_eval=64
```

```
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --is_eval_teacher --dataset='in_the_wild' --database_path='/datab/Dataset/cnsl_real_fake_audio' --protocols_path='in_the_wild.txt'  --eval_output="./epoch_3_from2.65_EER2.30_in_the_wild.txt" --student_model_path="/datab/hungdx/KDW2V-AASISTL/epoch_3_from2.65_EER2.30.pth" --batch_size_eval=64
```

### Eval again teacher

```
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datad/pretrained/vib_conf-5_gelu_2s_may27_epoch6.pth" --eval_output="./results/vib_conf-5_gelu_2s_may27_epoch6_eval_2s_detail.txt" --batch_size_eval=400 --database_path='/datab/Dataset/cnsl_real_fake_audio/0_large-corpus/' --protocols_path='protocol.txt' --dataset='largecorpus' --student_model_type='Distil_W2V2BASE_VIB' --is_eval_teacher --padding_size=32000
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datad/pretrained/vib_conf-5_gelu_2s_may27_epoch6.pth" --eval_output="./results/vib_conf-5_gelu_2s_may27_epoch6_eval_2s.txt" --batch_size_eval=400 --database_path='/datab/Dataset/cnsl_real_fake_audio/0_large-corpus/' --protocols_path='protocol.txt' --dataset='largecorpus' --student_model_type='Distil_W2V2BASE_VIB' --is_eval_teacher --padding_size=32000
```

### Train on large corpus (Jun) for KAIST project (4s)

configs/kaist_proj_on_large_corpus_jun_4s.yaml

```
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/home/hungdx/code/fairseq python torchdistill_main.py --yaml 'configs/kaist_proj_on_large_corpus_jun_4s.yaml' --padding_size=64000 --database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22/' --protocols_path='protocol.txt'
```

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/home/hungdx/code/fairseq python torchdistill_main.py --yaml 'configs/kaist_proj_on_large_corpus_jun_4s_raw5.yaml' --padding_size=64000 --database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22/' --protocols_path='protocol.txt'
```

#### eval on jan22

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/home/hungdx/code/KDW2V-AASISTL/runs/Distil_XLSR_5_Custom_Trans_Layer_Linear_noaudioaug_randomstart_large_corpus_jun_on_4s_kaist/best_checkpoint_105.pth" --eval_output="kaist_project/Distil_XLSR_5_Custom_Trans_Layer_Linear_noaudioaug_randomstart_large_corpus_jun_on_4s_kaist_epoch105_quant.txt" --batch_size_eval=32 --wrapper_ssl --database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22/' --protocols_path='protocol.txt' --yaml='configs/kaist_proj_on_large_corpus_jun_4s.yaml' --dataset='jan22' --student_model_type='Distil_XLSR_N_Trans_Layer_Linear' --padding_size=64000
```

#### eval on intern

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/home/hungdx/code/KDW2V-AASISTL/runs/Distil_XLSR_5_Custom_Trans_Layer_Linear_noaudioaug_randomstart_large_corpus_jun_on_4s_kaist/best_checkpoint_105.pth"  --eval_output="kaist_project/Distil_XLSR_5_Custom_Trans_Layer_Linear_noaudioaug_randomstart_large_corpus_jun_on_4s_kaist_epoch105_intern.txt" --batch_size_eval=32 --wrapper_ssl --database_path='/datab/Dataset/intern_2024_jan31' --protocols_path='protocol.txt' --yaml='configs/kaist_proj_on_large_corpus_jun_4s.yaml' --dataset='intern' --student_model_type='Distil_XLSR_N_Trans_Layer_Linear' --padding_size=64000
```

#### Eval again teacher

```
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datad/pretrained/AudioDeepfakeCMs/vocosig/conf-5-linear-nov22-newfinetune-r-epoch21.pth" --eval_output="./results/conf-5-linear-nov22-newfinetune-r-epoch21_eval_4s_intern.txt" --batch_size_eval=32 --database_path='/datab/Dataset/intern_2024_jan31/' --protocols_path='protocol.txt' --dataset='intern' --is_eval_teacher --padding_size=64000
```

```
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/datad/pretrained/AudioDeepfakeCMs/vocosig/conf-5-linear-nov22-newfinetune-r-epoch21.pth" --eval_output="./results/conf-5-linear-nov22-newfinetune-r-epoch21_eval_4s_jan22.txt" --batch_size_eval=32 database_path='/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22/' --protocols_path='protocol.txt' --dataset='jan22' --is_eval_teacher --padding_size=64000
```

### 4s

CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python export.py --student_model_path="/home/hungdx/code/KDW2V-AASISTL/runs/Distil_XLSR_5_Custom_Trans_Layer_Linear_noaudioaug_randomstart_large_corpus_jun_on_4s_kaist/best_checkpoint_105.pth" --comment="wrapper_4s" --padding=4 --yaml="/home/hungdx/code/KDW2V-AASISTL/configs/kaist_proj_on_large_corpus_jun_4s.yaml"



# Jul 2025

###2s - MDT-LoRA
CUDA_VISIBLE_DEVICES="" python export_v2.py --student_model_path="/nvme1/hungdx/KDW2V-AASISTL/runs/Distil_XLSR_5_Custom_Trans_Layer_ConformerTCM_conf-1-2/best.pth" --comment="mdt_lora_mini_wrapper_2s" --padding=2 --yaml="/nvme1/hungdx/KDW2V-AASISTL/configs/MDT/2025_July_25/xlsr_conformertcm_large_corpus_nov_conf-1-2.yaml"
###no-pad - MDT-LoRA
CUDA_VISIBLE_DEVICES="" python export_v2.py --student_model_path="/nvme1/hungdx/KDW2V-AASISTL/runs/Distil_XLSR_5_Custom_Trans_Layer_ConformerTCM_conf-1-2/best.pth" --comment="mdt_lora_mini_wrapper_no_pad" --padding=-1 --yaml="/nvme1/hungdx/KDW2V-AASISTL/configs/MDT/2025_July_25/xlsr_conformertcm_large_corpus_nov_conf-1-2.yaml"