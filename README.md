# Model and code for *A REAL-TIME AUDIO DEEPFAKE DETECTION ON LIMITED RESOURCES DEVICES*

# Clone the repository and submodules

```
git clone --recursive https://github.com/hungdinhxuan/A-REAL-TIME-AUDIO-DEEPFAKE-DETECTION-ON-LIMITED-RESOURCES-DEVICES
```

### Pre-trained models:

| Models | Link |
|--------|------|
|KDW2V-AASISTL|[google drive](https://drive.google.com/drive/folders/1-wm-Xt-Ek-GuYgWIcPtA8mHufKJAJD77?usp=sharing)| 


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
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/nfs/datab/hungdx/KDW2V-AASISTL/fairseq python torchdistill_main.py  --yaml ''
```

### Eval
```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONPATH=$PYTHONPATH:/nfs/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "/nfs/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_AASISTL_SelfKD_KDLoss_Without_teacher_lr_scheduler/best_checkpoint_57.pth" --eval_output="./W2V2BASE_AASISTL_SelfKD_KDLoss_Without_teacher_lr_scheduler_CosineAnnealingWarmRestarts_150k.txt" --batch_size_eval=300 --num_eval_samples=150000 --wrapper_ssl
```


# Fairseq Error:
1. ImportError: cannot import name 'metrics' from 'fairseq' (unknown location)
Please refer to [fairseq issues](https://github.com/facebookresearch/av_hubert/issues/70#issuecomment-1646736723) for more details.
In short you can fix it by adding the following line to your command:
```
PYTHONPATH=$PYTHONPATH:<your absolute path to fairseq submodule> python <your comand here>
```
PYTHONPATH=$PYTHONPATH:/nfs/datab/hungdx/KDW2V-AASISTL/fairseq