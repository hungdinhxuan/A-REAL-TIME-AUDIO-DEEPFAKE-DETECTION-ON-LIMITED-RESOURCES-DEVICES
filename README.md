# Model and code for KDW2V-AASISTL "*A REAL-TIME AUDIO DEEPFAKE DETECTION ON LIMITED RESOURCES DEVICES*

### Pre-trained models:

| Models | Link |
|--------|------|
|KDW2V-AASISTL|[google drive](https://drive.google.com/drive/folders/1-wm-Xt-Ek-GuYgWIcPtA8mHufKJAJD77?usp=sharing)| 


## Command to restore last step of training in KD cosine mode
```
CUDA_VISIBLE_DEVICES=0 python train.py --KD_cosine --comment "KD_cosine" --batch_size=30 --student_restore
```

## Command to restore last step of training in KD logits mode
```
CUDA_VISIBLE_DEVICES=0 python train.py --KD_logits --comment "KD_logits" --batch_size=30 --student_restore
```

## Command to restore last step of training in KD mse mode
```
CUDA_VISIBLE_DEVICES=0 python train.py --KD_mse --comment "KD_mse" --batch_size=30 --student_restore
```

## Evaluation
```
CUDA_VISIBLE_DEVICES=0 python train.py --comment "KD_cosine" --student_restore --model_path='' --eval
```