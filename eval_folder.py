import subprocess
import os
import time
FOLDER_ROOT = '/datad/hungdx/KDW2V-AASISTL/runs/Distil_XLSR_8_Last_Trans_Layer_VIB_DKDLoss_MSE_noaudioaug_b16_randomstart_feb07'

for ckpt in os.listdir(FOLDER_ROOT):
    if ckpt.startswith('best_checkpoint'):
        command = f"""
    CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datad/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "{os.path.join(FOLDER_ROOT, ckpt)}" --eval_output="{os.path.join(FOLDER_ROOT, ckpt)}.txt" --batch_size_eval=200 --wrapper_ssl --database_path='/home/hungdx/Datasets/supcon_cnsl_feb07' --protocols_path='protocol.txt' --yaml='/datad/hungdx/KDW2V-AASISTL/lst_configs/trial12.yaml' --dataset='feb07' --student_model_type='Distil_XLSR_N_Trans_Layer_VIB'
"""
        # Run command in parallel
        # subprocess.Popen(command, shell=True)
        subprocess.run(command, shell=True)
        # print(f"Running {command}")
        # with open(os.devnull, 'w') as devnull:
        #     subprocess.Popen(command, shell=True, stdin=devnull,
        #                      stdout=devnull, stderr=devnull)

        # time.sleep(300)
