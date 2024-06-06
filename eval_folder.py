import subprocess
import os
import time
FOLDER_ROOT = '/datad/hungdx/KDW2V-AASISTL/byot_runs/Distil_XLSR_5_Custom_Trans_Layer_VIB_aug3_b16_randomstart_feb07_byot_from_best_checkpoint_freeze_feature_extractor'
YAML_PATH = '/datad/hungdx/KDW2V-AASISTL/self-kd-config/byot_5layers_freeze_feature_extractor.yml'
STUDENT_MODEL_TYPE = 'Self_Distil_XLSR_N_Trans_Layer_VIB'

for ckpt in os.listdir(FOLDER_ROOT):
    if ckpt.startswith('best_checkpoint'):
        command = f"""
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datad/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "{os.path.join(FOLDER_ROOT, ckpt)}" --eval_output="{os.path.join(FOLDER_ROOT, ckpt)}.txt" --batch_size_eval=200 --wrapper_ssl --database_path='/home/hungdx/Datasets/supcon_cnsl_feb07' --protocols_path='protocol.txt' --yaml='{YAML_PATH}' --dataset='feb07' --student_model_type='{STUDENT_MODEL_TYPE}'
"""
        # Run command in parallel
        # subprocess.Popen(command, shell=True)
        subprocess.run(command, shell=True)
        # print(f"Running {command}")
        # with open(os.devnull, 'w') as devnull:
        #     subprocess.Popen(command, shell=True, stdin=devnull,
        #                      stdout=devnull, stderr=devnull)

        # time.sleep(300)
