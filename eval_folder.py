import subprocess
import os
import time
FOLDER_ROOT = '/home/hungdx/code/KDW2V-AASISTL/runs/Distil_XLSR_5_Custom_Trans_Layer_Linear_noaudioaug_randomstart_large_corpus_jun_on_4s_kaist'
YAML_PATH = 'configs/kaist_proj_on_large_corpus_jun_4s.yaml'
STUDENT_MODEL_TYPE = 'Distil_XLSR_N_Trans_Layer_Linear'
database_path = '/datab/Dataset/cnsl_real_fake_audio/supcon_cnsl_jan22/'
for ckpt in os.listdir(FOLDER_ROOT):
    if ckpt.startswith('best_checkpoint'):
        get_num_checkpoint = ckpt.split("best_checkpoint_")[1].split(".")[0]
        
        file_to_save = f"kaist_project/Distil_XLSR_5_Custom_Trans_Layer_Linear_noaudioaug_randomstart_large_corpus_jun_on_4s_kaist_epoch{get_num_checkpoint}.txt"
        print(file_to_save)
        if os.path.exists(file_to_save):
            print(f"File {file_to_save} already exists. Skipping")
            continue

        ckpt_path = os.path.join(FOLDER_ROOT, ckpt)
        command = f"""
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=5 PYTHONPATH=$PYTHONPATH:/datab/hungdx/KDW2V-AASISTL/fairseq python eval.py --student_model_path "{ckpt_path}" --eval_output="{file_to_save}" --batch_size_eval=32 --wrapper_ssl --database_path='{database_path}' --protocols_path='protocol.txt' --yaml='{YAML_PATH}' --dataset='jan22' --student_model_type='{STUDENT_MODEL_TYPE}' --padding_size=64000
"""
        # Run command in parallel
        # subprocess.Popen(command, shell=True)
        
        subprocess.run(command, shell=True)
        # print(f"Running {command}")
        # with open(os.devnull, 'w') as devnull:
        #     subprocess.Popen(command, shell=True, stdin=devnull,
        #                      stdout=devnull, stderr=devnull)

        # time.sleep(300)
