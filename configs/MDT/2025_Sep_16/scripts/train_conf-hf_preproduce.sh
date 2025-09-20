# This version only uses KD loss w/o copy weights

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=5 python torchdistill_main_v7_alpha_test_hf.py --yaml 'configs/MDT/2025_Sep_16/xlsr_conformertcm_large_corpus_nov_conf-1-1-1_v2_hf.yaml' --database_path='data/replay_cl_250610/' --protocols_path='protocol.txt'