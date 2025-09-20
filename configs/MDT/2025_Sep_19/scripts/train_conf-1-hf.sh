CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=5 python torchdistill_main_v3_hf_alpha_test.py \
 --yaml 'configs/MDT/2025_Sep_19/xlsr_conformertcm_large_corpus_nov_conf-1_hf_w_o_recon.yaml'\
  --database_path='data/replay_cl_250610/' \
  --protocols_path='new_protocol_Sep_17_2025_trim_vocoded_cleaned_v4_corrected_replay_cl_250610.txt'