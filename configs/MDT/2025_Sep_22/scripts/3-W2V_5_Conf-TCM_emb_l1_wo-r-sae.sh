CUDA_VISIBLE_DEVICES=MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494 OMP_NUM_THREADS=5 python unified_main.py \
--yaml 'configs/MDT/2025_Sep_22/W2V_5_Conf-TCM_emb_l1_wo-r-sae.yaml' \
--database_path='data/KD25/' \
--protocols_path='new_protocol_Sep_17_2025_trim_vocoded_cleaned_v4_corrected_replay_cl_250610.txt'