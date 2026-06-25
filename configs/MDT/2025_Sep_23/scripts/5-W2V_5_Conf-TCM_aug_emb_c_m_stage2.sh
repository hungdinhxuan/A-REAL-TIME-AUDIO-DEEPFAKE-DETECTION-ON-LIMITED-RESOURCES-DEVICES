CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=5 python unified_main.py \
--yaml 'configs/MDT/2025_Sep_23/W2V_5_Conf-TCM_aug_emb_c_m_stage2.yaml' \
--database_path='data/replay_cl_250610/' \
--protocols_path='new_protocol_Sep_17_2025_trim_vocoded_cleaned_v4_corrected_replay_cl_250610.txt'