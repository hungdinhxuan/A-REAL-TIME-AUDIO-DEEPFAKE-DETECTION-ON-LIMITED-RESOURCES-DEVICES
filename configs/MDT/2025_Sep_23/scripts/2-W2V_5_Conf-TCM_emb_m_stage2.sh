CUDA_VISIBLE_DEVICES=MIG-57de94a5-be15-5b5a-b67e-e118352d8a59 OMP_NUM_THREADS=5 python unified_main.py \
--yaml 'configs/MDT/2025_Sep_23/W2V_5_Conf-TCM_emb_m_stage2.yaml' \
--database_path='data/KD25/' \
--protocols_path='new_protocol_Sep_17_2025_trim_vocoded_cleaned_v4_corrected_replay_cl_250610.txt'