CUDA_VISIBLE_DEVICES=MIG-56c6e426-3d07-52cb-aa59-73892edacb69 OMP_NUM_THREADS=5 python unified_main.py \
--yaml 'configs/MDT/2025_Sep_23/W2V_5_Conf-TCM_ws_emb_c_m_stage2.yaml' \
--database_path='data/KD25/' \
--protocols_path='new_protocol_Sep_17_2025_trim_vocoded_cleaned_v4_corrected_replay_cl_250610.txt'