CUDA_VISIBLE_DEVICES=MIG-ad433dcf-e7b9-5a99-a0fa-6fdf3033b7cd OMP_NUM_THREADS=5 python unified_main.py \
--yaml 'configs/MDT/2025_Sep_24/W2V_5_Conf-TCM_wo_norm_ws_emb_c_m_stage2_op.yaml' \
--database_path='data/KD25/' \
--protocols_path='new_protocol_Sep_17_2025_trim_vocoded_cleaned_v4_corrected_replay_cl_250610.txt'