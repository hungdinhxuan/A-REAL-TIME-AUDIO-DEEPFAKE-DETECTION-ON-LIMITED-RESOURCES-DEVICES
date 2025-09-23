CUDA_VISIBLE_DEVICES=MIG-8cdeef83-092c-5a8d-a748-452f299e1df0 OMP_NUM_THREADS=5 python unified_main.py \
--yaml 'configs/MDT/2025_Sep_22/W2V_5_Conf-TCM_emb_c_m_wo-r-sae.yaml' \
--database_path='data/KD25/' \
--protocols_path='new_protocol_Sep_17_2025_trim_vocoded_cleaned_v4_corrected_replay_cl_250610.txt'