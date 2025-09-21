


<frontend>_<<optional_num_layer>>_<backend>

Model              | short notation
XLSR_Conformer     | X_Conf
XLSR_ConformerTCM  | X_Conf-TCM
XLSR_AASIST        | X_A
XLSR_VIB           | X_V
XLSR_Linear        | X_L
Wav2Vecbase_5_ConformerTCM | W2V_5_Conf-TCM


Ablation study     | short notation

emb = cosine + mse + recon (Shallow AE) | emb_c_m_r-sae
emb = cosine + mse + recon (Deep AE)    | emb_c_m_r-dae
