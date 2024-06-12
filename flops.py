from teacher import *
from student import *
from wav2vec2_linear_nll_multi import Model as W2V2_LIN
import torch
from calflops import calculate_flops
from wav2vec2_vib import Model as W2V2_VIB

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Teacher
t_model = W2V2_VIB(
    device, '/datad/hungdx/KDW2V-AASISTL/pretrained/xlsr2_300m.pt')
batch_size = 1
input_shape = (batch_size, 64600)  # 4s audio
flops, macs, params = calculate_flops(model=t_model,
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
print("Teacher(W2V2_VIB): FLOPs:%s   MACs:%s   Params:%s \n" %
      (flops, macs, params))

# Students
student_models_dict = {
    "XLSR_6_VIB": Distil_XLSR_N_Trans_Layer_VIB(device, num_layers=6),
    "XLSR_5_VIB":  Distil_XLSR_N_Trans_Layer_VIB(device, num_layers=5),
    "XLSR_4_VIB":  Distil_XLSR_N_Trans_Layer_VIB(device, num_layers=4),
}


for k, v in student_models_dict.items():
    flops, macs, params = calculate_flops(model=v,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4)
    print(f"Student({k}): FLOPs:{flops}   MACs:{macs}   Params:{params} \n")


# device, xlsr_ssl_cpkt_path, ssl_cpkt_path = 'cuda', '/datad/hungdx/KDW2V-AASISTL/xlsr2_300m.pt', '/datad/hungdx/KDW2V-AASISTL/wav2vec_small.pt'
# # XLSR-AASIST
# m_t = W2V2_AASIST(device, xlsr_ssl_cpkt_path)
# batch_size = 1
# input_shape = (batch_size, 64600)

# flops, macs, params = calculate_flops(model=m_t,
#                                       input_shape=input_shape,
#                                       output_as_string=True,
#                                       output_precision=4)

# print("Teacher(XLSR-AASIST): FLOPs:%s   MACs:%s   Params:%s \n" %
#       (flops, macs, params))
# m_s = Distil_W2V2BASE_AASISTL(device, ssl_cpkt_path)
# flops, macs, params = calculate_flops(model=m_s,
#                                       input_shape=input_shape,
#                                       output_as_string=True,
#                                       output_precision=4)

# print("Student(W2V2BASE_AASIST-L): FLOPs:%s   MACs:%s   Params:%s \n" %
#       (flops, macs, params))
