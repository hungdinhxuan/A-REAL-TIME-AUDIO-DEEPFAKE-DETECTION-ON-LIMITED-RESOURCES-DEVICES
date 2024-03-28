from teacher import *
from student import *
from wav2vec2_linear_nll_multi import Model as W2V2_LIN
import torch
from calflops import calculate_flops

device, xlsr_ssl_cpkt_path, ssl_cpkt_path = 'cuda', '/datad/hungdx/KDW2V-AASISTL/xlsr2_300m.pt', '/datad/hungdx/KDW2V-AASISTL/wav2vec_small.pt'
# XLSR-AASIST
m_t = W2V2_AASIST(device, xlsr_ssl_cpkt_path)
batch_size = 1
input_shape = (batch_size, 64600)

flops, macs, params = calculate_flops(model=m_t,
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)

print("Teacher(XLSR-AASIST): FLOPs:%s   MACs:%s   Params:%s \n" %
      (flops, macs, params))
m_s = Distil_W2V2BASE_AASISTL(device, ssl_cpkt_path)
flops, macs, params = calculate_flops(model=m_s,
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)

print("Student(W2V2BASE_AASIST-L): FLOPs:%s   MACs:%s   Params:%s \n" %
      (flops, macs, params))


# W2V2-BASE-Lin
m_t = W2V2_LIN(device, xlsr_ssl_cpkt_path)
flops, macs, params = calculate_flops(model=m_t,
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
print("Teacher(XLSR-Linear):FLOPs:%s   MACs:%s   Params:%s \n" %
      (flops, macs, params))
m_s = Distil_W2V2BASE_Linear(device, ssl_cpkt_path)
flops, macs, params = calculate_flops(model=m_s,
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)

print("Student(W2V2BASE_Linear):FLOPs:%s   MACs:%s   Params:%s \n" %
      (flops, macs, params))