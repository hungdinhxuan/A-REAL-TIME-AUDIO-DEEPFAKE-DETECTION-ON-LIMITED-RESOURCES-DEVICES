
from torchaudio.models.wav2vec2.utils import import_fairseq_model
import fairseq
import torch
from torchinfo import summary
import os
import sys
from fvcore.nn import FlopCountAnalysis, flop_count_table
sys.path.append('..')


def show_info():
    from teacher import W2V2_AASIST
    from student import Distil_W2V2BASE_AASISTL

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ssl_cpkt_student_path = "/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt"
    ssl_cpkt_path = "/datab/hungdx/KDW2V-AASISTL/xlsr2_300m.pt"
    student = Distil_W2V2BASE_AASISTL(device, ssl_cpkt_student_path)

    summary(student, input_size=(1, 64600), device=device)

    teacher = W2V2_AASIST(device, ssl_cpkt_path)

    summary(teacher, input_size=(1, 64600), device=device)


def flops():
    from teacher import W2V2_AASIST
    from student import Distil_W2V2BASE_AASISTL
    from main import W2V2_TA
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ssl_cpkt_student_path = "/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt"
    ssl_cpkt_path = "/datab/hungdx/KDW2V-AASISTL/xlsr2_300m.pt"
    student = Distil_W2V2BASE_AASISTL(device, ssl_cpkt_student_path)
    student.ssl_model = W2V2_TA(import_fairseq_model(
        student.ssl_model.model
    )).to(device)
    input = torch.randn(1, 64600).to(device)
    student = student.to(device)

    print(FlopCountAnalysis(student, input).total())

    teacher = W2V2_AASIST(device, ssl_cpkt_path)

    teacher.ssl_model = W2V2_TA(import_fairseq_model(
        teacher.ssl_model.model
    )).to(device)
    teacher = teacher.to(device)
    print(FlopCountAnalysis(teacher, input).total())


show_info()
# flops()
