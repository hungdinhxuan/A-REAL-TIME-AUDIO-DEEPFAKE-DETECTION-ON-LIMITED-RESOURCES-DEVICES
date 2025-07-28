import torch
from data_utils import *
from student import *
from teacher import *
from startup_config import set_random_seed
from menu import get_main_menu
import logging
import sys
from main import produce_evaluation_file, W2V2_TA
from torchaudio.models.wav2vec2.utils import import_fairseq_model
from wav2vec2_linear_nll_multi import BackEnd
from models import *
from aasist.AASIST import *
import yaml
import os
from torchdistill.models.registry import get_model
from wav2vec2_vib import Model as W2V2_VIB
from wav2vec2_linear_nll_multi import Model as W2V2_Linear
from tqdm import tqdm
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_main_menu()

set_random_seed(args.seed, args)
print("Current padding size is {}".format(args.padding_size))

student_model_name = ''


elif args.yaml != '':
    with open(args.yaml, 'r') as f:
        logger.info('Load configuration file {}'.format(args.yaml))
        config = yaml.safe_load(f)
        student_model_name = config['model']['student']['name']
        print(student_model_name)
        if student_model_name.startswith('Distil_XLSR_N_Trans_Layer'):
            student_model = get_model(
                student_model_name, device=device, **config['model']['student']['kwargs']).to(device)
        elif student_model_name == 'Self_Distil_XLSR_N_Trans_Layer_VIB':
            print("Using Self")
            student_model = get_model(
                student_model_name, device=device, **config['model']['student']['kwargs']).to(device)
        else:
            student_model = get_model(student_model_name, device=device, **config['model']['student']['kwargs']).to(device)
else:
    raise ValueError(f"Student model {args.student_model_type} not found")

student_model = torch.nn.DataParallel(student_model).to(device)
student_model.load_state_dict(torch.load(
    args.student_model_path, map_location=device))
logger.info("Loaded student model from {}".format(args.student_model_path))

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    


print("Number of parameters in student model: {}".format(
    sum(p.numel() for p in student_model.parameters())))


kd_method = 'KDADD' 
print(kd_method)
file_eval = genSpoof_list_v2(dir_meta=os.path.join(args.database_path, args.protocols_path),
                                is_train=False, is_dev=False, is_eval=True, special=False)
logger.info(f'no. of eval trials {len(file_eval)}')
eval_set = Dataset_cnsl_eval(
    list_IDs=file_eval, base_dir=os.path.join(args.database_path), padding_size=args.padding_size)
produce_evaluation_file(eval_set, student_model, device, args.eval_output,
                        batch_size=args.batch_size_eval, kd_method=kd_method, is_half=args.half)
logger.info("Done eval")
sys.exit(0)
