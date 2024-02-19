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


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_main_menu()

set_random_seed(args.seed, args)



if args.is_eval_teacher:
    raise ValueError("This script is for evaluating student model only")
    

if args.student_model_type == 'SelfDistil_W2V2BASE_AASISTL':
    logger.info("Using SelfDistil_W2V2BASE_AASISTL")
    student_model = SelfDistil_W2V2BASE_AASISTL(device, ssl_cpkt_path="/nfs/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt")
else:
    logger.info("Using Distil_W2V2BASE_AASISTL")
    student_model = Distil_W2V2BASE_AASISTL(device, ssl_cpkt_path="/nfs/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt")

student_model = torch.nn.DataParallel(student_model).to(device)
student_model.load_state_dict(torch.load(args.student_model_path,map_location=device))
logger.info("Loaded student model from {}".format(args.student_model_path))


logger.info("Wrapped ssl model to torchaudio")
student_model.module.ssl_model = W2V2_TA(import_fairseq_model(student_model.module.ssl_model.model)).to(device)
    

if args.dataset == 'DF21':
    args.track = 'DF'
    prefix_2021 = 'ASVspoof2021.{}'.format(args.track)
    _,file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_{}_cm_protocols/{}.cm.eval.trl.txt'.format(args.track,prefix_2021)),is_train=False,is_eval=True, num_eval_samples=args.num_eval_samples)
    logger.info(f'no. of eval trials {len(file_eval)}')
    eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(args.track)))
    kd_method = 'self_KD_Teacher'
    logger.info("Start eval")
    produce_evaluation_file(eval_set, student_model, device, args.eval_output, batch_size=args.batch_size_eval, kd_method=kd_method, is_half=args.half)
    logger.info("Done eval")

else:
    kd_method = 'self_KD_Teacher' if args.student_model_type == 'SelfDistil_W2V2BASE_AASISTL' else 'NaN'
    file_eval = genSpoof_list_v2(dir_meta = os.path.join(args.database_path, args.protocols_path), 
                                            is_train=False, is_dev=False, is_eval=True)
    logger.info(f'no. of eval trials {len(file_eval)}')
    eval_set = Dataset_cnsl_eval(list_IDs = file_eval, base_dir = os.path.join(args.database_path))
    produce_evaluation_file(eval_set, student_model, device, args.eval_output, batch_size=args.batch_size_eval, kd_method=kd_method, is_half=args.half)
    