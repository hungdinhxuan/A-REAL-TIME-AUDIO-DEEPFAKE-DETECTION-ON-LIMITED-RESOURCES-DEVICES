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
    logger.info("Evaluating teacher model")
    teacher = W2V2_AASIST(
        device, ssl_cpkt_path="/nfs/datab/hungdx/KDW2V-AASISTL/xlsr2_300m.pt").to(device)

    teacher.load_state_dict(torch.load(
        args.student_model_path))
    logger.info("Loaded teacher model from {}".format(args.student_model_path))

    kd_method = 'NaN'
    print(kd_method)
    if args.dataset == 'in_the_wild':

        file_eval = genSpoof_in_the_wild_list(
            dir_meta=os.path.join(args.database_path, args.protocols_path))
        logger.info(f'no. of eval trials {len(file_eval)}')
        eval_set = Dataset_in_the_wild_eval(
            list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, teacher, device, args.eval_output,
                                batch_size=args.batch_size_eval, kd_method=kd_method, is_half=args.half)
        sys.exit(0)
    file_eval = genSpoof_list_v2(dir_meta=os.path.join(args.database_path, args.protocols_path),
                                 is_train=False, is_dev=False, is_eval=True)
    logger.info(f'no. of eval trials {len(file_eval)}')
    eval_set = Dataset_cnsl_eval(
        list_IDs=file_eval, base_dir=os.path.join(args.database_path))
    produce_evaluation_file(eval_set, teacher, device, args.eval_output,
                            batch_size=args.batch_size_eval, kd_method=kd_method, is_half=args.half)
    sys.exit(0)


if args.student_model_type == 'SelfDistil_W2V2BASE_AASISTL':
    logger.info("Using SelfDistil_W2V2BASE_AASISTL")
    student_model = SelfDistil_W2V2BASE_AASISTL(
        device, ssl_cpkt_path="/nfs/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt")
elif args.student_model_type == 'Distil_W2V2BASE_Linear':
    logger.info("Using Distil_W2V2BASE_Linear")
    student_model = Distil_W2V2BASE_Linear(
        device, ssl_cpkt_path="/nfs/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt")
else:
    logger.info("Using Distil_W2V2BASE_AASISTL")
    student_model = Distil_W2V2BASE_AASISTL(
        device, ssl_cpkt_path="/nfs/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt")

student_model = torch.nn.DataParallel(student_model).to(device)
student_model.load_state_dict(torch.load(
    args.student_model_path, map_location=device), strict=False)
logger.info("Loaded student model from {}".format(args.student_model_path))


logger.info("Wrapped ssl model to torchaudio")
student_model.module.ssl_model = W2V2_TA(import_fairseq_model(
    student_model.module.ssl_model.model)).to(device)

# Compile model
student_model = torch.compile(student_model)
torch.set_float32_matmul_precision('high')

if args.bf16:

    with torch.cpu.amp.autocast():
        student_model.eval()
        student_model = torch.jit.script(student_model.module)
        student_model = torch.jit.freeze(student_model)


if args.dataset == 'DF21':
    args.track = 'DF'
    prefix_2021 = 'ASVspoof2021.{}'.format(args.track)
    _, file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path+'ASVspoof_{}_cm_protocols/{}.cm.eval.trl.txt'.format(
        args.track, prefix_2021)), is_train=False, is_eval=True, num_eval_samples=args.num_eval_samples)
    logger.info(f'no. of eval trials {len(file_eval)}')
    eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=os.path.join(
        args.database_path+'ASVspoof2021_{}_eval/'.format(args.track)))
    kd_method = 'self_KD_Teacher' if args.student_model_type == 'SelfDistil_W2V2BASE_AASISTL' else 'NaN'
    print(kd_method)
    logger.info("Start eval")
    produce_evaluation_file(eval_set, student_model, device, args.eval_output,
                            batch_size=args.batch_size_eval, kd_method=kd_method, is_half=args.half)
    logger.info("Done eval")

elif args.dataset == 'in_the_wild':
    kd_method = 'self_KD_Teacher' if args.student_model_type == 'SelfDistil_W2V2BASE_AASISTL' else 'NaN'
    print(kd_method)
    file_eval = genSpoof_in_the_wild_list(
        dir_meta=os.path.join(args.database_path, args.protocols_path))
    logger.info(f'no. of eval trials {len(file_eval)}')
    eval_set = Dataset_in_the_wild_eval(
        list_IDs=file_eval, base_dir=os.path.join(args.database_path))
    produce_evaluation_file(eval_set, student_model, device, args.eval_output,
                            batch_size=args.batch_size_eval, kd_method=kd_method, is_half=args.half)

else:
    kd_method = 'self_KD_Teacher' if args.student_model_type == 'SelfDistil_W2V2BASE_AASISTL' else 'NaN'
    print(kd_method)
    file_eval = genSpoof_list_v2(dir_meta=os.path.join(args.database_path, args.protocols_path),
                                 is_train=False, is_dev=False, is_eval=True)
    logger.info(f'no. of eval trials {len(file_eval)}')
    eval_set = Dataset_cnsl_eval(
        list_IDs=file_eval, base_dir=os.path.join(args.database_path))
    produce_evaluation_file(eval_set, student_model, device, args.eval_output,
                            batch_size=args.batch_size_eval, kd_method=kd_method, is_half=args.half)
