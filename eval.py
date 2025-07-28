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

def update_ssl_model_weights(model, model_list: nn.ModuleList):

    sd = model.state_dict()

    avg_sd = {}

    for i, m in enumerate(model_list):
        m_sd = m.state_dict()
        avg_sd['ssl_model.model'] += m_sd['ssl_model.model']

    avg_sd['ssl_model.model'] = avg_sd['ssl_model.model'] / len(model_list)

    sd['ssl_model.model'] = avg_sd['ssl_model.model']

    model.load_state_dict(sd)

    return model


def produce_evaluation_file_for_self_KD(dataset, model, device, save_path, kd_method=None, batch_size=4, is_half=False):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                             pin_memory=True if device != "cpu" else False, pin_memory_device=device)
    model.eval()
    fname_list = []
    score_list = []

    with torch.no_grad():
        for batch_x, utt_id in tqdm(data_loader):

            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)

            logits, features = model(batch_x)

            for index, logit in enumerate(logits):
                fname_list = []
                score_list = []

                logit = (logit[:, 1]
                         ).data.cpu().numpy().ravel()
            # add outputs
                fname_list.extend(utt_id)
                score_list.extend(logit.tolist())

                new_save_path = save_path.replace(
                    '.txt', '_{}.txt'.format(f'classifer_{index}'))

                with open(new_save_path, 'a+') as fh:

                    for f, cm in zip(fname_list, score_list):
                        fh.write('{} {}\n'.format(f, cm))
                fh.close()
    print('Scores saved to {}'.format(save_path))


if args.is_eval_teacher:
    logger.info("Evaluating teacher model")
    teacher = W2V2_Linear(
        device, ssl_cpkt_path="/datad/hungdx/KDW2V-AASISTL/pretrained/xlsr2_300m.pt").to(device)
    
    teacher = torch.nn.DataParallel(teacher).to(device)

    teacher.load_state_dict(torch.load(
        args.student_model_path))

    #  Keep 5 transformer layers
    # teacher.ssl_model.model.encoder.layers = teacher.ssl_model.model.encoder.layers[:5]

    # logger.info("Loaded teacher model from {}".format(args.student_model_path))

    # print("Number of parameters in teacher model: {}".format(
    #     sum(p.numel() for p in teacher.parameters())))

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
    else:
        special_datasets = ['moreko', 'largecorpus']
    
        print(f'Eval {args.dataset}')
        
        file_eval = genSpoof_list_v2(dir_meta=os.path.join(args.database_path, args.protocols_path),
                                    is_train=False, is_dev=False, is_eval=True, special=True if args.dataset in special_datasets else False)
        logger.info(f'no. of eval trials {len(file_eval)}')
        eval_set = Dataset_cnsl_eval(
            list_IDs=file_eval, base_dir=os.path.join(args.database_path), padding_size=args.padding_size)
        produce_evaluation_file(eval_set, teacher, device, args.eval_output,
                                batch_size=args.batch_size_eval, kd_method=kd_method, is_half=args.half)
    print("Done eval teachet")
    sys.exit(0)


student_model_name = ''
if args.student_model_type in globals() and args.yaml == '':
    logger.info(f"Using {args.student_model_type}")

    student_model = globals()[args.student_model_type](
        device, ssl_cpkt_path="/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt")


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
            student_model = get_model(
                student_model_name, d_args=config['model']['student']['kwargs']).to(device)

elif args.student_model_type == 'Distil_W2V2BASE_ConvNeXt_COAASISTL':
    logger.info("Using Distil_W2V2BASE_ConvNeXt_COAASISTL")
    student_model = Distil_W2V2BASE_ConvNeXt_COAASISTL(
        device, ssl_cpkt_path="/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt")

else:
    logger.info("Using Distil_W2V2BASE_AASISTL")

    student_model = Distil_W2V2BASE_AASISTL(
        device, ssl_cpkt_path="/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt")

student_model = torch.nn.DataParallel(student_model).to(device)
student_model.load_state_dict(torch.load(
    args.student_model_path, map_location=device))
logger.info("Loaded student model from {}".format(args.student_model_path))

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    
if not args.student_model_type == 'AASIST':
    logger.info("Wrapped ssl model to torchaudio")
    student_model.module.ssl_model = W2V2_TA(import_fairseq_model(
        student_model.module.ssl_model.model)).to(device)

    
    # Try to quantize the model
    print("Quantizing model")
    student_model = student_model.module
    # Count parameters before quantization
    # num_params_before = sum(p.numel() for p in student_model.parameters())
    # print("Number of parameters before quantization: {}".format(num_params_before))
    student_model = torch.quantization.quantize_dynamic(
        student_model, {torch.nn.Linear}, dtype=torch.qint8)

    # Count parameters after quantization
    # num_params_after = sum(p.numel() for p in student_model.parameters())
    # print("Number of parameters after quantization: {}".format(num_params_after))
    # import sys
    # print("Quantization done")
    # sys.exit(0)

print("Number of parameters in student model: {}".format(
    sum(p.numel() for p in student_model.parameters())))


# Compile model
# student_model = torch.compile(student_model)
# torch.set_float32_matmul_precision('high')

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
elif args.dataset == 'standard':
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
else:
    kd_method = 'self_KD_Teacher' if args.student_model_type == 'SelfDistil_W2V2BASE_AASISTL' else 'NaN'
    print(kd_method)

    special_datasets = ['moreko', 'largecorpus']
    
    print(f'Eval {args.dataset}')
    
    file_eval = genSpoof_list_v2(dir_meta=os.path.join(args.database_path, args.protocols_path),
                                 is_train=False, is_dev=False, is_eval=True, special=True if args.dataset in special_datasets else False)
    logger.info(f'no. of eval trials {len(file_eval)}')

    eval_set = Dataset_cnsl_eval(
        list_IDs=file_eval, base_dir=os.path.join(args.database_path), padding_size=args.padding_size)

    if student_model_name == 'Self_Distil_XLSR_N_Trans_Layer_VIB':
        print("Self KD evaluation")
        produce_evaluation_file_for_self_KD(eval_set, student_model, device, args.eval_output,
                                            batch_size=args.batch_size_eval, kd_method=kd_method, is_half=args.half)
    else:
        produce_evaluation_file(eval_set, student_model, device, args.eval_output,
                                batch_size=args.batch_size_eval, kd_method=kd_method, is_half=args.half)
