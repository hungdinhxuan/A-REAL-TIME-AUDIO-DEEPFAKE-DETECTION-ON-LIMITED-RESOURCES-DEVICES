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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_main_menu()

set_random_seed(args.seed, args)


class W2VBASE_Fusion_AASISTL_Linear(nn.Module):
    def __init__(self, device, ssl_cpkt_path):
        super().__init__()
        # Average weights
        filts = [128, [1, 32], [32, 32], [32, 24], [24, 24]]
        gat_dims = [24, 32]
        pool_ratios = [0.4, 0.5, 0.7, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(device, ssl_cpkt_path, 768).to(device)

        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=24)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(24, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 24, kernel_size=(1, 1)),

        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

        # Linear Backend
        self.backend = BackEnd(128, 128, 2, 0.5, False)

    def forward(self, x):
        x_ssl_feat = self.ssl_model(x.squeeze(-1))
        x = self.LL(x_ssl_feat)  # (bs,frame_number,feat_out_dim)

        # Branch 1
        x_1 = nn.ReLU()(x)
        output_1 = self.backend(x)
        # Branch 2

        # post-processing on front-end features
        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1)  # add channel
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        # ------------SA for spectral feature-------------#
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # ------------SA for temporal feature-------------#
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output_2 = self.out_layer(last_hidden)

        # Average the scores
        output = torch.mean(torch.stack([output_1, output_2]), dim=0)

        return output


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


# st1 = Distil_W2V2BASE_AASISTL(
#     device, ssl_cpkt_path="/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt")

# st1 = torch.nn.DataParallel(st1).to(device)
# st1.load_state_dict(torch.load(
#     "/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_AASISTL_DKDLoss_cnsl_audiomentations_5/best_checkpoint_11.pth", map_location=device), strict=False)

# st2 = Distil_W2V2BASE_Linear(
#     device, ssl_cpkt_path="/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt")

# st2 = torch.nn.DataParallel(st2).to(device)
# st2.load_state_dict(torch.load(
#     "/datab/hungdx/KDW2V-AASISTL/models/W2V2BASE_Linear_DKDLoss_cnsl_noaudiomentations/best_checkpoint_39.pth", map_location=device))

# fusion_st1_st2 = W2VBASE_Fusion_AASISTL_Linear(
#     device, ssl_cpkt_path="/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt")

# fusion_st1_st2 = torch.nn.DataParallel(fusion_st1_st2).to(device)

# # Load weights from W2V2-BASE Linear
# fusion_st1_st2.load_state_dict(st2.state_dict(), strict=False)

# # Load weights from W2V2-BASE AASIST-L
# fusion_st1_st2.load_state_dict(st2.state_dict(), strict=False)

# # Update the weights of the SSL model
# fusion_st1_st2 = update_ssl_model_weights(
#     fusion_st1_st2, nn.ModuleList([st1, st2]))

# # DEBUG
# sys.exit(0)

if args.is_eval_teacher:
    logger.info("Evaluating teacher model")
    teacher = W2V2_VIB(
        device, ssl_cpkt_path="/datab/hungdx/KDW2V-AASISTL/xlsr2_300m.pt").to(device)

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
        if args.dataset == 'moreko':
            print('Eval moreko')
        file_eval = genSpoof_list_v2(dir_meta=os.path.join(args.database_path, args.protocols_path),
                                     is_train=False, is_dev=False, is_eval=True, special=True if args.dataset == 'moreko' else False)
        logger.info(f'no. of eval trials {len(file_eval)}')
        eval_set = Dataset_cnsl_eval(
            list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, teacher, device, args.eval_output,
                                batch_size=args.batch_size_eval, kd_method=kd_method, is_half=args.half)
    print("Done eval teachet")
    sys.exit(0)


if args.student_model_type in globals() and args.yaml == '':
    logger.info(f"Using {args.student_model_type}")

    student_model = globals()[args.student_model_type](
        device, ssl_cpkt_path="/datab/hungdx/KDW2V-AASISTL/wav2vec_small.pt")

elif args.yaml != '':
    with open(args.yaml, 'r') as f:
        logger.info('Load configuration file {}'.format(args.yaml))
        config = yaml.safe_load(f)
        student_model_name = config['model']['student']['name']
        if student_model_name.startswith('Distil_XLSR_N_Trans_Layer'):
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


if not args.student_model_type == 'AASIST':
    logger.info("Wrapped ssl model to torchaudio")
    student_model.module.ssl_model = W2V2_TA(import_fairseq_model(
        student_model.module.ssl_model.model)).to(device)

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

else:
    kd_method = 'self_KD_Teacher' if args.student_model_type == 'SelfDistil_W2V2BASE_AASISTL' else 'NaN'
    print(kd_method)
    if args.dataset == 'moreko':
        print('Eval moreko')
    file_eval = genSpoof_list_v2(dir_meta=os.path.join(args.database_path, args.protocols_path),
                                 is_train=False, is_dev=False, is_eval=True, special=True if args.dataset == 'moreko' else False)
    logger.info(f'no. of eval trials {len(file_eval)}')
    print("Current padding size is {}".format(args.padding_size))

    eval_set = Dataset_cnsl_eval(
        list_IDs=file_eval, base_dir=os.path.join(args.database_path), padding_size=args.padding_size)
    produce_evaluation_file(eval_set, student_model, device, args.eval_output,
                            batch_size=args.batch_size_eval, kd_method=kd_method, is_half=args.half)
