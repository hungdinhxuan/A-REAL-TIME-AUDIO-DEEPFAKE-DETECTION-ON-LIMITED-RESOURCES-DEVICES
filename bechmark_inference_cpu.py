# import os
# from menu import get_main_benchmark_cpu
# from main import produce_evaluation_file
# from data_utils import genSpoof_list,Dataset_ASVspoof2021_eval
# import torch
# from startup_config import set_random_seed
# from student import Distil_W2V2_AASISTL, Distil_W2V2_AASISTL_Cosine, Distil_W2V2_AASISTL_Regressor, Distil_W2V2BASEHG_AASISTL_Self_KD
# from torch import nn
# from torch.utils.data import DataLoader
# import time
# from torch import Tensor
# from prettytable import PrettyTable
# # Disable GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# def produce_evaluation_file(dataset, model, device, batch_size=250, number_warmup=0, save_path='eval.txt'):
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
#     model.eval()

#     avg_time = 0
#     print('Number of samples: {}'.format(len(data_loader)))
#     print('Start inference...')
#     with torch.inference_mode():
#         start_warmp = False
#         for batch_x,utt_id in data_loader:
#             fname_list = []
#             score_list = []
#             batch_size = batch_x.size(0)
#             batch_x = batch_x.to(device)

#             # Warmup
#             # if not start_warmp:
#             #     print('Warmup...')
#             #     for i in range(number_warmup):
#             #         model(batch_x)
#             #     start_warmp = True
#             #     print('Warmup done.')

#             start_time = time.time()
#             batch_out = model(batch_x)

#             batch_score = (batch_out[:, 1]
#                         ).data.cpu().numpy().ravel()
#             # add outputs
#             fname_list.extend(utt_id)
#             score_list.extend(batch_score.tolist())

#             with open(save_path, 'a+') as fh:
#                 for f, cm in zip(fname_list,score_list):
#                     fh.write('{} {}\n'.format(f, cm))
#             fh.close()

#             # Compute average time in ms
#             avg_time += ((time.time() - start_time)) * 1000
#             #print('Inference time: {:.2f} ms'.format(avg_time))

#     print('Average time: {:.2f} ms'.format(avg_time / len(data_loader)))


# args = get_main_benchmark_cpu()
# device = "cpu"
# set_random_seed(args.seed)


# torch.set_num_threads(args.num_threads)

# track = "DF"
# prefix_2021 = 'ASVspoof2021.{}'.format(track)


# # Load model
# # if args.KD_logits:
# #     model = Distil_W2V2_AASISTL(device)
# #     kd_method = 'KD_logits'

# # elif args.KD_cosine:
# #     model = Distil_W2V2_AASISTL_Cosine(device)
# #     kd_method = 'KD_cosine'
# # elif args.KD_mse:
# #     model = Distil_W2V2_AASISTL_Regressor(device)
# #     kd_method = 'KD_mse'
# # else:
# #     raise ValueError("Unknown KD method")
# # # Produce evaluation file

# # nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
# # model = nn.DataParallel(model).to(device)
# # print('Number of parameters: {}'.format(nb_params))
# # model.load_state_dict(torch.load(args.model_path,map_location=device))
# # print(f"Model {kd_method} loaded from {args.model_path}")
# model = torch.jit.load("Distil_W2V2BASEHG_AASISTL_Self_KD_quantized_optimized.pt")
# model = nn.DataParallel(model).to(device)
# # Load eval set
# # _,file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_{}_cm_protocols/{}.cm.eval.trl.txt'.format(track,prefix_2021)),is_train=False,is_eval=True, num_eval_samples=args.num_eval_samples)
# # print('no. of eval trials',len(file_eval))
# # eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(track)))

# # produce_evaluation_file(eval_set, model , device, save_path="Distil_W2V2BASEHG_AASISTL_Self_KD_quantized_optimized_150k.txt")


# # Unwrap the model from DataParallel
# model = model.module

# # Count the number of parameters
# num_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters in the model: {num_params}")


import numpy as np
import torch
from teacher import W2V2_AASIST
from aasist.AASIST import Model as AASIST
import time
optimal_batch_size = 1
device = torch.device("cpu")
# model = W2V2_AASIST(
#     device, ssl_cpkt_path='/datab/hungdx/KDW2V-AASISTL/xlsr2_300m.pt')
config = {

    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0]
}

model = AASIST(config)
model.to(device)
dummy_input = torch.randn(optimal_batch_size, 64600,
                          dtype=torch.float).to(device)


####################################################### This is the code to benchmark the throughput of the model #######################################################
# repetitions = 100
# total_time = 0
# with torch.no_grad():
#     for rep in range(repetitions):
#         starter, ender = torch.cuda.Event(
#             enable_timing=True),   torch.cuda.Event(enable_timing=True)

#         starter.record()
#         _ = model(dummy_input)
#         ender.record()
#         torch.cuda.synchronize()
#         curr_time = starter.elapsed_time(ender)/1000
#         total_time += curr_time
# Throughput = (repetitions*optimal_batch_size)/total_time
# print('Final Throughput:', Throughput)  # in samples/sec

####################################################### This is the code to benchmark the inference time of the model #######################################################
# INIT LOGGERS
starter, ender = torch.cuda.Event(
    enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings = np.zeros((repetitions, 1))
# GPU-WARM-UP
for _ in range(10):
    _ = model(dummy_input)
# MEASURE PERFORMANCE
# with torch.no_grad():
#     for rep in range(repetitions):
#         starter.record()
#         _ = model(dummy_input)
#         ender.record()
#         # WAIT FOR GPU SYNC
#         torch.cuda.synchronize()
#         curr_time = starter.elapsed_time(ender)
#         timings[rep] = curr_time

# mean_syn = np.sum(timings) / repetitions
# std_syn = np.std(timings)
# print(mean_syn)
with torch.no_grad():
    for rep in range(repetitions):
        start_time = time.time()
        _ = model(dummy_input)
        end_time = time.time()

        curr_time = end_time - start_time
        timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print(mean_syn)  # in seconds
