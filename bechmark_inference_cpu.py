import os
from menu import get_main_benchmark_cpu
from main import produce_evaluation_file
from data_utils import genSpoof_list,Dataset_ASVspoof2021_eval
import torch
from startup_config import set_random_seed
from student import Distil_W2V2_AASISTL, Distil_W2V2_AASISTL_Cosine, Distil_W2V2_AASISTL_Regressor
from torch import nn
from torch.utils.data import DataLoader
import time

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def produce_evaluation_file(dataset, model, device, batch_size=1, number_warmup=2):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()

    avg_time = 0
    print('Number of samples: {}'.format(len(data_loader)))
    print('Start inference...') 
    with torch.inference_mode():
        start_warmp = False
        for batch_x,_ in data_loader:
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)

            # Warmup
            if not start_warmp:
                print('Warmup...')
                for i in range(number_warmup):
                    model(batch_x)
                start_warmp = True
                print('Warmup done.')

            start_time = time.time()
            model(batch_x)

            # Compute average time in ms
            avg_time += ((time.time() - start_time)) * 1000
            #print('Inference time: {:.2f} ms'.format(avg_time))
          
    print('Average time: {:.2f} ms'.format(avg_time / len(data_loader)))


args = get_main_benchmark_cpu()
device = "cpu"
set_random_seed(args.seed)


torch.set_num_threads(args.num_threads)

track = "DF"
prefix_2021 = 'ASVspoof2021.{}'.format(track)


# Load model
if args.KD_logits:
    model = Distil_W2V2_AASISTL(device)
    kd_method = 'KD_logits'

elif args.KD_cosine:
    model = Distil_W2V2_AASISTL_Cosine(device)
    kd_method = 'KD_cosine'
elif args.KD_mse:
    model = Distil_W2V2_AASISTL_Regressor(device)
    kd_method = 'KD_mse'
else:
    raise ValueError("Unknown KD method")
# Produce evaluation file 

nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
model = nn.DataParallel(model).to(device)
print('Number of parameters: {}'.format(nb_params))

if args.is_quant:
    model.load_state_dict(torch.load(args.model_path,map_location=device), strict=False)
    print("Quantized model loaded from {}".format(args.model_path))
else:
    model.load_state_dict(torch.load(args.model_path,map_location=device))
    print(f"Model {kd_method} loaded from {args.model_path}")


# Load eval set
file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_{}_cm_protocols/{}.cm.eval.trl.txt'.format(track,prefix_2021)),is_train=False,is_eval=True, num_eval_samples=args.num_eval_samples)
print('no. of eval trials',len(file_eval))
eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(track)))

produce_evaluation_file(eval_set, model , device)