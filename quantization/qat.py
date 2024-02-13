import sys
sys.path.append("..")
from student import Distil_SSL_WAV2VEC2_TA_Self_KD_Teacher2
import torch
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from menu import get_main_menu
from torch.utils.data import DataLoader
import os
import numpy as np
import torch.quantization
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub
from torch import nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.nn.functional as F
import logging
from tqdm import tqdm

def get_train_dev_dataloader(args):
    
       # define train dataloader
    d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),is_train=True,is_eval=False)
    
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_train/'),algo=args.algo)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=2, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn
    

    # define dev (validation) dataloader

    d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),is_train=False,is_eval=False)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_ASVspoof2019_train(args,list_IDs = file_dev,labels = d_label_dev,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_dev/'),algo=args.algo)

    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=2, shuffle=False)

    del dev_set,d_label_dev

    return train_loader, dev_loader
def produce_evaluation_file(dataset, model, device, save_path, kd_method=None, batch_size=4):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, pin_memory_device=device)
    model.eval()
    fname_list = []
    score_list = []

    with torch.no_grad():
        for batch_x,utt_id in tqdm(data_loader):
            fname_list = []
            score_list = []  
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            
            if kd_method == 'KD_logits':
                batch_out = model(batch_x)
            elif kd_method == 'KD_cosine':
                batch_out, _ = model(batch_x)
            elif kd_method == 'KD_mse':
                batch_out, _ = model(batch_x)
            elif kd_method == 'self_KD':
                batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2  = model(batch_x)
            elif kd_method == 'self_KD_Teacher':
                batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, hidden_features  = model(batch_x)
            else:
                batch_out = model(batch_x)
          
            batch_score = (batch_out[:, 1]  
                        ).data.cpu().numpy().ravel() 
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            
            with open(save_path, 'a+') as fh:
                for f, cm in zip(fname_list,score_list):
                    fh.write('{} {}\n'.format(f, cm))
            fh.close()   
    print('Scores saved to {}'.format(save_path))
def train_epoch(train_loader, model, optimizer, device):

    running_loss = 0
    
    num_total = 0.0
    
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight) #define loss function
    
    for batch_x, batch_y in tqdm(train_loader):
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)

        # 1, 0
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, hidden_features  = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    
    return running_loss
def print_size_of_model(model):
    """ Prints the real size of the model """
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
#calculate the accuracy
def self_KD_teacher_val_epoch(dev_loader, model, device, kd_method='self_KD_Teacher'):
    logging.log(logging.INFO, 'Validation Teacher self KD')
    val_loss = 0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_total = 0.0
    num_correct = 0.0

    with torch.inference_mode():
        for batch_x, batch_y in tqdm(dev_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            if kd_method == 'self_KD_Teacher':
                batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, hidden_features = model(batch_x)
            else:
                batch_out, batch_out2, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, hidden_features = model(batch_x)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            
            #true label
            # Calculate loss (label loss)
            batch_loss = criterion(batch_out, batch_y)

            # # Return [batch_size, 2]

            probabilities = F.softmax(batch_out, dim=1)
            predicted_labels = (probabilities[:,0] >= 0.5).int()
            #batch_pred = torch.sigmoid(batch_out)

            # # Batch prediction label if batch_pred > 0.5 then label = 1 else label = 0
            # # In sigmoid it will return like [0.1, 0.9] where 0.1 is fake and 0.9 is genuine. However, we just care about fake value
            # # So we just take the first value and compare it with 0.5 to get the label
            # # If first value > 0.5 then label = 1 else label = 0
            
            
            # # Batch prediction label if batch_pred > 0.5 then label = 1 else label = 0
            num_correct += (predicted_labels == batch_y).sum().item()
        accuracy = (num_correct / num_total) * 100
        print("accuracy",accuracy)
        return accuracy
args = get_main_menu()
device = "cuda" if torch.cuda.is_available() else "cpu"
#Load data
train_loader, dev_loader = get_train_dev_dataloader(args)
#Define a new model
model = Distil_SSL_WAV2VEC2_TA_Self_KD_Teacher2(device, fe='SSL_WAV2VEC2_ASR_BASE_960H_TA')
model = torch.nn.DataParallel(model).to(device)
conv_path=model.module.ssl_model.model.encoder.transformer.pos_conv_embed.conv
conv_path_second=model.module.ssl_model.model.feature_extractor.conv_layers

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

#QAT
model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack') #we choose quantization backend
conv_path.qconfig=None #ParametrizedConv1d not provided pytorch
conv_path_second.qconfig=None 
torch.quantization.prepare_qat(model, inplace=True)
model=model.to(device)
num_epochs = 1  # 에포크 수 설정
# num_epochs = args.num_epochs #100
best_accuracy=0.0
# print("start train")
# for epoch in range(num_epochs):
#     running_loss = train_epoch(train_loader, model, optimizer, device=device)
#     accuracy=self_KD_teacher_val_epoch(dev_loader=dev_loader,model=model,kd_method='self_KD_Teacher', device=device)
#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Accuracy: {accuracy}')
#     filename = f'model_epoch_{epoch+1}_acc_{accuracy:.4f}_loss_{running_loss:.4f}.pth'
    
#     if os.path.exists(filename):
#         os.remove(filename)
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         torch.save(model.state_dict(), filename)        
#     else:
#         print("Early stopping")        
#         break
# print("finished training")
model=model.cpu()
print("load model..")
model_path="model_epoch_1_acc_49.8712_loss_0.9742.pth"
model.load_state_dict(torch.load(model_path,map_location=device))
print("finished load")
# print("Size of model before quantization")
# print_size_of_model(model)
print("quantization..")   
torch.quantization.convert(model.eval() , inplace=True)
print("quantization finished.")
#model.eval()  # Set the model to evaluation mode
# correct = 0
# total = 0
# with torch.no_grad():  # No need to track gradients during evaluation
#     for data in dev_loader:
#         inputs, labels = data
#         inputs, labels = inputs.cpu(), labels.cpu()
        
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = 100 * correct / total
# print(f'Accuracy of the model on the dev dataset: {accuracy}%')

# #filename="QATmodel_epoch5.pth"
# #torch.save(model.state_dict(), filename)
#print("Size of model after quantization")
#print_size_of_model(model) 
# # print("load model..")
# # model_path="QATmodel_epoch5.pth"
# # model.load_state_dict(torch.load(model_path,map_location=device))
# # print("finished load")
print("calcul accuracy...")
accuracy=self_KD_teacher_val_epoch(dev_loader=dev_loader,model=model.eval(),kd_method='self_KD_Teacher', device='cpu')
print(accuracy)

# Export the quantized model for run on mobile devices
# scripted_model = torch.jit.script(model)
# optimized_model = optimize_for_mobile(scripted_model)
# optimized_model._save_for_lite_interpreter("Distil_SSL_WAV2VEC2_TA_Self_KD_Teacher2.ptl")
# print("Done _save_for_lite_interpreter")









