import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from tensorboardX import SummaryWriter
from startup_config import set_random_seed
from student import Distil_W2V2_AASISTL, Distil_W2V2_AASISTL_Cosine, Distil_W2V2_AASISTL_Regressor, Distil_W2V2BASE_AASISTL, Distil_W2V2BASE_AASISTL_Cosine, Distil_W2V2BASE_AASISTL_Regressor, Distil_W2V2FTBASE_AASISTL, Distil_W2V2BASE_AASISTL_Self_KD, Distil_W2V2BASE_AASISTL_Self_KD_Teacher
from teacher import W2V2_AASIST, W2V2_AASIST_Cosine, W2V2_AASIST_Regressor
from kdtoolkit import train_knowledge_distillation, train_kd_cosine_loss, train_kd_mse_loss, kd_loss_function, feature_loss_function
from menu import get_main_menu
from utils import EarlyStopping, AverageMeter


import logging

# Get the Numba logger
logger = logging.getLogger('numba')
logger.setLevel(logging.WARNING)  # Set level to WARNING, ERROR, or CRITICAL

__author__ = "Hungdx"
__email__ = "hungdx@soongsil.ac.kr"

def sefl_KD_train_epoch(train_loader, model, lr,optimizer, device, scaler, temperature=3, alpha=0.1, beta=1e-6, use_amp = True):
    logging.log(logging.INFO, 'Training self KD with temperature = {} and alpha = {} and beta = {}'.format(temperature, alpha, beta))
    running_loss = 0
    model.train()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_total = 0.0

    

    for batch_x, batch_y in train_loader:
        # Mixed precision training
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):

            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2 = model(batch_x)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # Calculate loss (label loss)
            batch_loss = criterion(batch_out, batch_y)
            spectral_loss = criterion(spectral_output, batch_y)
            temporal_loss = criterion(temporal_output, batch_y)
            graph_loss_S = criterion(graph_output_S, batch_y)
            graph_loss_T = criterion(graph_output_T, batch_y)
            hs_gal_loss_S = criterion(hs_gal_output_S, batch_y)
            hs_gal_loss_T = criterion(hs_gal_output_T, batch_y)

            # Calculate KD loss
            temp = batch_out / temperature
            temp = torch.softmax(temp, dim=1)
            temp_detach = temp.detach()
            kd_spectral_loss = kd_loss_function(temporal_output, temp_detach, temperature) * (temperature**2)
            kd_temporal_loss = kd_loss_function(temporal_output, temp_detach, temperature) * (temperature**2)
            kd_graph_loss_S = kd_loss_function(graph_output_S, temp_detach, temperature) * (temperature**2)
            kd_graph_loss_T = kd_loss_function(graph_output_T, temp_detach, temperature) * (temperature**2)
            kd_hs_gal_loss_S = kd_loss_function(hs_gal_output_S, temp_detach, temperature) * (temperature**2)
            kd_hs_gal_loss_T = kd_loss_function(hs_gal_output_T, temp_detach, temperature) * (temperature**2)

            # Calculate loss (feature loss)
            # We didn't apply backward for final feature
            feature_loss_1 = feature_loss_function(middle_feature1, final_feature1.detach())
            feature_loss_2 = feature_loss_function(middle_feature2, final_feature2.detach())

            # Calculate total loss
            # Total label loss
            total_label_loss = batch_loss + spectral_loss + temporal_loss + graph_loss_S + graph_loss_T + hs_gal_loss_S + hs_gal_loss_T

            # Total KD loss
            total_kd_loss = kd_spectral_loss + kd_temporal_loss + kd_graph_loss_S + kd_graph_loss_T + kd_hs_gal_loss_S + kd_hs_gal_loss_T

            # Total feature loss
            total_feature_loss = feature_loss_1 + feature_loss_2

            total_loss = (1 - alpha) * total_label_loss + alpha * total_kd_loss + beta * total_feature_loss

        # optimizer.zero_grad()
        # total_loss.backward()
        # optimizer.step()
            
        # Scaler
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        

        running_loss += (total_loss.item() * batch_size)
    running_loss /= num_total
    return running_loss

def sefl_KD_teacher_train_epoch(train_loader, student, teacher, lr,optimizer, device, scaler, temperature=3, alpha=0.1, beta=1e-6, hidden_rep_loss_weight=0.5, use_amp = True):
    logging.log(logging.INFO, 'Training self KD + teacher cosine with temperature = {} and alpha = {} and beta = {} and hidden_rep_loss_weight = {}'.format(temperature, alpha, beta, hidden_rep_loss_weight))
    running_loss = 0
    cosine_loss = nn.CosineEmbeddingLoss()
    student.train()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_total = 0.0

    for batch_x, batch_y in train_loader:
        # Mixed precision training
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):

            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2, student_hidden_representation = student(batch_x)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # Get teacher output
            with torch.no_grad():
                _, teacher_hidden_representation = teacher(batch_x)
            
            student_hidden_representation = student_hidden_representation.view(batch_size, -1)
            teacher_hidden_representation = teacher_hidden_representation.view(batch_size, -1)

            # Now pass the reshaped tensors to cosine_loss
            hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation, target=torch.ones(batch_size).to(device))

            # Calculate loss (label loss)
            batch_loss = criterion(batch_out, batch_y)
            spectral_loss = criterion(spectral_output, batch_y)
            temporal_loss = criterion(temporal_output, batch_y)
            graph_loss_S = criterion(graph_output_S, batch_y)
            graph_loss_T = criterion(graph_output_T, batch_y)
            hs_gal_loss_S = criterion(hs_gal_output_S, batch_y)
            hs_gal_loss_T = criterion(hs_gal_output_T, batch_y)

            # Calculate KD loss
            temp = batch_out / temperature
            temp = torch.softmax(temp, dim=1)
            temp_detach = temp.detach()
            kd_spectral_loss = kd_loss_function(temporal_output, temp_detach, temperature) * (temperature**2)
            kd_temporal_loss = kd_loss_function(temporal_output, temp_detach, temperature) * (temperature**2)
            kd_graph_loss_S = kd_loss_function(graph_output_S, temp_detach, temperature) * (temperature**2)
            kd_graph_loss_T = kd_loss_function(graph_output_T, temp_detach, temperature) * (temperature**2)
            kd_hs_gal_loss_S = kd_loss_function(hs_gal_output_S, temp_detach, temperature) * (temperature**2)
            kd_hs_gal_loss_T = kd_loss_function(hs_gal_output_T, temp_detach, temperature) * (temperature**2)

            # Calculate loss (feature loss)
            # We didn't apply backward for final feature
            feature_loss_1 = feature_loss_function(middle_feature1, final_feature1.detach())
            feature_loss_2 = feature_loss_function(middle_feature2, final_feature2.detach())

            # Calculate total loss
            # Total label loss
            total_label_loss = batch_loss + spectral_loss + temporal_loss + graph_loss_S + graph_loss_T + hs_gal_loss_S + hs_gal_loss_T

            # Total KD loss
            total_kd_loss = kd_spectral_loss + kd_temporal_loss + kd_graph_loss_S + kd_graph_loss_T + kd_hs_gal_loss_S + kd_hs_gal_loss_T

            # Total feature loss
            total_feature_loss = feature_loss_1 + feature_loss_2

            # Total loss
            total_loss = (1 - alpha) * total_label_loss + alpha * total_kd_loss + beta * total_feature_loss + hidden_rep_loss_weight * hidden_rep_loss

        # Scaler
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        

        running_loss += (total_loss.item() * batch_size)
    running_loss /= num_total
    return running_loss

def sefl_KD_val_epoch(dev_loader, model,  device):
    logging.log(logging.INFO, 'Validation self KD')
    val_loss = 0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    num_total = 0.0


    with torch.inference_mode():
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_out, spectral_output, temporal_output, graph_output_S, graph_output_T, hs_gal_output_S, hs_gal_output_T, middle_feature1, middle_feature2, final_feature1, final_feature2 = model(batch_x)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            # Calculate loss (label loss)
            batch_loss = criterion(batch_out, batch_y)
            spectral_loss = criterion(spectral_output, batch_y)
            temporal_loss = criterion(temporal_output, batch_y)
            graph_loss_S = criterion(graph_output_S, batch_y)
            graph_loss_T = criterion(graph_output_T, batch_y)
            hs_gal_loss_S = criterion(hs_gal_output_S, batch_y)
            hs_gal_loss_T = criterion(hs_gal_output_T, batch_y)

            # Total label loss
            total_label_loss = batch_loss + spectral_loss + temporal_loss + graph_loss_S + graph_loss_T + hs_gal_loss_S + hs_gal_loss_T
            
            total_loss = total_label_loss


            val_loss += (total_loss.item() * batch_size)
        val_loss /= num_total
        return val_loss


def evaluate_accuracy(dev_loader, model, device, kd_method=None):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)

            if kd_method == 'KD_logits':
                batch_out = model(batch_x)
            elif kd_method == 'KD_cosine':
                batch_out, _ = model(batch_x)
            elif kd_method == 'KD_mse':
                batch_out, _ = model(batch_x)
            else:
                batch_out = model(batch_x)
                
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)
            
        val_loss /= num_total
   
    return val_loss

def produce_evaluation_file(dataset, model, device, save_path, kd_method=None, batch_size=4):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    fname_list = []
    score_list = []

    with torch.no_grad():
        for batch_x,utt_id in data_loader:
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

def train_epoch(train_loader, model, lr,optim, device):
    running_loss = 0
    
    num_total = 0.0
    
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in train_loader:
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
       
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    
    return running_loss



if __name__ == '__main__':
    
    if not os.path.exists('models'):
        os.mkdir('models')
    args = get_main_menu()
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    track = args.track

    assert track in ['LA', 'PA','DF'], 'Invalid track given'

    #database
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    #define model saving path
    model_tag = 'model_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))

    if args.KD_logits:
        model = W2V2_AASIST()
        if args.ssl_type == 'Distil_XLSR':
            student = Distil_W2V2_AASISTL(device)
        elif args.ssl_type == 'ft':
             # W2V2BASE 95M fine-tuned
            student = Distil_W2V2FTBASE_AASISTL(device)
        else:
            # W2V2BASE 95M
            student = Distil_W2V2BASE_AASISTL(device)
        kd_method = 'KD_logits'

    elif args.KD_cosine:
        model = W2V2_AASIST_Cosine()
        if args.ssl_type == 'Distil_XLSR':
            student = Distil_W2V2_AASISTL_Cosine(device)
        else:
            # W2V2BASE 95M
            student = Distil_W2V2BASE_AASISTL_Cosine(device)
        kd_method = 'KD_cosine'

    elif args.KD_mse:
        model = W2V2_AASIST_Regressor()
        if args.ssl_type == 'Distil_XLSR':
            student = Distil_W2V2_AASISTL_Regressor(device)
        else:
            # W2V2BASE 95M
            student = Distil_W2V2BASE_AASISTL_Regressor(device)
        kd_method = 'KD_mse'
    elif args.self_KD:
        student = Distil_W2V2BASE_AASISTL_Self_KD(device)
        kd_method = 'self_KD'
    else:
        raise ValueError('Invalid KD method given')

    #print model parameters
    if not args.self_KD:
        nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
        model =nn.DataParallel(model).to(device)
        print('Teacher nb_params:',nb_params)

    nb_params = sum([param.view(-1).size()[0] for param in student.parameters()])
    student = nn.DataParallel(student).to(device)
    print('Student nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    if args.student_restore:
        try:        
            # Restore student model from best checkpoint
            cpt = sorted(os.listdir(model_save_path), key=lambda x: int(x.split('_')[2].split('.')[0]) if not x.startswith('epoch') else 0 )[-1]
            
            # Restore student model from last checkpoint
            # last_cpt = sorted(os.listdir(model_save_path), key=lambda x: int(x.split('_')[1].split('.')[0]) if not x.startswith('best') else 0 )[-1]

            student.load_state_dict(torch.load(os.path.join(model_save_path, cpt)))
            print('Student model loaded : {}'.format(os.path.join(model_save_path, cpt)))
            # print('Training from epoch {}'.format(int(last_cpt.split('_')[1].split('.')[0])))
        except Exception as e:
            print('No checkpoint student found in ', model_save_path)
            print(e)
            print('Training from scratch')

    if args.student_model_path:
        print('Loading student model from {}'.format(args.student_model_path))
        try:
            student.load_state_dict(torch.load(args.student_model_path,map_location=device))
            student.load_state_dict(torch.load(args.student_model_path,map_location=device))
            print('Student model loaded : {}'.format(args.student_model_path))
        except Exception as e:
            print('No checkpoint student found in ', args.student_model_path)
            print(e)
            print('Training from scratch')

    #evaluation 
    if args.eval:
        _,file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_{}_cm_protocols/{}.cm.eval.trl.txt'.format(track,prefix_2021)),is_train=False,is_eval=True, num_eval_samples=args.num_eval_samples)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(args.track)))

        # Produce evaluation file 
        produce_evaluation_file(eval_set, model if args.is_eval_teacher else student , device, args.eval_output, batch_size=args.batch_size_eval, kd_method=kd_method)
        
        sys.exit(0)
   
    
     
    # define train dataloader
    d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'),is_train=True,is_eval=False)
    
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_ASVspoof2019_train(args,list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_train/'),algo=args.algo)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn
    

    # define dev (validation) dataloader

    d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),is_train=False,is_eval=False)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_ASVspoof2019_train(args,list_IDs = file_dev,labels = d_label_dev,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_dev/'),algo=args.algo)

    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)

    del dev_set,d_label_dev

    
    

    # Training and validation
    start_epoch = 0 if not args.student_restore else int(cpt.split('_')[2].split('.')[0]) + 1
    assert start_epoch == 0 or type(start_epoch) == int, 'Invalid start epoch given'
    print('Start epoch: {}'.format(start_epoch))
    
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    early_stopping = EarlyStopping(patience=7, verbose=True, model_save_path=model_save_path)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    for epoch in range(start_epoch, num_epochs):
        if args.KD_logits:
            if args.ssl_type != 'Distil_XLSR' and args.ssl_type != 'ft':
                running_loss = train_knowledge_distillation(model, student, train_loader, optimizer, T=4.10431, soft_target_loss_weight=0.498657, ce_loss_weight=0.640261, device=device)
            else:
                running_loss = train_knowledge_distillation(model, student, train_loader, optimizer, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
            KD_method = 'KD_logits'
        elif args.KD_cosine:
            if args.ssl_type != 'Distil_XLSR':
                running_loss = train_kd_cosine_loss(model, student, train_loader, optimizer, hidden_rep_loss_weight=0.933, ce_loss_weight=0.059, device=device)
            else:
                running_loss = train_kd_cosine_loss(model, student, train_loader, optimizer, hidden_rep_loss_weight=0.25, ce_loss_weight=0.75, device=device)
            KD_method = 'KD_cosine'
        elif args.KD_mse:
            if args.ssl_type != 'Distil_XLSR':
                running_loss = train_kd_cosine_loss(model, student, train_loader, optimizer, hidden_rep_loss_weight=0.742208, ce_loss_weight=0.483691, device=device)
            else:
                running_loss = train_kd_mse_loss(model, student, train_loader, optimizer, feature_map_weight=0.25, ce_loss_weight=0.75, device=device)
            KD_method = 'KD_mse'
        elif args.self_KD:
            # Debug
            # Create dummy input data
            # for inputs, targets in train_loader:
            #     print(inputs.shape)
            #     break
            # inputs = torch.randn(64, 64600)  # Assuming input size is (3, 32, 32)

            # # Create dummy target data
            # targets = torch.randint(0, 2, (64,))  # Assuming binary classification

            # # Create a TensorDataset
            # dataset = TensorDataset(inputs, targets)

            # # Create a DataLoader
            # train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            running_loss = sefl_KD_train_epoch(train_loader, student, args.lr, optimizer, device, scaler, args.use_amp)
            KD_method = 'self_KD'
        else:
            logging.log(logging.ERROR, 'Invalid KD method given')
            raise ValueError('Invalid KD method given')
        
        # Validate student and save model
        if args.self_KD:
            val_loss = sefl_KD_val_epoch(dev_loader, student, device)
        else:
            val_loss = evaluate_accuracy(dev_loader, student, device, kd_method=KD_method)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {} '.format(epoch, running_loss,val_loss))

        if args.use_amp:
            checkpoint = {"model": student.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
            torch.save(checkpoint, os.path.join( model_save_path, 'epoch_{}.pth'.format(epoch)))
            
        else:
            torch.save(student.state_dict(), os.path.join( model_save_path, 'epoch_{}.pth'.format(epoch)))

        # Remove previous checkpoint
        if epoch > 0:
            prev_checkpoint = os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch - 1))
            if os.path.exists(prev_checkpoint):
                os.remove(prev_checkpoint)

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss, student, epoch)

        if early_stopping.early_stop:
            logging.log(logging.INFO, "Early stopping")
            break

        
    print('Finished Training')
