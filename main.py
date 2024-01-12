import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_utils import genSpoof_list,Dataset_ASVspoof2019_train,Dataset_ASVspoof2021_eval
from tensorboardX import SummaryWriter
from startup_config import set_random_seed
from student import Distil_W2V2_AASISTL, Distil_W2V2_AASISTL_Cosine, Distil_W2V2_AASISTL_Regressor
from teacher import W2V2_AASIST, W2V2_AASIST_Cosine, W2V2_AASIST_Regressor
from kdtoolkit import train_knowledge_distillation, train_kd_cosine_loss, train_kd_mse_loss
from menu import get_model
from utils import EarlyStopping

__author__ = "Hungdx"
__email__ = "hungdx@soongsil.ac.kr"


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

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=14, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    fname_list = []
    key_list = []
    score_list = []
    
    for batch_x,utt_id in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        
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
    args = get_model()
 
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
        student = Distil_W2V2_AASISTL()

    elif args.KD_cosine:
        model = W2V2_AASIST_Cosine()
        student = Distil_W2V2_AASISTL_Cosine()

    elif args.KD_mse:
        model = W2V2_AASIST_Regressor()
        student = Distil_W2V2_AASISTL_Regressor()
    
    else:
        raise ValueError('Invalid KD method given')

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
        
        last_cpt = sorted(os.listdir(model_save_path), key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
        student.load_state_dict(torch.load(os.path.join(model_save_path, last_cpt)))
        print('Student model loaded : {}'.format(os.path.join(model_save_path, last_cpt)))

    #evaluation 
    if args.eval:
        file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_{}_cm_protocols/{}.cm.eval.trl.txt'.format(track,prefix_2021)),is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2021_{}_eval/'.format(args.track)))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
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
    start_epoch = 0 if not args.student_restore else int(last_cpt.split('_')[1].split('.')[0])
    assert start_epoch == 0 or type(start_epoch) == int, 'Invalid start epoch given'
    print('Start epoch: {}'.format(start_epoch))
    
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    early_stopping = EarlyStopping(patience=7, verbose=True, model_save_path=model_save_path)
    
    for epoch in range(start_epoch, num_epochs):
        if args.KD_logits:
            running_loss = train_knowledge_distillation(model, student, train_loader, optimizer, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
            KD_method = 'KD_logits'
        elif args.KD_cosine:
            running_loss = train_kd_cosine_loss(model, student, train_loader, optimizer, hidden_rep_loss_weight=0.25, ce_loss_weight=0.75, device=device)
            KD_method = 'KD_cosine'
        elif args.KD_mse:
            running_loss = train_kd_mse_loss(model, student, train_loader, optimizer, feature_map_weight=0.25, ce_loss_weight=0.75, device=device)
            KD_method = 'KD_mse'
        else:
            raise ValueError('Invalid KD method given')
        
        # Validate student and save model
        val_loss = evaluate_accuracy(dev_loader, student, device, kd_method=KD_method)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {} '.format(epoch, running_loss,val_loss))
        torch.save(student.state_dict(), os.path.join( model_save_path, 'epoch_{}.pth'.format(epoch)))

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss, student, epoch)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        
    print('Finished Training')
