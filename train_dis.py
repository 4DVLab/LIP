import argparse
import os
from models.LIP_model import PP_distil
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm
import time
import pickle
from dataset.LIP_dataset import Dataset_dis

def test_one_epoch(model,device,test_loader,sv_name=None):
    model.eval()
    eval_joints = 0.0
    count = 0
    result = []
    for _,data in enumerate(tqdm(test_loader)):
        seq_pc = data['data']
        seq_gt = data['gt_j']
        seq_pc = seq_pc.to(device).float()
        seq_gt = seq_gt.to(device).float()
        out,_ = model(seq_pc,device)
        result.append(list(out.cpu().detach().numpy()[:,:,:]))
        loss1 = np.linalg.norm(out[:,:,6:].cpu().detach().numpy().reshape(-1,24,3)-seq_gt.cpu().detach().numpy().reshape(-1,24,3),axis=2).mean(axis=1).mean()
        eval_joints +=loss1.item()
        count+=1
    eval_joints = float(eval_joints)/count
    print('mpjpe in stage 1: ',eval_joints)

    result = np.concatenate(result).reshape(-1,32,78)
    if sv_name!=None:
        os.makedirs(sv_name,exist_ok=True)
        with open(os.path.join(sv_name,"dis.pkl"),'wb') as f:
           pickle.dump(result,f)

    return eval_joints

def train_one_epoch(model,device,train_loader,optimizer):
    model.train()
    train_loss1 = 0.0
    loss_fn = torch.nn.MSELoss(size_average=False)
    count = 0
    for _,data in (enumerate(tqdm(train_loader))):
        seq_pc = data['data']
        B,T = seq_pc.shape[0],seq_pc.shape[1]
        seq_gt = data['gt']
        seq_pc = seq_pc.to(device)
        seq_gt = seq_gt.to(device)
        out,_ = model(seq_pc,device)
        loss = (loss_fn(out.reshape(B,T,-1).float(),seq_gt.reshape(B,T,-1).float()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss1 +=loss.item()

        count+=1
    train_loss1 = float(train_loss1)/count

    return train_loss1
    
def train(model,device,train_loader,test_loader,args):
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(learnable_params,lr=args.lr,betas=(.9, .999),weight_decay=1e-4)
    best_loss = 10000000.0
    save_path = os.path.join(args.save_path,args.exp_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for epoch in range(args.start_epoch,args.start_epoch+args.epochs):
        time_s = time.time()
        train_loss1 = train_one_epoch(model,device,train_loader,optimizer)
        test_loss = test_one_epoch(model,device,test_loader)
        if test_loss<best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(),os.path.join(save_path,'dis_best_model.t7'))

        time_e = time.time()
        print('Epoch: %d, Best Loss: %f, Train Loss: %f, Cost Time: %f'%(epoch+1,best_loss,train_loss1,time_e-time_s))

def options():
    parser = argparse.ArgumentParser(description='Baseline network')
    parser.add_argument('--save_path',type=str,default='./save_models/')
    parser.add_argument('--test_data',type=str,default='LIPD_test.pkl')
    parser.add_argument('--train_data',type=str,default='LIPD_train.pkl')
    parser.add_argument('-n','--exp_name',type=str,default='LIP/')
    parser.add_argument('--save_name',type=str,default='./dis_results/LIPD/')
    parser.add_argument('--root_dataset_path',type=str,default='path_to_root_dir')
    parser.add_argument('--train',type=int,default=1)

    # settings of input data
    parser.add_argument('--frames',type=int,default=32)
    parser.add_argument('--num_points',type=int,default=256)

    #settings of training
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--start_epoch',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=200)
    
    parser.add_argument('--pretrained',default=None)
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()
    return args

def load_GPUS(model,model_path):
    state_dict = torch.load(model_path,map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model

def main():
    args = options()
    device = args.device
    if args.train:
        train_dataset = Dataset_dis(args,'t')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,drop_last=True,pin_memory=False)
    test_dataset = Dataset_dis(args,'e')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,drop_last=True,pin_memory=False)
    
    model = PP_distil(k=24*3+6,channel=3)
    if args.pretrained != None:
        pretrained = args.pretrained+'dis_model.t7'
        state_dict = torch.load(pretrained,map_location='cpu')
        model.load_state_dict(state_dict)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        # print(torch.cuda.device_count())

    model.to(device)
    if args.train:
        train(model,device,train_loader,test_loader,args)
    else:
        test_one_epoch(model,device,test_loader,args.save_name)

if __name__ == "__main__":
    main()

