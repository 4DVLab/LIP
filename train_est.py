import argparse
import torch
from models.LIP_model import Module_LINet
from dataset.LIP_dataset import Dataset_est
from scipy.spatial.transform import Rotation as R
import time
from tqdm import tqdm
import os
from models.loss_utils import *
import sys
import pickle
sys.path.append("./smpl")
from smpl import SMPL, SMPL_MODEL_DIR

def test_one_epoch(lipnet,testDataLoader,device,smpl,sv_name=None):
    lipnet.eval()
    loss_a = 0.0
    loss_v = 0.0
    loss_m = 0.0
    index=0
    result_p = []
    result_j = []

    for i, batch in enumerate(tqdm(testDataLoader)):
        pc_input = batch['data'].to(device)
        imu_acc = batch['imu_acc'].to(device)
        imu_input = batch['imu_ori'].to(device)
        gt_pose = batch['gt_pose'].to(device)

        nn_output = lipnet((pc_input,imu_input,imu_acc))
        output = nn_output['body_pose']
        joint = nn_output['joint_position']
        B,T,_ = output.shape
        pose = output.view(B*T,-1,6)
        gt_p = gt_pose.view(B*T,72)
        pose = matrix_to_axis_angle(rotation_6d_to_matrix(pose).view(-1, 3,3)).reshape(-1, 72)
        result_j.append(joint.cpu().detach().numpy().reshape(-1,72))
        result_p.append(pose.cpu().detach().numpy().reshape(-1,72))
        mpjpe = eval(gt_p,pose,smpl)
        mpvpe,anger = eval_v(gt_p,pose,smpl)

        loss_m+=mpjpe
        loss_v+=mpvpe
        loss_a+=anger
        index+=1
    result = {
        'pose':np.concatenate(result_p).reshape(-1,32,72),
        'joint':np.concatenate(result_j).reshape(-1,32,72),
    }
    if sv_name!=None:
        os.makedirs(sv_name,exist_ok=True)
        with open(f'{sv_name}/est.pkl','wb') as f:
           pickle.dump(result,f)

    print('--------------------')
    print('MPJPE:',loss_m/index)
    print('mesh error:',loss_v/index)
    print('anger error:',loss_a/index)
    return (loss_m+loss_v+loss_a)/(index*3)

def train_one_epoch(lipnet,device,trainDataLoader,optimizer):

    lipnet.train()
    total_loss = 0.0
    limb_position_loss = 0.0
    joint_position_loss = 0.0
    pose_loss = 0.0
    fk_loss = 0.0
    index = 0
    for i, batch in enumerate(tqdm(trainDataLoader)):

        pc_input = batch['data'].to(device)
        imu_input = batch['imu_ori'].to(device)
        imu_acc = batch['imu_acc'].to(device)
        gt_pose = batch['gt_pose'].to(device)
        gt_joint = batch['gt_j'].to(device)
        
        # optimize
        optimizer.zero_grad()
        nn_output = lipnet((pc_input,imu_input,imu_acc))
        loss = calc_loss_m(nn_output, (gt_joint,gt_pose),3)
        loss["total"].backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
        total_loss += loss["total"].item()
        joint_position_loss += loss["joint_loss"].item()
        pose_loss += loss["pose_loss"].item()
        fk_loss+=loss['fk_loss'].item()

        index+=1
    total_loss /= index
    limb_position_loss /= index
    joint_position_loss /= index
    pose_loss /= index
    fk_loss /= index
    return total_loss,joint_position_loss,pose_loss,fk_loss
    
def train(args,lipnet,trainDataLoader,testDataLoader,smpl,optimizer,scheduler):
    best_loss = 10000000
    device = args.device
    save_path = os.path.join(args.save_path,args.exp_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for epoch in range(args.start_epoch,args.start_epoch+args.epochs):
        s_time = time.time()
        train_loss,joint_position_loss,pose_loss,fk_loss = train_one_epoch(lipnet,device,trainDataLoader,optimizer)
        test_loss = test_one_epoch(lipnet,testDataLoader,device,smpl)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(lipnet.state_dict(), os.path.join(save_path,"est_best_model.t7"))

        e_time = time.time()
        cost_time = e_time-s_time
        print(f"Epoch {epoch} [train total:{train_loss:.5f}][test total:{test_loss:.5f}][best:{best_loss:.5f}]"
            f"[joint_position_loss: {joint_position_loss:.5f}][pose_loss: {pose_loss:.5f}][fk_loss: {fk_loss:.5f}][cose time: {cost_time:.5f}]")
        scheduler.step()
        # if epoch % 10 == 0:
        #     torch.save(lipnet.state_dict(), f"{save_path}/est_"+str(epoch)+".t7")

def options():
    parser = argparse.ArgumentParser(description='Baseline network')
    parser.add_argument('--save_path',type=str,default='./save_models/')
    parser.add_argument('--test_data',type=str,default='eRD')
    parser.add_argument('-n','--exp_name',type=str,default='LIP/')
    parser.add_argument('--root_dataset_path',type=str,default='path_to_root_dir')
    parser.add_argument('--train',type=int,default=1)
    parser.add_argument('--dis_result',type=str,default='./dis_results')
    parser.add_argument('--save_name',type=str,default='./est_results/LIPD/')

    # settings of input data
    parser.add_argument('--frames',type=int,default=32)
    parser.add_argument('--num_points',type=int,default=256)
    parser.add_argument('--imu_nums',type=int,default=4)
    parser.add_argument('--data',type=str,default='eRD')

    #settings of training
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--wd',type=float,default=1e-4)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--start_epoch',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=500)

    parser.add_argument('--pretrained',default=None)
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()
    return args

def main():
    args = options()
    device = args.device

    lipnet = Module_LINet(imu=args.imu_nums,module=3)
    if args.pretrained !=None:
        state_dict = torch.load(f'{args.pretrained}/est_model.t7',map_location='cpu')
        lipnet.load_state_dict(state_dict)
    lipnet.to(device)
    smpl = SMPL(SMPL_MODEL_DIR, create_transl=True).to(device)
    if args.train:
        trainingDataSet = Dataset_est(args,'train',False)
        trainDataLoader = torch.utils.data.DataLoader(trainingDataSet,batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=32,pin_memory=True)
        testDataset = Dataset_est(args,args.test_data,False)
        testDataLoader = torch.utils.data.DataLoader(testDataset,batch_size=args.batch_size,shuffle=False,drop_last=True,num_workers=32,pin_memory=True)

        optimizer = torch.optim.AdamW(lipnet.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        train(args,lipnet,trainDataLoader,testDataLoader,smpl,optimizer,scheduler)
    else:
        testDataset = Dataset_est(args,args.test_data,False)
        testDataLoader = torch.utils.data.DataLoader(testDataset,batch_size=args.batch_size,shuffle=False,drop_last=True,num_workers=32,pin_memory=True)
        test_one_epoch(lipnet,testDataLoader,device,smpl,args.save_name)

if __name__ == "__main__":
    main()