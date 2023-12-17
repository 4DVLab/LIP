import argparse
import torch
from models.LIP_model import trans_lipnet
from dataset.LIP_dataset import Dataset_ptc
from scipy.spatial.transform import Rotation as R
import time
from tqdm import tqdm
import os
from models.loss_utils import *
import sys
sys.path.append("./smpl")
from smpl import SMPL, SMPL_MODEL_DIR
from pytorch3d.loss import chamfer_distance

def gen_smpl(smpl,rot,device):
    pose_b = rot[:,3:].float()
    g_r = rot[:,:3].float()
    shape = torch.from_numpy(np.loadtxt('./smpl/shape.txt')).to(device).reshape(1,10)
    shape = shape.repeat([g_r.shape[0],1]).float()
    zeros = np.zeros((g_r.shape[0], 3))
    transl_blob = torch.from_numpy(zeros).float().to(device).float()
    mesh = smpl(betas=shape,body_pose=pose_b,global_orient = g_r,transl=transl_blob)
    v = mesh.vertices.reshape(-1,6890,3)
    j = mesh.joints.reshape(-1,24,3)
    return v,j

def test_one_epoch(lipnet,testDataLoader,device,smpl):
    lipnet.eval()
    loss_trans = 0.0
    index=0

    for i, batch in enumerate(tqdm(testDataLoader)):

        pc_input = batch['data'].to(device)
        # gt_trans = batch['trans'].to(device)

        pose = batch['pose'].to(device)
        gt_pose = batch['gt_pose'].to(device)
        joint = batch['joint'].to(device)

        root_output = lipnet((pc_input,pose,joint),device)
        v,_ = gen_smpl(smpl,gt_pose.float().reshape(-1,72),device)
        B,T,_ = root_output.shape
        final_pc = pc_input.reshape(B*T,256,3)-root_output.reshape(B*T,1,3)

        loss,_ = chamfer_distance(final_pc,v.reshape(B*T,-1,3))

        loss_trans+=loss.mean().item()
        index+=1

    print('--------------------')
    print('Trans:',loss_trans/index)
    return (loss_trans)/(index)

def train_one_epoch(lipnet,device,trainDataLoader,optimizer):

    lipnet.train()
    root_loss = 0.0

    index = 0
    for i, batch in enumerate(tqdm(trainDataLoader)):

        pc_input = batch['data'].to(device)
        gt_trans = batch['trans'].to(device)

        pose = batch['gt_pose'].to(device)
        joint = batch['gt_joint'].to(device)

        # optimize
        optimizer.zero_grad()
        root_output = lipnet((pc_input,pose,joint),device)
        B,T,_ = root_output.shape
        loss = euclidean_loss(root_output.view(B,T,-1,3),gt_trans.view(B,T,-1,3))
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        
        root_loss+=loss.item()

        index+=1
    root_loss = root_loss/index
    return root_loss
    
def train(args,lipnet,trainDataLoader,testDataLoader,smpl,optimizer,scheduler):
    best_loss = 10000000
    device = args.device
    save_path = args.save_path+args.exp_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for epoch in range(args.start_epoch,args.start_epoch+args.epochs):
        s_time = time.time()
        train_loss = train_one_epoch(lipnet,device,trainDataLoader,optimizer)
        test_loss = test_one_epoch(lipnet,testDataLoader,device,smpl)

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(lipnet.state_dict(), f"{save_path}/ptc_best_model.t7")

        e_time = time.time()
        cost_time = e_time-s_time
        print(f"Epoch {epoch} [train total:{train_loss:.5f}][test total:{test_loss:.5f}][best:{best_loss:.5f}][cose time: {cost_time:.5f}]")
        scheduler.step()
        # if epoch % 10 == 0:
        #     torch.save(lipnet.state_dict(), f"{save_path}/est_"+str(epoch)+".t7")

def options():
    parser = argparse.ArgumentParser(description='Baseline network')
    parser.add_argument('--save_path',type=str,default='./save_models/')
    parser.add_argument('-n','--exp_name',type=str,default='LIP/')
    parser.add_argument('--root_dataset_path',type=str,default='path_to_root_dir')
    parser.add_argument('--train',type=int,default=1)
    parser.add_argument('--est_result',type=str,default='./est_results/LIPD/est.pkl')

    # settings of input data
    parser.add_argument('--frames',type=int,default=32)
    parser.add_argument('--num_points',type=int,default=256)
    parser.add_argument('--imu_nums',type=int,default=4)

    #settings of training
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--wd',type=float,default=1e-4)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--start_epoch',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=2000)

    parser.add_argument('--pretrained',default=None)
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()
    return args

def main():
    args = options()
    device = args.device

    lipnet = trans_lipnet()
    if args.pretrained!=None:
        state_dict = torch.load(f'{args.pretrained}/ptc_model.t7',map_location='cpu')
        lipnet.load_state_dict(state_dict)
    lipnet.to(device)
    smpl = SMPL(SMPL_MODEL_DIR, create_transl=True).to(device)

    if args.train:
        trainingDataSet = Dataset_ptc(args,'t')
        trainDataLoader = torch.utils.data.DataLoader(trainingDataSet,batch_size=args.batch_size,shuffle=True,drop_last=True,num_workers=32,pin_memory=True)
        testDataset = Dataset_ptc(args,'e')
        testDataLoader = torch.utils.data.DataLoader(testDataset,batch_size=args.batch_size,shuffle=False,drop_last=True,num_workers=32,pin_memory=True)

        optimizer = torch.optim.AdamW(lipnet.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        train(args,lipnet,trainDataLoader,testDataLoader,smpl,optimizer,scheduler)
    else:
        testDataset = Dataset_ptc(args,'e')
        testDataLoader = torch.utils.data.DataLoader(testDataset,batch_size=args.batch_size,shuffle=False,drop_last=True,num_workers=32,pin_memory=True)
        test_one_epoch(lipnet,testDataLoader,device,smpl)

if __name__ == "__main__":
    main()