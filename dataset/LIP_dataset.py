import torch
import pickle
import numpy as np
import os

def farthest_point_sample(xyz, npoint):
    ndataset = xyz.shape[0]
    if ndataset<npoint:
        repeat_n = int(npoint/ndataset)
        xyz = np.tile(xyz,(repeat_n,1))
        xyz = np.append(xyz,xyz[:npoint%ndataset],axis=0)
        return xyz
    centroids = np.zeros(npoint)
    distance = np.ones(ndataset) * 1e10
    farthest =  np.random.randint(0, ndataset)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[int(farthest)]
        dist = np.sum((xyz - centroid) ** 2, 1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return xyz[np.int32(centroids)]

class Dataset_dis(torch.utils.data.Dataset):
    
    def __init__(self,args,m):
        self.dataset = []
        if m == 'e':
            data_info_path = os.path.join(args.root_dataset_path,args.test_data)#'LIPD_test.pkl'
        else:
            data_info_path = os.path.join(args.root_dataset_path,args.train_data)

        self.num_points = args.num_points
        T = args.frames
        file = open(data_info_path,'rb')
        datas = pickle.load(file)
        file.close()
        old_motion_id = datas[0]['seq_path']

        seq = []
        j=0
        if T == 1:
            self.dataset = datas
        else:
            while True:
                if j >=len(datas):
                    break
                motion_id = datas[j]['seq_path']
                if motion_id==old_motion_id:
                    seq.append(datas[j])
                    j+=1
                else:
                    old_motion_id = motion_id
                    seq=[datas[j]]
                    j+=1
                if len(seq) == T:
                    self.dataset.append(seq)
                    seq=[]
    
    def __getitem__(self, index):
        example_seq = self.dataset[index]
        seq_pc = []
        seq_gt = []
        seq_j = []
        for example in example_seq:
            pc_data = example['pc']
            if type(pc_data)==str:
                pc_data = np.fromfile(pc_data,dtype=np.float32).reshape(-1,3)
            if len(pc_data)==0:
                pc_data = np.array([[0,0,0]])
            pc_data = pc_data - pc_data.mean(0)
            pc_data = farthest_point_sample(pc_data,self.num_points)
            seq_pc.append(pc_data)
            gt_r = example['gt_r']
            gt_j = example['gt_joint'].reshape(-1)
            seq_j.append(gt_j)
            gt = np.concatenate((gt_r,gt_j))
            seq_gt.append(gt)

        Item = {
            'data': np.array(seq_pc),
            'gt' : np.array(seq_gt),
            'gt_j': np.array(seq_j),
        }
        return Item

    def __len__(self):
        return len(self.dataset)
    

class Dataset_est(torch.utils.data.Dataset):
    def __init__(self,args,module,pc=True):
        self.dataset = []
        self.pc = pc
        root_dataset_path = args.root_dataset_path
        dis_result = args.dis_result
        m = module
        if m == 'eDIP':
            data_info_path = root_dataset_path+'DIP_test.pkl'
            self.pc_data = np.load(os.path.join(dis_result,"DIP","dis.pkl"),allow_pickle=True)
        elif m=='eTC':
            data_info_path = root_dataset_path+'TC_test.pkl'
            self.pc_data = np.load(os.path.join(dis_result,"TC","dis.pkl"),allow_pickle=True)
        elif m =='eLIPD':
            data_info_path = root_dataset_path+'LIPD_test.pkl'
            self.pc_data = np.load(os.path.join(dis_result,"LIPD","dis.pkl"),allow_pickle=True)
        elif m == 'eLH':
            data_info_path = root_dataset_path+'Test_lidarhuman.pkl'
            self.pc_data = np.load(os.path.join(dis_result,"LH","dis.pkl"),allow_pickle=True)
        else:
            data_info_path =root_dataset_path+'LIPD_train.pkl'
            self.pc_data = np.load(os.path.join(dis_result,"LIPD_Train","dis.pkl"),allow_pickle=True)

        self.pc_data = np.concatenate(self.pc_data).reshape(-1,32,78)
        self.num_points = args.num_points
        T = args.frames

        file = open(data_info_path,'rb')
        datas = pickle.load(file)
        file.close()
        old_motion_id = datas[0]['seq_path']  

        seq = []
        j=0
        if T == 1:
            self.dataset = [datas]
        else:
            while True:           
                if j >=len(datas):
                    break
                motion_id = datas[j]['seq_path']
                if motion_id == old_motion_id:
                    seq.append(datas[j])
                    j+=1
                else:
                    old_motion_id = motion_id
                    seq=[datas[j]]
                    j+=1
                if len(seq) == T:
                    self.dataset.append(seq)
                    seq=[]

    def __getitem__(self, index):
        example_seq = self.dataset[index]
        pc_data = np.array(self.pc_data[index]) # 32 x 78
        seq_pc = []
        seq_gt = []
        seq_j = []
        seq_imu_ori = []
        seq_imu_acc = []
        for example in example_seq:
            if self.pc:
                pc_path = example['pc']
                if len(pc_data)==0:
                    print(pc_path)
                pc_data = pc_data - pc_data.mean(0)
                pc_data = farthest_point_sample(pc_data,self.num_points)
                seq_pc.append(pc_data)
        
            gt = example['gt']
            gt_j = example['gt_joint']
            seq_imu_ori.append(example['imu_ori'].reshape([-1,9]))
            seq_imu_acc.append(example['imu_acc'].reshape([-1,3]))
            seq_j.append(gt_j)
            seq_gt.append(gt)

        Item = {
            'data':np.array(pc_data),
            'gt_pose' : np.array(seq_gt),
            'gt_j': np.array(seq_j),
            'id':example['seq_path'],
            'imu_ori':np.array(seq_imu_ori),
            'imu_acc':np.array(seq_imu_acc),
        }
        if self.pc:
            Item['data'] = np.array(seq_pc)
        return Item
    
    def __len__(self):
        return len(self.pc_data)

class Dataset_ptc(torch.utils.data.Dataset):
    def __init__(self,args,m):
        self.dataset = []
        root_dataset_path = args.root_dataset_path
        self.m = m
        est_result = args.est_result
        if m == 'e':
            data_info_path = os.path.join(root_dataset_path,'LIPD_test.pkl')
            self.pose_path = est_result
            self.pose_data = pickle.load(open(self.pose_path,'rb'))
        else:
            data_info_path = root_dataset_path+'Trans_train.pkl'
  
        self.num_points = args.num_points
        T = args.frames

        file = open(data_info_path,'rb')
        datas = pickle.load(file)
        file.close()
        old_motion_id = datas[0]['seq_path']
        seq = []
        j=0
        if T == 1:
            self.dataset = [datas]
        else:
            while True:
                if j >=len(datas):
                    break
                motion_id = datas[j]['seq_path']
                if motion_id==old_motion_id:
                    seq.append(datas[j])
                    j+=1
                else:
                    old_motion_id = motion_id
                    seq=[datas[j]]
                    j+=1
                if len(seq) == T:
                    self.dataset.append(seq)
                    seq=[]
    
    def __getitem__(self, index):
        example_seq = self.dataset[index]
        seq_pc = []
        seq_gtp = []
        seq_gtj = []
        seq_pose = []
        seq_j = []
        seq_norm = []
        seq_trans = []
        for example in example_seq:
            pc_data = example['pc']
            if type(pc_data)==str:
                pc_data = np.fromfile(pc_data,dtype=np.float32).reshape(-1,3)
            if len(pc_data)==0:
                pc_data=np.array([[0,0,0]])
            norm_trans = pc_data.mean(0)
            pc_data = pc_data-norm_trans
            pc_data = farthest_point_sample(pc_data,self.num_points)
            seq_pc.append(pc_data)

            gt = example['gt']
            gt_j = example['gt_joint']
            seq_gtj.append(gt_j)
            seq_gtp.append(gt)
            if self.m == 'e':
                seq_pose = self.pose_data['pose'][index]
                seq_j = self.pose_data['joint'][index]
            else:
                seq_trans.append(example['trans'])
            seq_norm.append(norm_trans)
            
        Item = {
            'pose' : np.array(seq_pose),
            'joint' : np.array(seq_j),
            'gt_pose': np.array(seq_gtp),
            'gt_joint': np.array(seq_gtj),
            'id':example['seq_path'],
            'norm':np.array(seq_norm),
            'data':np.array(seq_pc),
        }
        if self.m != 'e':
            Item['trans']:np.array(seq_trans)
        return Item
    
    def __len__(self):
        return len(self.dataset)