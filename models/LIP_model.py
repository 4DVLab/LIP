import torch.nn as nn
import torch
import torch.utils.data
import torch.nn.functional as F
from .model_utils import PointNetEncoder,Tracker
from torch.autograd import Variable

class PP_distil(nn.Module):
    def __init__(self,k=36, channel=3):
        super(PP_distil, self).__init__()
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.gru = nn.GRU(1024,512,bidirectional=True,batch_first=True)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def encode(self, x):
        x, _, _ = self.feat(x)
        return x
    
    def forward(self,x,device):
        xyz = x.permute(0,1,3,2)
        B,T,N,C = xyz.shape
        x = self.encode(xyz.reshape(B*T,N,C))
        x = x.reshape(B,T,-1)
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(2, x.size(0), 512)).to(device)
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        out,hid = self.gru(x,h0)
        x = F.relu((self.fc1(out)))
        x = F.relu((self.dropout(self.fc2(x))))
        x = self.fc3(x).view(B,T,-1)
        return x.view(B,T,-1),out.reshape(B,T,1024)

class Module_LINet(nn.Module):
    def __init__(
        self,
        imu=4,
        hidden_size=1024,
        module = 1,
    ):
        super(Module_LINet, self).__init__()

        self.m = module
        if module==1:
            self.hybrik = Tracker(
                input_size=imu*12+78,
                hidden_size=hidden_size,
                output_size=24*6,
            )
        if module==2:
            self.limb_tracker = Tracker(
            input_size=imu*12+78,
            hidden_size=hidden_size,
            output_size=24,
        )
            self.hybrik = Tracker(
                input_size=imu*12+24+78,
                hidden_size=hidden_size,
                output_size=24*6,
        )

        if module==3:
            self.joint_tracker = Tracker(
            input_size=imu*12+78,
            hidden_size=hidden_size,
            output_size=72,
        )
            self.hybrik = Tracker(
                input_size=imu*12+72+78,
                hidden_size=hidden_size,
                output_size=24*6,
        )

    def forward(self, input):
        pc_input, imu_input, imu_acc = input     # pc_input.shape: (Batch, Temporal, N, 3)
        #pc_input = self.pointnet(pc_input, device)    # pc_feature.shape: (Batch, Temporal, 1024)

        leftForeArm_imu, rightForeArm_imu,leftLeg_imu, rightLeg_imu,_,_ = imu_input.permute(2,0,1,3)
        leftForeArm_acc, rightForeArm_acc,leftLeg_acc, rightLeg_acc,_,_ = imu_acc.permute(2,0,1,3)
        B, T, _ = leftLeg_imu.shape
        imu_input_cat = torch.cat(  # shape: (Batch, Temporal, 4*9)
            [
            leftLeg_imu.view(B, T, 9),
            rightLeg_imu.view(B, T, 9),
            leftForeArm_imu.view(B, T, 9),
            rightForeArm_imu.view(B, T, 9),
            leftLeg_acc.view(B, T, 3),
            rightLeg_acc.view(B, T, 3),
            leftForeArm_acc.view(B, T, 3),
            rightForeArm_acc.view(B, T, 3),
            ]
            , axis=2
        )
        Tracker_input = torch.cat([pc_input, imu_input_cat], axis=2)
        if self.m == 1:
            body_pose = self.hybrik(Tracker_input)
            return {'body_pose' : body_pose}
        if self.m == 2:
            limb_position = self.limb_tracker(Tracker_input)
            hybriIK_input = torch.cat([Tracker_input, limb_position], 2)
            body_pose = self.hybrik(hybriIK_input)
            return {
            "limb_position": limb_position,
            "body_pose": body_pose
            }
        if self.m == 3:
            joint_position = self.joint_tracker(Tracker_input)
            hybriIK_input = torch.cat([Tracker_input, joint_position], 2)
            body_pose = self.hybrik(hybriIK_input)
            return {
                "joint_position": joint_position,
                "body_pose": body_pose
            }

class LIPNet_multiIMU(nn.Module):
    def __init__(
        self,
        imu=4,
        hidden_size=1024,
    ):
        super(LIPNet_multiIMU, self).__init__()

        self.joint_tracker = Tracker(
            input_size=imu*12+78,
            hidden_size=hidden_size,
            output_size=72,
        )

        self.hybrik = Tracker(
            input_size=imu*12+72+78,
            hidden_size=hidden_size,
            output_size=24*6,
        )

    def forward(self,input,imu_num=4):
        pc_input,imu_input, imu_acc = input 

        if imu_num==12:
            B,T=imu_input.shape[0],imu_input.shape[1]
            imu_input_cat = torch.cat(  # shape: (Batch, Temporal, 12*9)
                [
                imu_input.view(B,T,-1),
                imu_acc.view(B,T,-1),
                ]
                , axis=2
            )
        else:
            leftForeArm_imu, rightForeArm_imu,leftLeg_imu, rightLeg_imu,_,root_imu = imu_input.permute(2,0,1,3)
            leftForeArm_acc, rightForeArm_acc,leftLeg_acc, rightLeg_acc,_,root_acc = imu_acc.permute(2,0,1,3)
            B, T, _ = leftLeg_imu.shape

        if imu_num==2:
            imu_input_cat = torch.cat(  # shape: (Batch, Temporal, 2*9)
                [
                rightLeg_imu.view(B, T, 9),
                leftForeArm_imu.view(B, T, 9),
                rightLeg_acc.view(B, T, 3),
                leftForeArm_acc.view(B, T, 3),
                ]
                , axis=2
            )

        if imu_num==4:
            imu_input_cat = torch.cat(  # shape: (Batch, Temporal, 4*9)
                [
                leftLeg_imu.view(B, T, 9),
                rightLeg_imu.view(B, T, 9),
                leftForeArm_imu.view(B, T, 9),
                rightForeArm_imu.view(B, T, 9),
                leftLeg_acc.view(B, T, 3),
                rightLeg_acc.view(B, T, 3),
                leftForeArm_acc.view(B, T, 3),
                rightForeArm_acc.view(B, T, 3),
                ]
                , axis=2
            )
        if imu_num==5:
            imu_input_cat = torch.cat(  # shape: (Batch, Temporal, 5*9)
                [
                leftLeg_imu.view(B, T, 9),
                rightLeg_imu.view(B, T, 9),
                leftForeArm_imu.view(B, T, 9),
                rightForeArm_imu.view(B, T, 9),
                root_imu.view(B,T,9),
                leftLeg_acc.view(B, T, 3),
                rightLeg_acc.view(B, T, 3),
                leftForeArm_acc.view(B, T, 3),
                rightForeArm_acc.view(B, T, 3),
                root_acc.view(B,T,3),
                ]
                , axis=2
            )

        jointTracker_input = torch.cat([pc_input, imu_input_cat], axis=2)  # shape: (Batch, Temporal, 78+4*9)
        joint_position = self.joint_tracker(jointTracker_input)

        hybriIK_input = torch.cat([jointTracker_input, joint_position], 2)
        body_pose = self.hybrik(hybriIK_input)

        return {
            "joint_position": joint_position,
            "body_pose": body_pose
        }

class trans_lipnet(nn.Module):
    def __init__(
        self,
        hidden_size=1024,
    ):
        super(trans_lipnet, self).__init__()
        self.pointnet = PP_distil(k=69)

        self.hybrik = Tracker(
            input_size=1024+72+72,
            hidden_size=hidden_size,
            output_size=3,
        )

    def forward(self, input, device):
        pc_input, pose, joints= input     # pc_input.shape: (Batch, Temporal, N, 3)
        _,pc_feature = self.pointnet(pc_input, device)    # pc_feature.shape: (Batch, Temporal, 1024)
        hybriIK_input = torch.cat([pc_feature, pose,joints], 2)
        root_joint = self.hybrik(hybriIK_input)

        return root_joint
