import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def local2global(pose):
    kin_chains = [
        [20, 18, 16, 13, 9, 6, 3, 0],   # left arm
        [21, 19, 17, 14, 9, 6, 3, 0],   # right arm
        [7, 4, 1, 0],                   # left leg
        [8, 5, 2, 0],                   # right leg
        [12, 9, 6, 3, 0],               # head
        [0],                            # root, hip
    ]
    T = pose.shape[0]
    Rb2l = []
    cache = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    for chain in kin_chains:
        leaf_rotmat = torch.eye(3).unsqueeze(0).repeat(T,1,1).to(pose.device).float()
        for joint in chain:
            joint_rotvec = pose[:, joint*3:joint*3+3]
            joint_rotmat = axis_angle_to_matrix(joint_rotvec).float()
            # joint_rotmat = torch.from_numpy(R.from_rotvec(joint_rotvec).as_matrix().astype(np.float32))
            leaf_rotmat = torch.einsum("bmn,bnl->bml", joint_rotmat, leaf_rotmat)
            cache[joint] = leaf_rotmat
        Rb2l.append(cache)
    return cache

def cal_mpjpe(pred, gt):
    gt = gt.reshape(gt.shape[0], -1, 3)[:pred.shape[0]]
    distance = np.linalg.norm(pred-gt, axis=2)
    return distance.mean(axis=1).mean()

def eval(gt_pose,pose,smpl):
    pose_data = pose
    pose_blob = pose_data[:, 3:].float()
    globalR = pose_data[:, :3].float()

    gt_data = gt_pose
    gt_pose_blob = (gt_data[:pose_data.shape[0], 3:]).float()
    gt_globalR = (gt_data[:pose_data.shape[0], :3]).float()

    shape_blob = torch.from_numpy(np.loadtxt("../smpl/shape.txt")).to(gt_pose.device)
    shape_blob1 = shape_blob[None, :].repeat([globalR.shape[0], 1]).float()
    shape_blob2 = shape_blob[None, :].repeat([gt_globalR.shape[0], 1]).float()

    zeros = np.zeros((globalR.shape[0], 3))
    transl_blob = torch.from_numpy(zeros).to(gt_pose.device).float()

    output = smpl(betas=shape_blob1, body_pose=pose_blob, global_orient=globalR, transl=transl_blob)
    gt = smpl(betas=shape_blob2, body_pose=gt_pose_blob, global_orient=gt_globalR, transl=transl_blob)

    joints = gt.joints.reshape(-1,24,3)
    joints_pose = output.joints
    mpjpe = cal_mpjpe(joints_pose.cpu().detach().numpy().reshape((-1,24,3)),joints.cpu().detach().numpy())

    return mpjpe

def eval_v(gt_pose,pose,smpl):
    pose_data = pose

    pose_blob = pose_data[:, 3:].float()
    globalR = pose_data[:, :3].float()

    gt_data = gt_pose
    gt_pose_blob = (gt_data[:pose_data.shape[0], 3:]).float()
    gt_globalR = (gt_data[:pose_data.shape[0], :3]).float()

    shape_blob = torch.from_numpy(np.loadtxt("./smpl/shape.txt")).to(gt_pose.device)
    shape_blob1 = shape_blob[None, :].repeat([globalR.shape[0], 1]).float()
    shape_blob2 = shape_blob[None, :].repeat([gt_globalR.shape[0], 1]).float()

    zeros = np.zeros((globalR.shape[0], 3))
    transl_blob = torch.from_numpy(zeros).to(gt_pose.device).float()

    output = smpl(betas=shape_blob1, body_pose=pose_blob, global_orient=globalR, transl=transl_blob)
    gt = smpl(betas=shape_blob2, body_pose=gt_pose_blob, global_orient=gt_globalR, transl=transl_blob)

    joints_pose = output.vertices
    joints = gt.vertices
    mpvpe = cal_mpjpe(joints_pose.cpu().detach().numpy().reshape((globalR.shape[0],-1,3)),joints.cpu().detach().numpy().reshape((globalR.shape[0],-1,3)))

    gt_matrix = local2global(gt_pose.reshape(-1,72))
    pose_matrix = local2global(pose.reshape(-1,72))

    gt_matrix = torch.from_numpy(np.array([item.cpu().detach().numpy() for item in gt_matrix if item!=None])).reshape(-1,3,3)
    pose_matrix = torch.from_numpy(np.array([item.cpu().detach().numpy() for item in pose_matrix if item!=None])).reshape(-1,3,3)
    gt_axis = quaternion_to_axis_angle(matrix_to_quaternion(gt_matrix))
    pose_axis = quaternion_to_axis_angle(matrix_to_quaternion(pose_matrix))

    gt_norm = np.rad2deg(np.linalg.norm(gt_axis.numpy(),axis=1)).reshape((globalR.shape[0],-1,1))
    pose_norm = np.rad2deg(np.linalg.norm(pose_axis.numpy(),axis=1)).reshape((globalR.shape[0],-1,1))
    anger = np.abs((gt_norm-pose_norm)).mean(axis=1).mean()
    return mpvpe,anger

def mse_loss(batch_pred, batch, batch_weight=None):
    residual = batch_pred - batch #(...,3, 3) (...,3, 3)
    loss = residual.pow(2).sum(dim=-1)
    if batch_weight is not None:
        loss = loss * batch_weight
    loss = loss.mean()
    return loss

def euclidean_loss(batch_pred, batch, batch_weight=None):
    residual = batch_pred - batch #(...,3, 3) (...,3, 3)
    loss = residual.norm(dim=3)
    if batch_weight is not None:
        loss = loss * batch_weight
    loss = loss.mean()
    return loss

def pose_diff(output, pose_gt):
    pose = output["body_pose"]
    B, T, _ = pose.shape
    net_pose = pose.view(B,T,-1,6)
    # gt_pose = torch.from_numpy(R.from_rotvec(pose_gt.view(-1,3).cpu().detach().numpy()).as_matrix()).to(pose.device).view(B,T,-1,3,3)
    gt_pose = matrix_to_rotation_6d(axis_angle_to_matrix(pose_gt.view(B*T,-1,3))).view(B,T,-1,6)
    return mse_loss(net_pose, gt_pose)

def fk_distance(output, joint_gt):
    def fk(pose):
        kin_chains = [
            [20, 18, 16, 13, 9, 6, 3, 0],   # left arm
            [21, 19, 17, 14, 9, 6, 3, 0],   # right arm
            [7, 4, 1, 0],                   # left leg
            [8, 5, 2, 0],                   # right leg
            [12, 9, 6, 3, 0],               # head
            [0],                            # root, hip
        ]

        skeleton = [
            [torch.tensor([ 0.2414698 ,  0.00888374, -0.00123079]), torch.tensor([ 0.25392494, -0.01296091, -0.02624109]), torch.tensor([ 0.08503542,  0.03124243, -0.00682255]), torch.tensor([ 0.07392273,  0.11611712, -0.03430349]), torch.tensor([0.00177833, 0.04944363, 0.02488606]), torch.tensor([0.00432832, 0.12895165, 0.00087818]), torch.tensor([-0.00251997,  0.10273486, -0.02064193]), torch.tensor([0, 0, 0])],
            [torch.tensor([-0.24873409,  0.00830102, -0.00497455]), torch.tensor([-0.24776915, -0.01408127, -0.01928832]), torch.tensor([-0.08927801,  0.03277531, -0.00855309]), torch.tensor([-0.07627045,  0.11371064, -0.03883788]), torch.tensor([0.00177833, 0.04944363, 0.02488606]), torch.tensor([0.00432832, 0.12895165, 0.00087818]), torch.tensor([-0.00251997,  0.10273486, -0.02064193]), torch.tensor([0, 0, 0])],
            [torch.tensor([-0.01209266, -0.38906169, -0.04199572]), torch.tensor([ 0.03373346, -0.37117104, -0.00400516]), torch.tensor([ 0.0680889 , -0.08839001, -0.00908553]), torch.tensor([0, 0, 0])],
            [torch.tensor([ 0.01457455, -0.38898903, -0.04038934]), torch.tensor([-0.0383419 , -0.37704535, -0.00744271]), torch.tensor([-0.06591475, -0.08762817, -0.00634244]), torch.tensor([0, 0, 0])],
            [torch.tensor([-0.00255715,  0.20534551, -0.04346146]), torch.tensor([0.00177833, 0.04944363, 0.02488606]), torch.tensor([0.00432832, 0.12895165, 0.00087818]), torch.tensor([-0.00251997,  0.10273486, -0.02064193]), torch.tensor([0, 0, 0])],
            [torch.tensor([0, 0, 0])]
        ]

        B,T,_ = pose.shape
        SE3_cache = torch.zeros((B,T,24,4,4)).to(pose.device)
        check_list = []
        for i in range(len(kin_chains)):
            chain = kin_chains[i]
            leaf_SE3 = torch.eye(4).view(1,1,4,4).repeat((B,T,1,1)).to(pose.device)
            for j in range(len(chain)-1,-1,-1):
                joint = chain[j]
                if joint in check_list:
                    leaf_SE3 = SE3_cache[:,:,joint,:,:].clone()
                    continue
                joint_mat = rotation_6d_to_matrix(pose[:,:,joint*6:joint*6+6])
                joint_SE3 = torch.eye(4).view(1,1,4,4).repeat((B,T,1,1)).to(pose.device)
                joint_SE3[:,:,:3,:3] = joint_mat
                joint_SE3[:,:,:3,3] = skeleton[i][j].view(1,1,3).repeat((B,T,1)).to(pose.device)
                leaf_SE3 = torch.einsum("btmn,btnl->btml", leaf_SE3, joint_SE3)
                SE3_cache[:,:,joint,:,:] = leaf_SE3.clone()
                check_list.append(joint)
        return SE3_cache[:,:,:,:3,3].view(B,T,24,3)

    B, T, _ = joint_gt.shape
    pose = output["body_pose"]
    op_21_joints = [20,21,7,8,18,19,4,5,16,17,1,0,2,12]
    s = [0,1,2,3,4,5,6,7,8,9,10,11,12,14]
    skeleton = [0,1,2,3,4,5,6,7,8,9,12,13,14,16,17,18,19,20,21]
    fk_joints = fk(pose)[:,:,skeleton,:]
    return euclidean_loss(fk_joints, joint_gt.view(B,T,-1,3)[:,:,skeleton,:])

def calc_loss_m(output, gt,m):
    joint_position_gt, pose_gt = gt#, transl_gt = gt
    pose_loss = pose_diff(output, pose_gt)
    fk_loss = fk_distance(output, joint_position_gt)
    loss = {}
    loss["pose_loss"] = pose_loss
    loss['fk_loss'] = fk_loss
    loss['total'] = pose_loss+fk_loss
    #co_limb, co_joint, co_pose = 0.1,0.2,0.7
    co_joint, co_pose,co_fk = 0.2,0.4,0.4
    B,T = pose_gt.shape[0],pose_gt.shape[1]
    if m == 2:
        co_all = co_joint+co_pose
        limb_position = output["limb_position"]
        limb_position_loss = euclidean_loss(limb_position.view(B,T,-1,3), joint_position_gt.view(B,T,-1,3)[:, :, [20,21,7,8,18,19,4,5]].view(B,T,-1,3))
        loss["limb_loss"] = limb_position_loss
        loss["total"] = (co_joint/co_all)*limb_position_loss + (co_pose/co_all)*(pose_loss+fk_loss)
    if m == 3:
        co_all = co_joint+co_pose+co_fk
        joint_position = output['joint_position']
        joint_position_loss = euclidean_loss(joint_position.view(B,T,-1,3), joint_position_gt.view(B,T,-1,3))
        loss["joint_loss"] = joint_position_loss
        loss["total"] = (co_joint/co_all)*joint_position_loss + (co_pose/co_all)*(pose_loss)+(co_fk/co_all)*(fk_loss)
    return loss
