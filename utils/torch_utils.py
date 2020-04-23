import torch
import math
from torchvision.ops import nms
def to_gpu(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    return tuple([t.cuda() for t in x])



def gaussian(size, std=0.5):
    y, x = torch.meshgrid([torch.linspace(0, 1, steps=size[0]), torch.linspace(0, 1, steps=size[1])])
    x = 2 * (x - 0.5)
    y = 2 * (y - 0.5)
    g = (x * x + y * y) / (2 * std * std)
    g = torch.exp(-g)
    g = g / (std * math.sqrt(2 * math.pi))
    return g

def gaussian2(size, center=None, std=0.5):
    if center is None:
        center = torch.tensor([[0.5, 0.5]])
    
    y, x = torch.meshgrid([torch.linspace(0, 1, steps=size[0]), torch.linspace(0, 1, steps=size[1])])
    # print(x.unsqueeze(0).shape, .shape)
    x = 2 * (x.unsqueeze(0) - center[:,0,None,None])
    y = 2 * (y.unsqueeze(0) - center[:,1,None,None])
    g = (x * x + y * y) / (2 * std * std)
    g = torch.exp(-g)
    return g


def circle_mask(size, center=None, radius=0.5):
    if center is None:
        center = torch.tensor([[0.5, 0.5]])
    
    y, x = torch.meshgrid([torch.linspace(0, 1, steps=size[0]), torch.linspace(0, 1, steps=size[1])])
    # print(x.unsqueeze(0).shape, .shape)
    x = 2 * (x.unsqueeze(0) - center[:,0,None,None])
    y = 2 * (y.unsqueeze(0) - center[:,1,None,None])
    d = (x * x + y * y) < (radius * radius)
    return d.float()

def half_mask(shape):
    angle = torch.rand((shape[0], 1, 1)) * math.pi * 2
    y, x = torch.meshgrid([torch.linspace(-1, 1, steps=shape[-2]), torch.linspace(-1, 1, steps=shape[-1])])
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    nx = torch.cos(angle)
    d = x * torch.cos(angle) + y * torch.sin(angle)
    mask = (d > 0).float()
    return mask

def square_mask(shape, size = 0.5):
    y, x = torch.meshgrid([torch.linspace(-1, 1, steps=shape[-2]), torch.linspace(-1, 1, steps=shape[-1])])
    d = torch.max(torch.abs(x), torch.abs(y))
    mask = (d < size).float()
    return mask

def calc_center_of_mass(heatmap,kernel_size):
    heatmap_exp = torch.exp(heatmap)
    heatmap_unf = torch.nn.functional.unfold(heatmap_exp, (kernel_size, kernel_size),padding = kernel_size//2).transpose(1,2)
    w_x = (torch.arange(kernel_size)-kernel_size//2).unsqueeze(0).expand(kernel_size,-1).reshape(-1,1).float()
    w_y = (torch.arange(kernel_size)-kernel_size//2).unsqueeze(1).expand(-1,kernel_size).reshape(-1,1).float()
    w_s = torch.ones(kernel_size,kernel_size).reshape(-1,1).float()
    heatmap_unf_x = heatmap_unf.matmul(w_x)
    heatmap_unf_y = heatmap_unf.matmul(w_y)
    heatmap_unf_s = heatmap_unf.matmul(w_s)
    offset_unf = torch.cat([heatmap_unf_x/heatmap_unf_s, heatmap_unf_y/heatmap_unf_s],dim=-1).transpose(1, 2)
    offset = torch.nn.functional.fold(offset_unf, (heatmap.shape[2], heatmap.shape[3]), (1, 1))
    grid_x = torch.arange(heatmap.shape[3]).unsqueeze(0).expand(heatmap.shape[2],-1).float()
    grid_y = torch.arange(heatmap.shape[2]).unsqueeze(1).expand(-1,heatmap.shape[3]).float()
    grid_xy = torch.cat([grid_x.unsqueeze(0),grid_y.unsqueeze(0)],dim=0)
    center = grid_xy+offset
    return center

def inverse_heatmap_int(keypoints, out_shape):
    # out_shape (B, 1, H, W)
    # TODO: 2x2 discretization
    # TODO: multi-scale
    keypoints = keypoints.int()
    heatmap = torch.zeros(out_shape)
    batch = torch.arange(keypoints.shape[0]).repeat(keypoints.shape[1], 1).permute(1, 0)
    x = keypoints[:, :, 0]
    y = keypoints[:, :, 1]
    x = torch.clamp((x + 0.5).long(), 0, out_shape[-1] - 1)
    y = torch.clamp((y + 0.5).long(), 0, out_shape[-2] - 1)
    heatmap[batch, 0, y, x] = 1.0
    return heatmap

def inverse_heatmap(bboxes, out_shape):
    # out_shape (B, 1, H, W)
    # TODO: 2x2 discretization
    # TODO: multi-scale
    keypoints = bboxes[:,:,:2]
    # Clamp keypoints within image size
    keypoints[:,:,0] = torch.clamp(keypoints[:,:,0], 0, out_shape[-1] - 1.001)
    keypoints[:,:,1] = torch.clamp(keypoints[:,:,1], 0, out_shape[-2] - 1.001)
    # Generate scatter index 
    seed = torch.tensor([0,1])   
    seq_1 = seed.unsqueeze(0).repeat(2,1).transpose(0,1).reshape(-1,1)    
    seq_2 = seed.unsqueeze(0).repeat(2,1).reshape(-1,1)    
    seq = torch.cat((seq_1,seq_2),1)
    scatter_index = keypoints[:,:,[0,1]].floor().unsqueeze(2).long()
    scatter_index = scatter_index + seq  
    # Generate scatter value 
    scatter_value = 1-torch.abs(keypoints[:,:,[0,1]].unsqueeze(2).repeat(1,1,4,1)-scatter_index.float()) 
    scatter_value = scatter_value[:,:,:,0] * scatter_value[:,:,:,1]
    # Fill scatter values to heatmaps
    heatmaps = torch.zeros(out_shape).repeat(1, scatter_value.shape[1], 1, 1)
    batch = torch.arange(scatter_value.shape[0]).repeat(scatter_value.shape[1], 1).permute(1, 0).unsqueeze(2)   
    chanel = torch.arange(scatter_value.shape[1]).repeat(scatter_value.shape[2], 1).permute(1, 0)
    x = scatter_index[:, :, :, 0]
    y = scatter_index[:, :, :, 1]
    heatmaps[batch, chanel, y, x] = scatter_value
    # sumover heatmaps
    heatmap = torch.sum(heatmaps,dim=1,keepdim=True)

    return heatmap

def inverse_heatmap_gaussian(bboxes, out_shape, var_scale=0.125):

    # bboxes[0]=torch.tensor([[1.0,1.0,1.0,1.0],[4.0,1.0,1.0,1.0],[7.0,4.0,2.0,2.0],[10.0,14.0,1.0,5.0],[1.0,14.0,5.0,1.0]])  

    # out_shape (B, 1, H, W)
    # TODO: 2x2 discretization
    # TODO: multi-scale
    # Clamp keypoints within image size
    B, _, H, W = out_shape
    bboxes = torch.clamp(bboxes, 0.000001)
    index_x = torch.arange(W).repeat(H)  
    index_y = torch.arange(H).unsqueeze(-1).repeat(1,W).reshape(-1) 
    index = torch.cat((index_x.unsqueeze(-1),index_y.unsqueeze(-1)),dim=-1).float()
    exp_term = torch.matmul(torch.pow(((bboxes[:,:,:2].unsqueeze(-2)-index)/(bboxes[:,:,2:]*var_scale).unsqueeze(-2)),2),torch.tensor([[0.5],[0.5]])).squeeze()
    norm = torch.exp(-exp_term)/(bboxes[:,:,[2]]*var_scale*bboxes[:,:,[3]]*var_scale)/2/math.pi
    heatmap = torch.sum(norm,dim=1).reshape(out_shape)

    return heatmap

def construct_dist_mat(kp_1, kp_2):
    # distance square matrix between two sets of points 
    # kp_1, kp_2 [B,N,2]
    xy_1_sq_sum_vec = torch.matmul(kp_1**2,torch.ones(2,1)) 
    xy_2_sq_sum_vec = torch.matmul(kp_2**2,torch.ones(2,1))
    # row: kp_1 column: kp_2
    xy_12_sq_sum_mat =  xy_1_sq_sum_vec + xy_2_sq_sum_vec.transpose(2,1)
    xy_mat = torch.matmul(kp_1, kp_2.transpose(2,1))
    dist_mat = xy_12_sq_sum_mat - 2*xy_mat
    dist_mat = torch.max(dist_mat,torch.zeros_like(dist_mat))  
    return dist_mat

def nms_detector(score, proposals, det_th, iou_th):

    proposals[:,2] = proposals[:,2] + proposals[:,0] 
    proposals[:,3] = proposals[:,3] + proposals[:,1] 
    detect = torch.empty([0,6])
    for cls in range(1,score.shape[1]):
        proposal_scores = torch.cat((proposals,score[:,[cls]]),dim=1)
        proposal_scores = proposal_scores[proposal_scores[:,4]>det_th,:]   
        # if proposal_scores.shape[0]>4:
        #     import IPython
        #     IPython.embed()
        #     assert(0)            
        nms_idx = nms(proposal_scores[:,:4],proposal_scores[:,4],iou_th)
        tmp_detect = torch.cat((proposal_scores[nms_idx,:],torch.ones([nms_idx.shape[0],1])*cls),dim=1)  
        tmp_detect[:,2] = tmp_detect[:,2] - tmp_detect[:,0] 
        tmp_detect[:,3] = tmp_detect[:,3] - tmp_detect[:,1] 
        detect = torch.cat((detect,tmp_detect),dim=0) 
    return detect


if __name__ == '__main__':
    eps = 1e-5
    # test center of mass
    print("Testing clac_center_of_mass ...")
    heatmap = heatmap = torch.randn(32, 1, 5, 5)
    kernel_size = 3
    center = calc_center_of_mass(heatmap, kernel_size)
    # check shape
    if heatmap.shape[0] != center.shape[0] or heatmap.shape[2] != center.shape[2] or heatmap.shape[3] != center.shape[3]:
        raise Exception("output shape of calc_center_of_mass is different from input shape")
    # check calculation
    heatmap_exp = torch.exp(heatmap)
    c_x = 1+(-torch.sum(heatmap_exp[0,0,0:3,0])+torch.sum(heatmap_exp[0,0,0:3,2]))/torch.sum(heatmap_exp[0,0,0:3,0:3])
    c_y = 1+(-torch.sum(heatmap_exp[0,0,0,0:3])+torch.sum(heatmap_exp[0,0,2,0:3]))/torch.sum(heatmap_exp[0,0,0:3,0:3])
    if torch.abs(c_x - center[0,0,1,1])>eps or torch.abs(c_y - center[0,1,1,1])>eps:
        raise Exception("calc_center_of_mass output wrong result")
    print("Pass")

