import torch
import torch.nn.functional as nnf
import argparse
import numpy as np
import random
import os
import time
import math
import IPython
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

# from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from summary import CustomSummaryWriter
from utils.utils import xywh_to_xyxy
from dataset import Dataset, PascalVOCMetaData
from utils.config_utils import print_config, str2bool, get_oicr_config, config_to_string
from utils.torch_utils import to_gpu, nms_detector
from utils.eval_utils import _do_python_eval
from oicr import OICR
from loss_functions import calc_wsdd_loss, calc_oicr_loss

def xywh_to_xyxy(boxes):
    K_min = (boxes[:,:,0:2])
    K_max = (boxes[:,:,0:2] + boxes[:,:,2:4])
    return torch.cat([K_min, K_max], dim=2)

class OICRTrainer():
    def __init__(self, config, oicr):
        # copy config
        self.config = config

        # init iteration counter
        self.num_iter = 0
        # init summary writer
        run = config.log_dir + '/' + config.name   
        self.summary = CustomSummaryWriter(run)

        # create the solvers
        self.solver = torch.optim.Adam(oicr.parameters(), lr=config.lr)

        self.num_iter = 0

        self.model_path = os.path.join(config.model_dir, config.name+'_'+config.net+'_model')
        self.solver_path = os.path.join(config.model_dir, config.name+'_'+config.net+'_checkpoint')

        if not os.path.isdir(config.model_dir):
            os.mkdir(config.model_dir)

        self.resume = config.resume
        self.result_dir = config.result_dir
        if not os.path.isdir(config.result_dir):
            os.mkdir(config.result_dir)
    def train(self, oicr, dataset_tr, dataset_va, metadata_tr, metadata_va):
        if self.resume:
            if os.path.isfile(self.model_path) and os.path.isfile(self.solver_path):
                checkpoint = torch.load(self.solver_path)
                self.solver.load_state_dict(checkpoint['solver'])
                self.num_iter = checkpoint['num_iter']
                model = torch.load(self.model_path)
                oicr.load_state_dict(model['model'])

        init_flag = False
        # main training loop
        for epoch in tqdm(range(self.config.epochs)):
            for x in tqdm(dataset_tr, smoothing=0.1):

                # get data
                image_ids, scale, images, bbox_gt, labels_gt, ss_proposals = to_gpu(x)

                # train model
                score_list = oicr.forward(images.squeeze(), ss_proposals.squeeze())

                if self.config.net =='wsdd':
                    wsdd_loss, oicr_loss = calc_wsdd_loss(score_list,ss_proposals.squeeze(), labels_gt.squeeze())
                else:
                    wsdd_loss, oicr_loss = calc_oicr_loss(score_list,ss_proposals.squeeze(), labels_gt.squeeze())
                (wsdd_loss + oicr_loss).backward()
                
                if self.num_iter%8 ==0:
                    self.solver.step()
                    self.solver.zero_grad()
                # add loss to tensorboard
                self.summary.add_scalar('wsdd loss', wsdd_loss, self.num_iter)
                self.summary.add_scalar('oicr loss', oicr_loss, self.num_iter)
                self.summary.add_scalar('total loss', wsdd_loss+oicr_loss, self.num_iter)
                # evaluate 
                if self.num_iter % self.config.eval_period == 0:
                    if self.config.net =='wsdd':
                        bboxes = nms_detector(torch.cat((torch.zeros(score_list[0].shape[0],1),score_list[0]),dim=1),ss_proposals.squeeze(), 1e-1, 0.4)
                    else:
                        score = (score_list[1] + score_list[2] + score_list[3])/3
                        bboxes = nms_detector(score,ss_proposals.squeeze(), 1e-2, 0.3)

                    self.summary.add_images('predicted bbox', images, self.num_iter, 
                                       boxes_infer=xywh_to_xyxy(bboxes.unsqueeze(0)[:,:,:4]), 
                                       boxes_gt=bbox_gt, 
                                       labels=bboxes.unsqueeze(0)[:,:,5], resize=2.0)
                    if self.config.net =='wsdd':
                        gt_vector = torch.zeros(20)
                        gt_vector[labels_gt-1] = 1 
                    else:
                        gt_vector = torch.zeros(21)
                        gt_vector[labels_gt] = 1 
                    gt_vector = (gt_vector - torch.sum(score_list[0],dim=0)).cpu()
                    gt_str = ' '.join(['{:.2f}'.format(s) for s in gt_vector]) 
                    self.summary.add_text('label difference', gt_str, self.num_iter)   
                       

                    # gt_label = labels_gt
                    # self.summary.add_text('predicted bbox', images, self.num_iter, 
                    #                    boxes_infer=bbox.unsqueeze(0)[:,:,:4], 
                    #                    boxes_gt=bbox_gt, 
                    #                    labels=bbox.unsqueeze(0)[:,:,5], resize=2.0)
                # validation
                fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
                if self.num_iter % self.config.valid_period == 0 and self.num_iter!=0:

                    print('saving model')
                    torch.save({'model': oicr.state_dict()}, 
                                self.model_path)
                    torch.save({'solver': self.solver.state_dict(), 
                                'num_iter': self.num_iter}, 
                                self.solver_path)

                    if self.config.valid_flag:                    
                        print('start evaluation ...')
                        f = []
                        for cls in range(0,21):

                            file_name = '_'.join(['det','val',metadata_va.get_class_name(cls)]) + '.txt'
                            file_path = os.path.join(self.result_dir,file_name)
                            f.append(open(file_path,'w'))
                        loss = 0
                        counter = 0

                        for x in tqdm(dataset_va, smoothing=0.1):

   
                            image_ids, scale, images, bbox_gt, labels_gt, ss_proposals = to_gpu(x)

                            # import IPython
                            # IPython.embed()
                            # assert(0)
                            score_list= []
                            init_flag = True
                            num_aug = 0
                            for size in [480,576,688,864]:
                                for flip_flag in [True,False]:
                                    tmp_ss_proposals = ss_proposals.clone()
                                    tmp_images = images.clone()
                                    if flip_flag:
                                        tmp_images = torch.flip(images, [-1])   
                                        tmp_ss_proposals[:,:,0] = images.shape[3] - ss_proposals[:,:,0] - 1
                                    tmp_scale = max(tmp_images.shape[-1], tmp_images.shape[-2])/size
                                    tmp_images = nnf.interpolate(tmp_images,scale_factor = 1/tmp_scale, mode='bilinear',align_corners=True).squeeze()
                                    tmp_ss_proposals = tmp_ss_proposals.squeeze()/tmp_scale
                                    if init_flag:
                                        score_list = [_score_1.detach() for _score_1 in oicr.forward(tmp_images, tmp_ss_proposals)]
                                        init_flag = False
                                    else:
                                        score_list =[_score_1.detach()+_score_2 for _score_1, _score_2 in zip(oicr.forward(tmp_images, tmp_ss_proposals),score_list)]
                                    num_aug = num_aug + 1

                            score_list = [score/num_aug for score in score_list]

                            if self.config.net =='wsdd':
                                wsdd_loss, oicr_loss = calc_wsdd_loss(score_list,ss_proposals.squeeze(), labels_gt.squeeze())
                            else:
                                wsdd_loss, oicr_loss = calc_oicr_loss(score_list,ss_proposals.squeeze(), labels_gt.squeeze())
                            tmp_loss = (wsdd_loss + oicr_loss).detach().cpu()
            
                            loss = loss + tmp_loss
                            counter = counter + 1
                            if self.config.net =='wsdd':
                                bboxes = nms_detector(torch.cat((torch.zeros(score_list[0].shape[0],1),score_list[0]),dim=1),ss_proposals.squeeze(), 1e-2, 0.5)
                            else:
                             
                                score = (score_list[1] + score_list[2] + score_list[3])/3
                                bboxes = nms_detector(score,ss_proposals.squeeze(), 1e-2, 0.3)

                            image_name = metadata_va.get_image_name(image_ids[0])
                            # visulize results
                            viz_folder = './viz'
                            if not os.path.isdir(viz_folder):
                                os.mkdir(viz_folder)

                            # import IPython
                            # IPython.embed()
                            # assert(0)
                            im = Image.fromarray((images[0].permute(1,2,0).cpu().data.numpy()*255).astype(np.uint8))
                            draw = ImageDraw.Draw(im)
                            draw.text((0,0), '{:.2f}'.format(tmp_loss),font=fnt,  align ="left", fill=(0,0,255,255))   
                            for bbox in bboxes:     

                                # if bbox[0]>bbox[2] and bbox[1]<bbox[3]:
                                #     import IPython
                                #     IPython.embed()
                                #     assert(0)
                                draw.rectangle([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]],outline = 'red')  
                                cls = int(bbox[-1].item())   
                                
                                draw.text((bbox[0], bbox[1]), '{} {:.2f}'.format(metadata_va.get_class_name(cls),bbox[-2]),font=fnt, fill=(255,0,0,255))                    

                                bbox_coor = bbox[0:-2] * scale

                                bbox_coor[2] = bbox_coor[0] + bbox_coor[2]
                                bbox_coor[3] = bbox_coor[1] + bbox_coor[3]
                                str_buffer = image_name+' '+' '.join('{:6f}'.format(x) for x in [bbox[-2].tolist()]+bbox_coor.tolist()) +'\n'   

                                f[cls].write(str_buffer)
                            for bbox in bbox_gt[0]:
                                draw.rectangle([bbox[0],bbox[1],bbox[2],bbox[3]],outline = 'green')   
                            im.save(os.path.join(viz_folder,'{}.jpg'.format(image_name)),'JPEG')
                        loss = loss/counter
                        self.summary.add_scalar('val total loss', loss, self.num_iter)
                        for cls in range(20):
                            f[cls].close()
     
                        _do_python_eval(os.path.join(self.result_dir,'_'.join(['det','val'])+'_'), output_dir = 'output/val',mode='val')
                        

                self.summary.flush()
                            
                self.num_iter += 1


    # def eval_accuracy (self, bboxes, bboxes_gt, labels, labels_gt, image_shape, num_class=10):
    #     import IPython
    #     IPython.embed()
    #     assert(0)
    #     output = {}
    #     if len(bbox) == 0:
    #         output['num_corr_dect'] = 0
    #         output['num_all'] = 0
    #         output['acc_loc'] = 0
    #         output['acc_class_all'] = 0
    #         output['acc_class_corr_dect'] = 0
    #         return 
    #     for bbox in bboxes:

    #     # K [B, N, (x,y,w,h)]
    #     # labels [B, N]
    #     output = {}

    #     # total number of bbox in a batch
    #     N = (bbox.shape[0] * bbox.shape[1])
        
    #     # min IOU used for evaluation
    #     minIoU = 0.5

    #     # compute IoU between each keypoint and keypoint_gt
    #     bbox_min = (bbox[:,:,0:2] - bbox[:,:,2:4] / 2).unsqueeze(2) # [B, nk, 1, 2]
    #     bbox_max = (bbox[:,:,0:2] + bbox[:,:,2:4] / 2).unsqueeze(2)
    #     bbox_gt_min = (bbox_gt[:,:,0:2] - bbox_gt[:,:,2:4] / 2).unsqueeze(1) # [B, 1, nkg, 2]
    #     bbox_gt_max = (bbox_gt[:,:,0:2] + bbox_gt[:,:,2:4] / 2).unsqueeze(1)

    #     botleft = torch.max(bbox_min, bbox_gt_min)
    #     topright = torch.min(bbox_max, bbox_gt_max)

    #     inter = torch.prod(torch.nn.functional.relu(topright - botleft), dim=3)
    #     area_bbox = torch.prod(bbox_max - bbox_min, dim=3)
    #     area_bbox_gt = torch.prod(bbox_gt_max - bbox_gt_min, dim=3)
    #     union = area_bbox + area_bbox_gt - inter
    #     iou = inter / union # [B, k, kg, 1]   
    #     iou[iou != iou] = 0

    #     # set iou of ground truth background class to 0
    #     # iou = (labels_gt!=0).unsqueeze(1).type(torch.float32)*iou     
    #     # total number of objects in batch
    #     # num_objects = (labels_gt!=0).sum() 
    #     num_objects = (labels_gt>=0).sum() 

    #     acc_iou = ((torch.max(iou, dim=1)[0] > minIoU)*(labels_gt!=0)).sum().float() / num_objects
    #     match_det = (torch.max(iou, dim=2)[0] > minIoU)
    #     selected_gt = torch.gather(labels_gt, dim=1, index=torch.max(iou, dim=2)[1])
    #     match_class = torch.eq(labels, selected_gt)
    #     output['keypoint_match_detection'] = torch.stack([match_det.float(), torch.max(iou, dim=2)[0],match_class.float()], dim=2)


    #     labels_match = torch.eq(labels.unsqueeze(2), labels_gt.unsqueeze(1)) # [B, k, kg]
    #     iou = iou * labels_match.float()


    #     matches = 0
    #     for b in range(bbox.shape[0]):
    #         for k in range(bbox.shape[1]):
    #             val, idx = torch.max(iou[b,k,:], dim=0)
    #             if val >= minIoU:
    #                 iou[b, :, idx] = 0.0
    #                 matches += 1
    #     acc = matches / num_objects.item() 

    #     dt_1h = torch.nn.functional.one_hot(labels, num_class).sum(dim=1)[:,1:] 
    #     gt_1h = torch.nn.functional.one_hot(labels_gt, num_class).sum(dim=1)[:,1:]  
    #     acc_class_all_detect = 1.0 - torch.relu(gt_1h-dt_1h).sum().float()/num_objects

    #     # find closet match between detetced kp and ground truth kp
    #     # distance matrix between predicted kp and ground truth kp
    #     # row: predicted kp column: ground truth kp
    #     dist_mats = construct_dist_mat(bbox[:,:,:2],bbox_gt[:,:,:2])
    #     # dist_mats = dist_mats + (labels_gt==0).unsqueeze(1).type(torch.float32)*torch.ones(iou.shape)*torch.max(dist_mats)    
    #     best_dist = torch.zeros(dist_mats.shape[0],dist_mats.shape[1])  
    #     for idx, dist_mat in enumerate(dist_mats): 
    #         kp_opt_match_idx = linear_sum_assignment(dist_mat[:,labels_gt[idx]!=0] .cpu().detach().numpy())
    #         best_dist[idx,0:kp_opt_match_idx[0].shape[0]] = dist_mat[kp_opt_match_idx[0],kp_opt_match_idx[1]]    
    #     acc_loc = torch.mean(torch.sqrt(best_dist+0.001))/ math.sqrt(image_shape[0]**2+image_shape[1]**2)*100   


    #     if torch.sum(match_det).cpu().detach().numpy() ==0:
    #         acc_class_corr_detect = 0
    #     else:
    #         acc_class_corr_detect = torch.sum(match_class & match_det).cpu().detach().numpy()/ \
    #             torch.sum(match_det).cpu().detach().numpy() 

    #     output['acc'] = acc
    #     output['acc_det'] = acc_iou
    #     output['acc_loc'] = acc_loc
    #     output['acc_class_all'] = acc_class_all_detect
    #     output['acc_class_corr_dect'] = acc_class_corr_detect


    #     return output

    def validate(self, mist, dataset):

        images, keypoints_gt, labels_gt = to_gpu(next(iter(dataset)))
        bboxs , labels = mist.forward(images)

        eval_val = self.eval_accuracy (bboxs, keypoints_gt, labels, labels_gt, images.shape[1:], mist.info['num_classes'])

        return eval_val



def main():

    config = get_oicr_config()
    print_config(config)

    # set torch seed
    if config.set_seed:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False   

    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_printoptions(profile="full")
    torch.set_printoptions(threshold=5000)
    torch.set_printoptions(precision=10)

    # init data loader
    dataset_tr = Dataset(config, mode='train')
    dataset_va = Dataset(config, mode='valid')

    # init dataset metadata
    metadata_tr = PascalVOCMetaData(config, mode='train')
    metadata_va = PascalVOCMetaData(config, mode='val')   
    # init network
    oicr = OICR(config)

    # init network trainer
    oicr_trainer = OICRTrainer(config, oicr)
    
    # # resume model
    # if config.resume:
    #     mist_trainer.resume(wsdd)

    # # wirte meta data if first time run
    # if not mist_trainer.resumed:
    #     mist_trainer.write_meta_data()

    # train model
    oicr_trainer.train(oicr, dataset_tr, dataset_va, metadata_tr, metadata_va)


if __name__ == '__main__':
    main()
    

    



