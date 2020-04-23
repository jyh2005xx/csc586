import torch
from net import BackBoneVGG
from spp import SpatialPyramidPooling
from torchvision.ops import RoIPool



class OICR(torch.nn.Module):
    def __init__(self, config):
        super(OICR, self).__init__()
        self.net = config.net
        # Pretrained CNN to Extract Feature Map
        self.backbone = BackBoneVGG(config).cuda()


        # Fully Connected Layer
        self.fc6 = torch.nn.Linear(21*512, 4096)
        self.fc7 = torch.nn.Linear(4096, 4096)
        self.fc8c = torch.nn.Linear(4096, 20)
        self.fc8d = torch.nn.Linear(4096, 20)
        self.roi_pool = RoIPool((7,7), spatial_scale=1.0)
        self.spp = SpatialPyramidPooling(mode='avg')
        self.softmax_c = torch.nn.Softmax(dim=1)
        self.softmax_d = torch.nn.Softmax(dim=0)

        if self.net == 'oicr':
            self.fc_icr_1 = torch.nn.Linear(4096, 21)
            self.fc_icr_2 = torch.nn.Linear(4096, 21)
            self.fc_icr_3 = torch.nn.Linear(4096, 21)

        # info
        self.info = {}
        # self.info['num_classes'] = config.num_classes
        

    def forward(self, image, proposals):


        # extract feature
        featuremap = self.backbone.forward(image.unsqueeze(0))

        stride_h = image.shape[-2]/featuremap.shape[-2] 
        stride_w = image.shape[-1]/featuremap.shape[-1] 
        scaled_proposals = torch.zeros(proposals.shape)
        scaled_proposals[:,[0,2]] = proposals[:,[0,2]] / stride_w
        scaled_proposals[:,[1,3]] = proposals[:,[1,3]] / stride_h



        scaled_proposals=torch.cat([torch.zeros(scaled_proposals.shape[0],1),scaled_proposals],dim=1)  
        featuremap = self.roi_pool(featuremap,scaled_proposals)
        # spatial pyramid pool

        feature = self.spp(featuremap).view(featuremap.shape[0],-1)

        # # fully connected layers
        feature = self.fc6(feature)
        feature = self.fc7(feature)
        feature_c = self.fc8c(feature)
        feature_d = self.fc8d(feature)

        feature_c = self.softmax_c(feature_c)
        feature_d = self.softmax_d(feature_d)

        score = feature_c * feature_d 



        score_list = []
        score_list.append(score)
        if self.net == 'oicr':
            score_list.append(self.softmax_c(self.fc_icr_1(feature)))
            score_list.append(self.softmax_c(self.fc_icr_2(feature)))
            score_list.append(self.softmax_c(self.fc_icr_3(feature)))

        return score_list







