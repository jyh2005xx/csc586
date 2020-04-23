import bisect
import copy
import torch
import torch.utils.data
import numpy as np
import os
import skimage.io
import skimage.transform
import random
from tqdm import tqdm
import scipy.io as sio
from six.moves import cPickle
import random
# from grouped_batch_sampler import GroupedBatchSampler

import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

CLASSES = (
    "__background__ ",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

class PascalVOCMetaData():
    def __init__(self, config, mode):
        self.root_dir = config.dataset_dir+'/'+config.dataset+'/'
        self.image_set = mode
        self._imgsetpath = os.path.join(self.root_dir, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]

        self.cls = CLASSES
    def get_class_name(self, class_id):
        return self.cls[class_id]
    def get_image_name(self, image_id):
        return self.ids[image_id]
# def compute_aspect_ratios(dataset):
#     aspect_ratios = []
#     for i in range(len(dataset)):
#         img_info = dataset.get_img_info(i)
#         aspect_ratio = float(img_info["height"]) / float(img_info["width"])
#         aspect_ratios.append(aspect_ratio)
#     return aspect_ratios

# def _quantize(x, bins):
#     bins = copy.copy(bins)
#     bins = sorted(bins)
#     quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
#     return quantized

def Dataset(config, mode='train'):
    if mode=='valid':
        mode='val'
    # init dataset
    dataset = PascalVOCDataset(config, mode)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    
    return loader

# modified based on 'maskrcnn-benchmark' 
# git@github.com:facebookresearch/maskrcnn-benchmark.git
class PascalVOCDataset(torch.utils.data.Dataset):


    def __init__(self, config, mode):
        self.root_dir = config.dataset_dir+'/'+config.dataset+'/'
        self.image_set = mode
        self.keep_difficult = False
        self._annopath = os.path.join(self.root_dir, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root_dir, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root_dir, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

    def __getitem__(self, index):
        # import IPython
        # IPython.embed()
        # assert(0)
        img_id = self.ids[index]
        image = skimage.io.imread(self._imgpath%img_id)
        # import IPython
        # IPython.embed()
        # assert(0)
        # resize image
        if self.image_set == 'train':
            max_length = random.choice([480,576,688,864,1200])
        else: 
            max_length = max(image.shape[0], image.shape[1])

        scale = max(image.shape[0], image.shape[1])/max_length
        size = [int(image.shape[0]/scale),int(image.shape[1]/scale)]
        image = skimage.transform.resize(image,size)

        image = torch.from_numpy(image).permute(2, 0, 1).float()

        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)
        bboxes = anno['boxes']
        bboxes = bboxes/scale

        labels = anno['labels']

        ss_proposals = self._get_ss_proposal(img_id)/scale

        ss_proposals[:,[2,3]] = ss_proposals[:,[0,1]] + ss_proposals[:,[2,3]]
        ss_proposals[:,2] = torch.clamp(ss_proposals[:,2],0,image.shape[2]-1)
        ss_proposals[:,3] = torch.clamp(ss_proposals[:,3],0,image.shape[1]-1)
        ss_proposals[:,[2,3]] = ss_proposals[:,[2,3]] - ss_proposals[:,[0,1]]
        if self.image_set == 'train':
            if bool(random.getrandbits(1)):
                image = torch.flip(image, [2])   
                bboxes[:,[0,2]] = image.shape[2] - bboxes[:,[2,0]] - 1
                bboxes = torch.clamp(bboxes,0)
                ss_proposals[:,0] = image.shape[2] - ss_proposals[:,0] - 1
        return torch.tensor(index,device=torch.device('cpu')), torch.tensor(scale,device=torch.device('cpu')), image, bboxes, labels, ss_proposals

    def __len__(self):
        return len(self.ids)

    def _get_ss_proposal(self, img_id):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        if not os.path.isdir(os.path.join(self.root_dir, 'SSProposals')):
            print ('First time run. Refomatting selective search files ...')
            self._reformat_ss_data()

        cache_file = os.path.join(self.root_dir, 'SSProposals',
                                  img_id + '.pkl')

        with open(cache_file, 'rb') as fid:
            ss_proposals = cPickle.load(fid)
            return torch.from_numpy(ss_proposals['boxes'].astype(int)).float()

    def _reformat_ss_data(self):
        os.mkdir(os.path.join(self.root_dir, 'SSProposals'))
        for image_set in ['trainval','test']:
            ss_raw_file = os.path.abspath(os.path.join(self.root_dir,
                                                    'selective_search_data',
                                                    'voc_2007_' + image_set + '.mat'))
            assert os.path.exists(ss_raw_file), \
                   'Selective search data not found at: {}'.format(ss_raw_file)
            raw_data = sio.loadmat(ss_raw_file)
            # import IPython
            # IPython.embed()
            # assert(0)
            boxes = raw_data['boxes'].ravel()
            images = raw_data['images'].ravel()

            for i in range(images.shape[0]):
                ss_dict = {}
                ss_dict['img_id'] = images[i][0]
                ss_dict['boxes'] = boxes[i][:, (1, 0, 3, 2)] - 1
                ss_dict['boxes'][:,2] =  ss_dict['boxes'][:,2] - ss_dict['boxes'][:,0]
                ss_dict['boxes'][:,3] =  ss_dict['boxes'][:,3] - ss_dict['boxes'][:,1]
                ss_file = os.path.join(self.root_dir, 'SSProposals',
                                       images[i][0] + '.pkl')
                with open(ss_file, 'wb') as fid:
                    cPickle.dump(ss_dict, fid, cPickle.HIGHEST_PROTOCOL)
        return 

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        num_objects = 0
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            num_objects = num_objects + 1
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32, device=torch.device('cpu')),
            "labels": torch.tensor(gt_classes, device=torch.device('cpu')),
            "difficult": torch.tensor(difficult_boxes, device=torch.device('cpu')),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return CLASSES[class_id]

# # modified based on 'maskrcnn-benchmark' 
# # git@github.com:facebookresearch/maskrcnn-benchmark.git
# class BatchCollector(object):
#     """
#     From a list of samples from the dataset,
#     returns the batched images and targets.
#     This should be passed to the DataLoader
#     """

#     def __init__(self):
#         pass

#     def __call__(self, batch):
#         # transposed_batch = list(zip(*batch))
#         # images = to_image_list(transposed_batch[0], self.size_divisible)
#         # targets = transposed_batch[1]
#         # img_ids = transposed_batch[2]
#         max_h = max([sample[0].shape[1] for sample in batch]) 
#         max_w = max([sample[0].shape[2] for sample in batch])
#         image_tensor = torch.zeros([len(batch),batch[0][0].shape[0],max_h,max_w], device=torch.device('cpu'))
#         keypoints_tensor = torch.zeros([len(batch),batch[0][1].shape[0],4], device=torch.device('cpu'))
#         labels_tensor = torch.zeros([len(batch),batch[0][2].shape[0]], dtype=torch.long, device=torch.device('cpu'))

#         for idx, sample in enumerate(batch):
#             image, keypoints, labels = sample
#             image_tensor[idx,:,:image.shape[1],:image.shape[2]] = image 
#             keypoints_tensor[idx,:,:] = keypoints
#             labels_tensor[idx,:] = labels

#         return image_tensor, keypoints_tensor, labels_tensor

