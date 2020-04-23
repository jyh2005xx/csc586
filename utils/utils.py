import torch
def xys_to_xywh(boxes):
    return torch.cat([boxes, boxes[...,2,None]], dim=-1)

def xywh_to_xyxy(boxes):
    K_min = (boxes[:,:,0:2] - boxes[:,:,2:4] / 2)
    K_max = (boxes[:,:,0:2] + boxes[:,:,2:4] / 2)
    return torch.cat([K_min, K_max], dim=2)