import torch

def calc_wsdd_loss(score_list, proposals, labels):
    eps = 1e-5
    score =  torch.sum(score_list[0],dim=0)
    labels_vector = torch.zeros(20)
    labels_vector[labels-1] = 1
    loss = -torch.sum(labels_vector*torch.log(torch.clamp(score, min=eps))+(1-labels_vector)*torch.log(torch.clamp(1-score,min=eps)))

    return loss, 0

def calc_iou(bboxes,bbox_r):
	ixmin = torch.max(bboxes[:, 0], bbox_r[0])
	iymin = torch.max(bboxes[:, 1], bbox_r[1])
	ixmax = torch.min(bboxes[:, 2], bbox_r[2])
	iymax = torch.min(bboxes[:, 3], bbox_r[3])
	iw = torch.clamp(ixmax - ixmin + 1., 0.)
	ih = torch.clamp(iymax - iymin + 1., 0.)
	inters = iw * ih

	# union
	uni = ((bbox_r[2] - bbox_r[0] + 1.) * (bbox_r[3] - bbox_r[1] + 1.) +
	       (bboxes[:, 2] - bboxes[:, 0] + 1.) *
	       (bboxes[:, 3] - bboxes[:, 1] + 1.) - inters)

	overlaps = inters / uni
	return overlaps

def calc_oicr_loss(score_list, proposals, labels):

	proposals[:,[2,3]] = proposals[:,[0,1]] + proposals[:,[2,3]]
	num_r = proposals.shape[0]
	num_c = 21
	# import IPython
	# IPython.embed()
	# assert(0)
	img_score = torch.sum(score_list[0],dim=0)
	eps = 1e-5
	labels_vector = torch.zeros(20)
	labels_vector[labels-1] = 1
	wsdd_loss = -torch.sum(labels_vector*torch.log(torch.clamp(img_score, min=eps))+(1-labels_vector)*torch.log(torch.clamp(1-img_score,min=eps)))
	oicr_loss = 0
	score_list[0] = torch.cat((torch.zeros(num_r,1),score_list[0]),dim=1)
	for i in range(1,len(score_list)):

		pseudo_labels = torch.zeros(score_list[i-1].shape)
		pseudo_labels[:,0] = 1
		best_ious = torch.zeros(num_r)-0.1
		w = torch.zeros(num_r,1)
		try:
			a= len(labels)
		except:
			labels = labels.unsqueeze(0)
		for label in labels:
			tmp_label_vector = torch.zeros(num_c)
			tmp_label_vector[label] = 1
			max_s_r = torch.argmax(score_list[i-1][:,label]) 
			max_s = score_list[i-1][max_s_r,label].detach()
			pseudo_labels[max_s_r,:] = tmp_label_vector
			tmp_ious = calc_iou(proposals, proposals[max_s_r])
			w[tmp_ious>best_ious] = max_s
			pseudo_labels[(tmp_ious>0.5)*(tmp_ious>best_ious),:] = tmp_label_vector
			best_ious[(tmp_ious>0.5)*(tmp_ious>best_ious)] = tmp_ious[(tmp_ious>0.5)*(tmp_ious>best_ious)]
		# tmp_loss = -torch.sum(torch.clamp(w*score_list[i],min=eps).log()*pseudo_labels)/num_r
		tmp_loss = -torch.sum(torch.clamp(score_list[i],min=eps).log()*pseudo_labels)/num_r

		oicr_loss = oicr_loss + tmp_loss


	return wsdd_loss, oicr_loss

