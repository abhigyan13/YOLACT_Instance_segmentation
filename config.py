
import os
import numpy as np
import torch
import torch.distributed as dist

class set_config:
  def __init__(self):
    self.img_size = 550
    self.pascal_classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    self.label_id =  {'aeroplane':0, 'bicycle':1 , 'bird' : 2 , 'boat' : 3 , 'bottle' : 4 ,
                  'bus' : 5 , 'car' : 6 , 'cat' : 7 , 'chair' : 8 , 'cow' : 9 ,
                  'diningtable' : 10 , 'dog' : 11 , 'horse': 12 , 'motorbike' : 13 , 'person' : 14 ,
                  'pottedplant' : 15 , 'sheep' : 16 , 'sofa' : 17 , 'train' : 18, 'tvmonitor': 19 }
    self.num_classes = 21
    self.coef_dim = 32
    self.num_anchors = 3
    self.lr=0.001
    self.mode = 'train'
    self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.cfg_name = "resnet50_pascal"
    self.norm_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
    self.norm_std = np.array([57.38, 57.12, 58.40], dtype=np.float32)
    self.scales = [int(self.img_size / 550 * aa) for aa in (32, 64, 128, 256, 512)]
    self.aspect_ratios = [1, 1 / 2, 2]
    self.data_root = 'content/drive/MyDrive/YOLACT/'
    self.train_bs = 8
    self.img_path = self.data_root + 'VOCdevkit/VOC2012/JPEGImages'
    self.pos_iou_thre = 0.6
    self.neg_iou_thre = 0.4
    self.masks_to_train = 100
    self.resume = True 
    self.weight_path =   "/content/drive/MyDrive/YOLACT/weights4"
    self.batch_size = 8
    self.test_image_path = '/content/drive/MyDrive/YOLACT/test/i1.jpg'
    self.video = None
    self.top_k = 200
    self.no_crop=False
    self.hide_mask=False
    self.hide_bbox=False
    self.hide_score=False
    self.cutout=False
    self.real_time=False
    self.save_lincomb=False
    self.visual_thre=0.2
    self.nms_score_thre = 0.05
    self.nms_iou_thre = 0.3
    self.max_detections = 100


    self.conf_alpha = 1
    self.bbox_alpha = 1.5
    self.mask_alpha = 6.125
    self.semantic_alpha = 1


def save_latest(net, cfg_name , step , path =  "/content/drive/MyDrive/YOLACT/weights4"):
  weight = glob.glob(path + '/latest*')
  weight = [aa for aa in weight if cfg_name in aa]
  assert len(weight) <= 1, 'Error, multiple latest weight found.'
  if weight:
    os.remove(weight[0])

  print(f'\nSaving the latest model as \'latest_{cfg_name}_{step}.pth\'.\n')
  torch.save(net.state_dict(), f'{path}/latest_{cfg_name}_{step}.pth')    