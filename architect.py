import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
dtype = torch.cuda.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.append('drive/MyDrive/YOLACT/')
from utils.box_utils import match, crop, make_anchors
from utils.output_utils import *
import pdb
import time
import pickle
import cv2
import numpy as np
import glob
from PIL import Image

class Recurrent_res( nn.Module ):

  def __init__(self , inplane , outplane  , stride = 1 , downsample = None ):
    
    super().__init__()
    self.conv1 = nn.Conv2d(inplane , outplane , kernel_size = 1 , bias = False )
    self.bn1 = nn.BatchNorm2d(outplane)
    self.conv2 = nn.Conv2d(outplane , outplane , kernel_size = 3 , padding = 1 ,  stride = stride , bias = False )
    self.bn2 = nn.BatchNorm2d(outplane)
    self.conv3 = nn.Conv2d(outplane , outplane*4 , kernel_size = 1 ,  bias = False )
    self.bn3 = nn.BatchNorm2d(outplane *4 )
    self.relu = nn.ReLU(inplace = True)
    self.downsample = downsample
    self.stride = stride
  
  def forward(self , x ):
    
    # Save residual for skip connection
    residual = x 
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    
    out = self.conv3(out)
    out = self.bn3(out)
    
    if self.downsample is not None:
      residual = self.downsample(x)
    out+= residual
    out = self.relu(out)
    
    return out

class Resnet(nn.Module):
  # Resnet Backbone

  def __init__(self ):
    super().__init__()
    self.conv1 = nn.Conv2d(3,64,kernel_size = 7 , stride = 2 , padding = 3 , bias = False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace = True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.layers = nn.ModuleList()
    self.inplanes = 64
    self.make_recurrent_layer(64,3,1)
    self.make_recurrent_layer(128,4,2)
    self.make_recurrent_layer(256,6,2)
    self.make_recurrent_layer(512,3,2)


  def make_recurrent_layer(self , inplanes , num_blocks , stride = 1 ):
    downsample = nn.Sequential( nn.Conv2d(self.inplanes , inplanes*4 , kernel_size = 1 , stride = stride , bias = False ) , nn.BatchNorm2d( inplanes*4) )
    layers = [Recurrent_res(self.inplanes , inplanes , stride = stride , downsample = downsample  )]
    self.inplanes = inplanes*4

    for i in range(num_blocks - 1):
      layers.append(Recurrent_res(self.inplanes , inplanes  ))
    
    layer = nn.Sequential(*layers)
    self.layers.append(layer)
  
  def init_resnet(self):
    net_50 = torchvision.models.resnet50(pretrained=True, progress=True)
    net_50 = net_50.to(device)
    state_dict = net_50.state_dict()
    
    state_dict.pop("fc.weight")
    state_dict.pop("fc.bias")
    keys = list(state_dict)
    for key in keys:
      if(key.startswith('layer')):
        layer_no = int(key[5])
        new_key = 'layers.'+str(layer_no-1)+key[6:]
        state_dict[new_key] = state_dict.pop(key)


    """ print('sanity check')
    for key in state_dict:
      s1 = state_dict[key]
      s1 = s1.to(torch.device('cuda'))
      s2 = self.state_dict()[key]
      
      if(s1.shape != s2.shape ):
        print(key)
      if(key not in self.state_dict().keys()):
        print(key)
    """

    self.load_state_dict(state_dict , strict = False )




  def forward(self , x ):

    # Returns a list of Conv final layers needed in yolact 
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    outs = []
    for i in range(4):
      x = self.layers[i](x)
      if i>0:
        outs.append(x)
    
    return tuple(outs)

class FPN(nn.Module):
  def __init__(self , in_channels ):
    super().__init__()
    self.in_channels = in_channels
    self.num_lateral_layers = len(in_channels)
    self.lateral_layers = nn.ModuleList()
    self.pred_layers = nn.ModuleList()
    for channel in reversed(self.in_channels):
      self.lateral_layers.append(nn.Conv2d(channel , 256 , kernel_size = 1 ))
      self.pred_layers.append(nn.Conv2d(256,256,kernel_size = 3 , padding = 1 ))  
    self.downsample1 = nn.Conv2d(256 , 256 ,kernel_size = 3 ,  padding = 1 , stride = 2   )
    self.downsample2 = nn.Conv2d(256 , 256 ,kernel_size = 3 ,  padding = 1 , stride = 2  )
  
  def forward(self , resnet_out ):
    outs = []

    out3 = self.lateral_layers[0](resnet_out[2])
    x = torch.zeros( out3.shape , device=resnet_out[0].device)
    x += out3

    _, _, h, w = resnet_out[1].size()
    x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
    out2 = self.lateral_layers[1](resnet_out[1])
    x+= out2
    out2 = x

    _, _, h, w = resnet_out[0].size()
    x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
    out1 = self.lateral_layers[2](resnet_out[0])
    x += out1
    out1 = x
    
    out3 = F.relu(self.pred_layers[2](out3))
    out2 = F.relu(self.pred_layers[1](out2))
    out1 = F.relu(self.pred_layers[0](out1))

    outs.append(out1)
    outs.append(out2)
    outs.append(out3)

    outs.append(self.downsample1(outs[-1]))
    outs.append(self.downsample2(outs[-1]))
    return outs


class Protonet(nn.Module):
  def __init__( self ):
    super().__init__()
    self.layers = nn.ModuleList()
    for i in range(3):
      self.layers.append(nn.Sequential( nn.Conv2d( 256, 256,kernel_size = 3 , padding = 1  ) , nn.ReLU(inplace = True) ) )
    self.proto_conv1 = nn.Conv2d(256,256,kernel_size = 3 , padding = 1 )
    self.relu = nn.ReLU(inplace = True)
    self.proto_conv2 = nn.Conv2d(256 , 32 , kernel_size = 1)
  
  def forward(self , x ):
    out = x
    for layer in self.layers:
      out = layer(out)
    out = F.interpolate(out , (138,138) , mode = 'bilinear' , align_corners = False )
    out = self.relu(out)
    out = self.proto_conv1(out)
    out = self.relu(out)
    out = self.proto_conv2(out)
    return out

class Prediction_module(nn.Module):
  def __init__(self , cfg , coefdim = 32 ):
    super().__init__()
    self.num_classes = cfg.num_classes
    self.coefdim = coefdim
    self.num_anchors = len(cfg.aspect_ratios)
    self.conv1 = nn.Conv2d(256 , 256 , kernel_size = 3 , padding = 1  )
    self.relu = nn.ReLU(inplace= True)
    self.bbox_layer = nn.Conv2d( 256 , self.num_anchors*4 , kernel_size = 3 , padding = 1  )
    self.conf_layer = nn.Conv2d( 256 , self.num_anchors*self.num_classes , kernel_size = 3 , padding = 1 )
    self.mask_layer = nn.Conv2d( 256 , self.num_anchors*self.coefdim , kernel_size = 3, padding = 1 )

  def forward( self , x ):
    x = self.conv1(x)
    x = self.relu(x)
    conf = self.conf_layer(x).permute(0,2,3,1).reshape(x.size(0) , -1 , self.num_classes )
    box = self.bbox_layer(x).permute(0,2,3,1).reshape(x.size(0) , -1 , 4 )
    coef = self.mask_layer(x).permute(0,2,3,1).reshape(x.size(0) , -1 , self.coefdim )
    coef = torch.tanh( coef )
    return conf , box , coef 





