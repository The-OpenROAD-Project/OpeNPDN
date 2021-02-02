import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
from torch.utils.data import Dataset
import numpy  as np

class Net(nn.Module):
  def __init__(self,num_classes):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, padding=2)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.pool3 = nn.MaxPool2d(3, 3)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
    self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
    self.fc1 = nn.Linear(12*12*64 , 1024) #100 x 100 region 
    self.fc2 = nn.Linear(1024, num_classes)
    self.Dropout = nn.Dropout(p=0.5)

  def forward(self, x):
    x = self.pool3(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = self.pool2(F.relu(self.conv3(x)))
    x = self.pool2(F.relu(self.conv4(x)))
    x = x.view(-1,12*12*64)
    x = self.Dropout(x)
    x = F.relu(self.fc1(x))
    x = self.Dropout(x)
    x = self.fc2(x)
    #x = self.Dropout(x)
    return x

class OpeNPDNDataset(Dataset):
  def __init__(self, input_data, labels_present=False, normalize=False):
    self.labels_present = labels_present
    for n,data_pt in enumerate(tqdm(input_data)):
      if n == 0:
        self.current_maps    = data_pt['current_maps']
        self.macro_maps      = data_pt['macro_maps']
        self.congestion_maps = data_pt['congestion_maps']
        self.eff_dist_maps   = data_pt['eff_dist_maps']
        if labels_present:
          self.labels = data_pt['state']
      else:
        self.current_maps    = np.append(self.current_maps, data_pt['current_maps'], axis=0)
        self.macro_maps      = np.append(self.macro_maps, data_pt['macro_maps'], axis=0)
        self.congestion_maps = np.append(self.congestion_maps, data_pt['congestion_maps'], axis=0)
        self.eff_dist_maps   = np.append(self.eff_dist_maps, data_pt['eff_dist_maps'], axis=0)
        if labels_present:
          self.labels =  np.append(self.labels, data_pt['state'], axis=0)
    if labels_present:
      self.num_classes = np.max(self.labels)+1#len(np.unique(self.labels))
    if normalize:
      self.normalize()
  
  def normalize(self,):
    self.max_cur = np.max(self.current_maps)
    self.max_cong = np.max(self.congestion_maps)
    self.max_dist = np.max(self.eff_dist_maps  )
    self.current_maps    = (self.current_maps)/self.max_cur
    self.congestion_maps = (self.congestion_maps)/self.max_cong
    self.eff_dist_maps   = (self.eff_dist_maps)/np.max(self.eff_dist_maps  )
        
  def __len__(self):
    return self.current_maps.shape[0]
  
  def store_normalize(self, chk_pt_dir):
    np.savez( "%s/norm_params.npz"%chk_pt_dir, 
              max_curr = self.max_cur,
              max_cong = self.max_cong,
              max_dist = self.max_dist)
    
  def load_normalize(self,chk_pt_dir, normalize=False):
    norm_data = np.load("%s/norm_params.npz"%chk_pt_dir)
    self.max_cur  = norm_data['max_curr']
    self.max_cong = norm_data['max_cong']
    self.max_dist = norm_data['max_dist']
    if normalize:
      self.current_maps    = (self.current_maps)/self.max_cur
      self.congestion_maps = (self.congestion_maps)/self.max_cong
      self.eff_dist_maps   = (self.eff_dist_maps)/np.max(self.eff_dist_maps  )
    
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    X = torch.empty((4,)+self.current_maps.shape[1:])
    X[0,...] = torch.tensor(self.current_maps[idx])    
    X[1,...] = torch.tensor(self.congestion_maps[idx])
    X[2,...] = torch.tensor(self.macro_maps[idx])   
    X[3,...] = torch.tensor(self.eff_dist_maps[idx])
    if self.labels_present:
      Y = torch.tensor(self.labels[idx])
    else:
      Y = torch.empty((1,)+self.current_maps.shape[1:-2])
    return (X,Y)


