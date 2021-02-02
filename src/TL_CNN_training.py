import torch
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torch.optim as optim
import torch.nn as nn
import os
from tqdm import trange, tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt
from CNN import Net, OpeNPDNDataset
from glob import glob
import re

def main(data_dir,log_level, syn_chk_pt, tl_chk_pt, logger_file=None):
  global logger_h
  logger_h = create_logger(log_level, logger_file)
  TL_data = load_data(data_dir)
  plot_data = logger_h.getEffectiveLevel() <= logging.DEBUG 
  if plot_data:
    plot_label_dist(TL_data)
  TL_dataset = OpeNPDNDataset(list(TL_data.values()), 
                              labels_present=True,
                              normalize=False)
  TL_dataset.load_normalize(os.path.dirname(syn_chk_pt), True)
  set_global_hyperparams()
  TL_model = train_loop(TL_dataset, syn_chk_pt, plot_data)
  torch.save(TL_model, tl_chk_pt) 
  if plot_data:
    plt.show()

def set_global_hyperparams():
  global learning_rate
  global epochs
  global weight_decay_rate
  global LR_decay_rate
  epochs = 20
  learning_rate = 0.0001
  LR_decay_rate = 0.98
  weight_decay_rate = 1e-4

def load_data(data_dir):
  logger_h.info("Loading data from: %s"%data_dir)
  input_data = {}
  for n,filename in enumerate(tqdm(glob("%s/*/TL_data.npz"%data_dir))):
    logger_h.debug("Loading file: %s"%filename)
    design_name = os.path.basename(os.path.dirname(filename))
    input_data[design_name] = np.load(filename)
  logger_h.debug("Number of data files: %d"%(len(input_data)))
  return input_data

def create_logger(log_level, log_file=None):
  # Create a custom logger
  logger = logging.getLogger("TLTR")
  logger.setLevel(log_level)
  
  c_handler = logging.StreamHandler()
  c_handler.setLevel(log_level)
  
  # Create formatters and add it to handlers
  c_format = logging.Formatter('[%(name)s][%(levelname)s][%(message)s]')
  c_handler.setFormatter(c_format)
  
  # Add handlers to the logger
  logger.addHandler(c_handler)

  # Process only if log file defined
  if log_file is not None:
    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(logging.WARNING)
    f_format = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(message)s]')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
  return logger

def plot_label_dist(input_data):
  input_labels = []
  for des, data_pt in input_data.items():
    input_labels.extend(data_pt['state'])
  
  plt.figure()
  bins = np.arange(-1,max(input_labels)+1)+0.5
  plt.hist(input_labels,bins)
  plt.xticks(np.arange(0,max(input_labels)+1))

def train_loop(dataset, synth_chk_pt, plot=False): 
  batch_size =1
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  train_size = int(0.8*len(dataset))
  val_size = int(0.1*len(dataset))
  test_size = len(dataset) - val_size - train_size 
  train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,val_size,test_size])
  train_loader = torch.utils.data.DataLoader(train_dataset, 
                              batch_size=batch_size, shuffle=True)
  val_loader = torch.utils.data.DataLoader(val_dataset, 
                              batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset,
                              batch_size=batch_size, shuffle=False)
  #net = Net(dataset.num_classes) 
  net = torch.load(synth_chk_pt)
  net.to(torch.device("cpu"))
  logger_h.debug("Dataset: %d, Train: %d, valid: %d, test: %d"%(len(dataset), len(train_loader), len(val_loader), len(test_loader)))
  if  logger_h.getEffectiveLevel() <= logging.DEBUG :
    summary(net, input_size=(4,)+dataset.current_maps.shape[1:],device="cpu")
  net.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
  lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, LR_decay_rate)
  criterion = criterion.to(device)
  train_loss_results = [];
  val_loss_results = []
  epoch =0
  running_total = 0.0
  running_correct = 0.0
  running_train_loss = 0.0
  running_val_loss = 0.0
  inner_loop = tqdm(desc='Batches',unit='batch',total=len(train_loader),position=1) 
  outer_loop = tqdm(desc='Epochs',unit='epoch',total=epochs,position=0) 
  while epoch<epochs:
    for step, data in enumerate(train_loader):
      inputs, labels = (data[0].to(device,non_blocking=True), 
                        data[1].to(device,non_blocking=True))
      optimizer.zero_grad()
      outputs = (net.train())(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      _, predicted = torch.max(outputs, 1)
      with torch.no_grad():
        running_train_loss += loss.item()
        running_total += len(labels)
        running_correct += (predicted == labels).sum()
        inner_loop.set_postfix({'Train_Acc':"%05.2f"%(100*(running_correct/running_total)),
                                'Loss':"%5.3e"%(running_train_loss/(step+1))},refresh=False)
        inner_loop.update(1)
        if(step%len(train_loader) == len(train_loader)-1):
          train_loss_results.append(running_train_loss/(step+1))
          valid_total = 0.0
          valid_correct = 0.0
          val_loss = 0
          for j, data in enumerate(val_loader, 0):   
            inputs, labels = (data[0].to(device,non_blocking=True), 
                              data[1].to(device,non_blocking=True))
            outputs = (net.eval())(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            running_val_loss += loss.item()
            valid_total += len(labels)
            valid_correct += (predicted == labels).sum()
          outer_loop.set_postfix(Valid_Acc="%06.3f"%(100*(valid_correct/valid_total)),refresh=False)
          val_loss_results.append(val_loss/valid_total)
          outer_loop.update(1)
          epoch += 1
          lr_decay.step()
          if epoch<epochs :
            inner_loop.reset()
            inner_loop.refresh()
            running_total = 0.0
            running_correct = 0.0
            running_train_loss = 0.0
            running_val_loss = 0.0
  if plot:
    plt.figure()
    plt.plot(train_loss_results,label='train')
    plt.plot(np.array(val_loss_results)*4,label='test')
    plt.legend() 
  return net

if __name__ == '__main__':
  log_level = logging.DEBUG
  #log_level = logging.INFO
  data_dir = "./run/TL_data"
  syn_chk_pt = "./run/checkpoint/synth_CNN.ckp"
  tl_chk_pt = "./run/checkpoint/TL_CNN.ckp"
  main(data_dir, log_level,  syn_chk_pt, tl_chk_pt) 
