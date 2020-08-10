# the great stream of imports
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from collections import OrderedDict
from support_functions import *
from model_functions import *


# Main program function defined below
def main():
    data_dir = 'flowers'
    model_save_dir = 'models'
    model = 'vgg16'
    learning_rate = 0.01
    #input_size = 
    hidden_units = [1024, 512]
    output_size = 102
    epochs = 7
    device = 'cpu'
    # get input arguments
    in_arg = get_input_args_train()
    model = in_arg.arch
    model_save_dir = in_arg.save_dir
    #hidden_units = in_arg.hidden_units
    
    # validate the training inputs
    
    # process --GPU argument
    data_dir = in_arg.dir
    
    print(in_arg.gpu)
    device = set_device(in_arg.gpu)
    epochs = in_arg.epochs
    arch = in_arg.arch
    print(device)
    
    # prepare the data loaders
    trainloader, testloader, validloader, label_idx = prepare_date(data_dir)
    
    #create the new model, return model, optimizer and criterion
    model, optimizer, criterion = create_model(arch, hidden_units, output_size, learning_rate)
    
    # retrive the model set the optimizer and criterion
    #model, optimizer, criterion = load_checkpoint(model_save_dir, model, learning_rate)
    
    print(model)
   
    # train the network
    train_model(model, trainloader, validloader, device, epochs, optimizer, criterion)
    
    
    # validate the network
    test_model(model, validloader, device, criterion)
    
    # save the network 
    checkpoint_name = 'checkpoint1.pth'
    checkpoint_dir = 'models'
    save_model(model, optimizer, hidden_units, output_size, epochs, checkpoint_name, checkpoint_dir, label_idx, arch)
    
    
    
    
    
main()    
    