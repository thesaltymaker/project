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
    hidden_units = [1024, 512]
    output_size = 102
    epochs = 7
    device = 'cpu'
    
    # get input arguments
    in_arg = get_input_args_train()
    model = in_arg.arch
    model_save_dir = in_arg.save_dir
    hidden_units = in_arg.hidden_units
    data_dir = in_arg.dir
    device = set_device(in_arg.gpu)
    epochs = in_arg.epochs
    arch = in_arg.arch
    learning_rate = in_arg.learning_rate
    
    
    print("Training Platform:")
    print(device)
    
    # prepare the data loaders
    trainloader, testloader, validloader, label_idx = prepare_date(data_dir)
    
    #create the new model, return model, optimizer and criterion
    model, optimizer, criterion = create_model_gen(arch, hidden_units, output_size, learning_rate)
    
    print("The Model:")
    print(model)
   
    # train the network
    train_model(model, trainloader, validloader, device, epochs, optimizer, criterion)
    
    
    # validate the network
    test_model(model, validloader, device, criterion)
    
    # save the network 
    save_model(model, optimizer, hidden_units, output_size, epochs, model_save_dir, label_idx, arch)
    
    
    
    
    
main()    
    