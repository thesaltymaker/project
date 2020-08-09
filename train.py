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
    epochs = 7
    device = 'cpu'
    # get input arguments
    in_arg = get_input_args_train()
    model = in_arg.arch
    model_save_dir = in_arg.save_dir
    
    # validate the training inputs
    
    # process --GPU argument
    data_dir = in_arg.dir
    
    print(in_arg.gpu)
    device = set_device(in_arg.gpu)
    
    print(device)
    
    # prepare the data loaders
    trainloader, testloader, validloader = prepare_date(data_dir)
    
    
    # retrive the model set the optimizer and criterion
    model, optimizer, criterion = load_checkpoint(model_save_dir, model, learning_rate)
    
       
    # train the network
    #train_model(model, trainloader, device, epochs)
    
    
    # validate the network
    #test_model(model, validloader, device)
    
    # save the network 
    #save_model(model, optimizer, input_size, hidden_sizes, output_size, epochs, checkpoint_name, checkpoint_dir)
    
    
    
    
    
main()    
    