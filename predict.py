import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from collections import OrderedDict
from support_functions import *
from model_functions import *
from predict_functions import *

# Main program function defined below
def main():
    
    #set defaults
    device = 'cpu'
    checkpoint_name = 'checkpoint.pth'
    topk = 5
    category_name = 'cat_to_name.json'
    
    # get input arguments
    in_arg = get_input_args_predict()
    
    #process args
    path_to_image = in_arg.path_to_image
    checkpoint_name =  in_arg.checkpoint + '.pth'
    topk = in_arg.top_k
    category_name = in_arg.category_name
    device = in_arg.gpu
    
    
    # process --GPU argument
    device = set_device(device)
    print(device)
    
    # read the default or inouted json file
    with open(category_name, 'r') as f:
        cat_to_name = json.load(f)
    
    
    # load the model from the input
    model, optimizer, criterion = load_checkpoint_predict(checkpoint_name)
    print(model)
    
       
    # make the prediction and match clases to name from json file
    probs, classes = predict(path_to_image, model, topk, device)

    
    # print the results
    print_results(probs, classes, topk, cat_to_name, model.idx_to_class)
    
main()
    
    
    
    
    
    