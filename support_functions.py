import argparse
import torch
import json


def get_input_args_train():
    """
    Retrieves and parse train.py command line args provided by the user when
    run the program from a terminal window. If the user fails to provide 
    arguments, then default values are used.
    Command Line Arguments:
      1. Data directory --dir with default value 'flowers'
      2. Save Model Directory --save_dir with default value 'models'
      3. Model Architecture as --arch with default value 'vgg16'
      4. Learning Rate as --learning_rate with default vaule 0.01
      5. Hidden Units as --hidden_units with default as 512
      6. Epochs as --epochs with default value 7
      7. GPU toggle --GPU to move model trainign to the GPU CUDA cores, default is CPU
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
     
    Usage examples:
     python train.py data_dir
     python train.py data_dir --save_dir save_directory
     python train.py data_dir --arch "vgg16"
     python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
     python train.py data_dir --gpu 
    
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Create 7 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('dir', type = str, default = 'flowers',
                    help = 'path to the data directory, default "flowers"' ) 
    parser.add_argument('--save_dir', type = str, default = 'models', 
                    help = 'save_dir for model, default "models/"') 
    parser.add_argument('--arch', type = str, default = 'vgg16', 
                    help = 'Architeture model supports vgg13 and vgg16, default  "vgg16"') 
    parser.add_argument('--learning_rate', type=float, default = '0.01', 
                    help = 'learning_rate, tune per architecture choice, default "0.01"') 
    parser.add_argument('--hidden_units', nargs='+', type=int, default = [1024, 512], 
                    help = 'hidden_units list , default [1024, 512]') 
    parser.add_argument('--epochs', type = int, default = '7', 
                    help = 'number of epochs to train model, default 7') 
    parser.add_argument('--gpu', action = 'store_true', default = False,
                    help = 'Toggle to turn on GPU for prediction, default is "CPU"')  
    return parser.parse_args()
                        
                        
def get_input_args_predict():
    """
    Retrieves and parse predict.py command line arguments provided by the user when
    run the program from a terminal window. If the user fails to provide arguments, 
    default values are used.
    Command Line Arguments:
      1. Path to image 
      2. Model checkpoint name 
      3. --topk Number of top predictions to display
      4. --category_name Category JSON file
      5. --GPU Toggle the GPU on for prediction
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
     
    Usage examples:
     python predict.py /path/to/image checkpoint
     python predict.py input checkpoint --top_k 3
     python predict.py input checkpoint --category_names cat_to_name.json
     python predict.py input checkpoint --gpu
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create 5 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('path_to_image', type = str,
                    help = 'path to image, default "upload/"') 
    parser.add_argument('checkpoint', type = str, default = 'checkpoint',
                    help = 'path to the data directory, default "checkpoint"') 
    parser.add_argument('--top_k', type = int, default = '5', 
                    help = 'Number of top predictions, default  "5"') 
    parser.add_argument('--category_name', type = str, default = 'cat_to_name.json', 
                    help = 'JSON file with mapping of index to category, default "cat_to_name"') 
    parser.add_argument('--gpu', action = 'store_true', default = False,
               help = 'Toggle to turn on GPU for prediction, default is "CPU"') 
                        
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()         

# Determine if CUDA is avaiable                        
def set_device(gpu_flag): 
    if(gpu_flag):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        #print("GPU is not available on this platform. check configuration and hardware")
    #print(device)
    return device
                        