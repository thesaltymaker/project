import torch

from torchvision import models

import time
import json
from collections import OrderedDict

from PIL import Image
import numpy as np


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
        # open and resize the image
    PIL_image = Image.open(image)
    PIL_image = PIL_image.resize((256, 256))
    PIL_image = PIL_image.crop((16, 16, 240, 240))
        
    # convert to numpy array
    np_image = np.array(PIL_image)
    
    # Normalize the np_image
    np_image = np_image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
 
    # transpose the color channel in 3rd dimension to 1st dimension 
    np_image = np.transpose(np_image,(2, 0, 1))
    
    #Citation Ask the mentors Arun R https://knowledge.udacity.com/questions/282939
    np_image_tensor = torch.from_numpy(np_image)
    
    return np_image_tensor
    
    
    
    
    
    
def predict(image_path, model, top, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    # process the image
    image_tensor = process_image(image_path)
    
    # Citation for unsqueeze: https://jbencook.com/adding-a-dimension-to-a-tensor-in-pytorch/
    image_tensor = image_tensor.unsqueeze_(0).float()
        
    # Execute the model
    model.eval()
    labels = model.class_to_idx
    
    # apply input and labels to device, run model
    img_in = image_tensor.to(device)
    output = model.forward(img_in)
    
    #reverse output to percentages and get topk 5
    output_tensor = torch.exp(output)
    probs, classes= output_tensor.topk(top, dim=1)
    
    return probs, classes
    
    
    
    
def print_results(probs, classes, topk, cat_to_name):
  
    print(probs)
    print(classes)

    #convert the tensors and reverse the output for probs and classes
#    np_probs = np.flipud(probs[0].detach().cpu().numpy())
#    np_classes = np.flipud(classes[0].detach().cpu().numpy())
#    for i in range (topk):
#        print(str(i) + " " + str(np_probs[i]) +" " + str((cat_to_name.get(str(np_classes[i])))))
    np_probs = np.array(probs[0].detach().cpu().numpy())
    np_classes = np.array(classes[0].detach().cpu().numpy())    
    for i in range (topk):
        print(str(i) + " " + str(np_probs[i]) +" " + str((cat_to_name.get(str(np_classes[i])))))


    #print (cat_to_name.get(str(np_classes[2])) )

    #print (cat_to_name)
#fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
#ax1.axis('off')
#ax2.barh(np.arange(5), np_probs)
#ax2.set_aspect(0.1)
#ax2.set_yticks(np.arange(5))
#ax2.set_yticklabels([cat_to_name.get(str(np_classes[0])),
#            cat_to_name.get(str(np_classes[1])),
#            cat_to_name.get(str(np_classes[2])),
#            cat_to_name.get(str(np_classes[3])),
#            cat_to_name.get(str(np_classes[4]))], 
#            size='medium');
#ax2.set_title('Class Probability')
#ax2.set_xlim(0, 1.1)#

#plt.tight_layout()
