#this file contains model support functions
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from collections import OrderedDict


#create transforms
def prepare_date(data_dir):
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])])

    # DONE: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)

    # DONE: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    #debug
    label_idx = train_data.class_to_idx
    print(label_idx)
    return trainloader, testloader, validloader, label_idx

#create a new classifier model
def create_model(model_name, hidden_sizes, output_size, learning_rate):

    #    # citation for using --arch to load the corect model: https://knowledge.udacity.com/questions/262667
    model = getattr(models, model_name)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    input_size = model.classifier[0].in_features
    # Build a feed-forward network for classifier 
    # Citaton help from ask a mentor: https://knowledge.udacity.com/questions/295613
    classifier = nn.Sequential(OrderedDict([
         ('inputs', nn.Linear(input_size, hidden_sizes[0])),
         ('relu1', nn.ReLU()),
         ('dropout1', nn.Dropout(p=0.3, inplace=False)),
         ('hidden_layer1', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
         ('relu2', nn.ReLU()),
         ('dropout2', nn.Dropout(p=0.2, inplace=False)),
         ('hidden_layer2', nn.Linear(hidden_sizes[1], output_size)),
         ('relu3', nn.ReLU()),
         ('Softmax1', nn.LogSoftmax(dim=1)) ]))
    model.classifier = classifier

    # Set criterion and optimizer paramters
    criterion = nn.CrossEntropyLoss() #for use with softmax
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)
    
    return model, optimizer, criterion




#load model for trianing
def load_checkpoint_train(filepath, model_name, learning_rate):
    print(filepath+"/checkpoint.pth")
    checkpoint = torch.load(filepath+"/checkpoint.pth")
    
    #model = models.vgg16(pretrained=True)
    
    # citation for using --arch to load the corect model: https://knowledge.udacity.com/questions/262667
    model = getattr(models, model_name)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    # Build a feed-forward network for classifier 
    # Citaton help from ask a mentor: https://knowledge.udacity.com/questions/295613
    classifier = nn.Sequential(OrderedDict([
         ('inputs', nn.Linear(checkpoint['input_size'], checkpoint['hidden_sizes'][0])),
         ('relu1', nn.ReLU()),
         ('dropout1', nn.Dropout(p=0.3, inplace=False)),
         ('hidden_layer1', nn.Linear(checkpoint['hidden_sizes'][0],checkpoint['hidden_sizes'][1])),
         ('relu2', nn.ReLU()),
         ('dropout2', nn.Dropout(p=0.2, inplace=False)),
         ('hidden_layer2', nn.Linear(checkpoint['hidden_sizes'][1], checkpoint['output_size'])),
         ('relu3', nn.ReLU()),
         ('Softmax1', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    model.idx_to_class = checkpoint['idx_to_class']

    # Set criterion and optimizer paramters
    criterion = nn.CrossEntropyLoss() #for use with softmax
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, criterion

# load model for prediction   
def load_checkpoint_predict(checkpoint_name):
    print(checkpoint_name)
    checkpoint = torch.load(checkpoint_name)
    model = checkpoint['model']
    #model = models.vgg16(pretrained=True)
    
    # citation for using --arch to load the corect model: https://knowledge.udacity.com/questions/262667
    #model = getattr(models, model_name)(pretrained=True)
    #for param in model.parameters():
    #    param.requires_grad = False
    
    # Build a feed-forward network for classifier 
    # Citaton help from ask a mentor: https://knowledge.udacity.com/questions/295613
    #classifier = nn.Sequential(OrderedDict([
    #     ('inputs', nn.Linear(checkpoint['input_size'], checkpoint['hidden_sizes'][0])),
    #     ('relu1', nn.ReLU()),
    #     ('dropout1', nn.Dropout(p=0.3, inplace=False)),
    #     ('hidden_layer1', nn.Linear(checkpoint['hidden_sizes'][0],checkpoint['hidden_sizes'][1])),
    #     ('relu2', nn.ReLU()),
    #     ('dropout2', nn.Dropout(p=0.2, inplace=False)),
    #     ('hidden_layer2', nn.Linear(checkpoint['hidden_sizes'][1], checkpoint['output_size'])),
    #     ('relu3', nn.ReLU()),
    #     ('Softmax1', nn.LogSoftmax(dim=1))]))
    model.classifier = checkpoint['classifier']
    #model = nn.Sequential(checkpoint['classifier'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    model.idx_to_class = checkpoint['idx_to_class']

    # Set criterion and optimizer paramters
    criterion = nn.CrossEntropyLoss() #for use with softmax
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, criterion

#save model
#print("Our model: \n\n", model, '\n
def save_model(model, optimizer, hidden_sizes, output_size, epochs, checkpoint_name, checkpoint_dir, label_idx, model_name):
    print("The state dict keys: \n\n", model.state_dict().keys())
    torch.save(model.state_dict(), 'checkpoint.pth')
    print("The oprimizer state dict keys: \n\n", optimizer.state_dict().keys())
    torch.save(optimizer.state_dict(), 'checkpoint.pth')
    model.class_to_idx = label_idx
    input_size = model.classifier[0].in_features
    #Citation: https://stackoverflow.com/questions/483666/reverse-invert-a-dictionary-mapping
    model.idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    checkpoint = {'input_size': input_size,
              'output_size': output_size,
              'classifier': model.classifier,
              'hidden_sizes': hidden_sizes,
              'model_state_dict': model.state_dict(),
              'model_state_dict_keys': model.state_dict(),
              'epochs': epochs,
              'optimizer_state_dict': optimizer.state_dict(),
              'optimizer_state_dict_keys': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'idx_to_class': model.idx_to_class, 
              'model': model}

    torch.save(checkpoint, checkpoint_dir + '/checkpoint_' + model_name + '.pth' )
    return


#validate model
def validation(model, validloader, criterion, device):
    test_loss = 0
    accuracy = 0
    model.to(device)

    for images, labels in validloader:
        images.resize_(images.size()[0], 3, 224, 224)
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy


#train model
#Train the classifier
def train_model(model, trainloader, validloader, device, epochs, optimizer, criterion):
    print(str (epochs) )
    model.to(device)
    print_every = 20
    steps = 0
    running_loss = 0
    for e in range(epochs):
        model.train() 
        for images, labels in trainloader:
            steps +=1   
        
            optimizer.zero_grad()
            images.resize_(images.size()[0], 3, 224, 224)
            # Move input and label tensors to the GPU
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            start = time.time()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/steps))
    
        print(f"Device : {device}; Training Time per batch: {(time.time() - start)/len(trainloader):.3f} seconds")
        print(f"Training Time per epoch: {(time.time() - start):.3f} seconds")
    
        start = time.time()
        # Make sure network is in eval mode for inference
        model.eval()
          
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            test_loss, accuracy = validation(model, validloader, criterion, device)
                
        print("Epoch: {}/{}.. ".format(e+1, epochs),
            " Training Loss: {:.3f}. ".format(running_loss/len(trainloader)),
            " Validation Loss: {:.3f}. ".format(test_loss/len(validloader)),
            " Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

        print(f"Device : {device}; Validating Time per batch: {(time.time() - start)/len(validloader):.3f} seconds")
        print(f"Validating Time per epoc: {(time.time() - start):.3f} seconds")
    
        running_loss = 0
        steps = 0        
    return model


#test model
# DONE: Do validation on the test set
def test_model(model, testloader, device, criterion):
    test_loss = 0
    accuracy = 0
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    # Turn off gradients for validation, saves memory and computation
    start = time.time()
    
    with torch.no_grad():
        test_loss, accuracy = validation(model, testloader, criterion, device)

    print(f"Device: {device}; Testing Time per batch: {(time.time() - start)/len(testloader):.3f} seconds")
    print(f"Total Testing Time: {(time.time() - start):.3f} seconds")
    print( "Test set batches: {:.0f}".format(len(testloader)),
        "   Test Loss: {:.3f} ".format(test_loss/len(testloader)),
        "   Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
    return


