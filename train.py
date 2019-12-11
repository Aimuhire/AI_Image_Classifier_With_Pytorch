"""
This program is used to train a model using an image dataset
__author__ : Arsene I. Muhire
"""
from utils import settings_arg_parser,display_header,get_padding,draw_line,displayDuration
import numpy as np 
import torch
from torch import utils,nn,optim 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets,transforms,models
import time  
import sys

class ModelTrainer():
    def __init__(self,data_dir='flowers',with_device="cpu",
    checkpoint='my_checkpoint.pth', 
    arch='vgg',
    hidden_units=8192,
    epochs=5,
    learning_rate=0.001,
    save_dir=None):

        self.train_dir = data_dir + '/train'
        self.valid_dir = data_dir + '/valid'
        self.test_dir = data_dir + '/test'
        self.checkpoint = checkpoint
        self.learning_rate =learning_rate
        self.hidden_units=hidden_units
        self.arch=arch
        self.batch_size=32
        self.save_dir=save_dir
        self.EPOCHS =epochs 
        self.device=torch.device(with_device) 
        self.prepare_data_loaders()
        self.prepare_neural_net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.classifier.parameters(),lr=self.learning_rate)

        

    def prepare_data_loaders(self):
        train_transforms = transforms.Compose([
                                       transforms.RandomHorizontalFlip(), 
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                       
        val_transforms = transforms.Compose([transforms.Resize(250),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(), 
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    
        test_transforms = transforms.Compose([transforms.Resize(250),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),  
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                             
        self.train_set = datasets.ImageFolder(self.train_dir,transform=train_transforms)
        self.val_set =  datasets.ImageFolder(self.valid_dir,transform=val_transforms)
        self.test_set = datasets.ImageFolder(self.test_dir,transform=test_transforms) 

        self.train_loader = utils.data.DataLoader(self.train_set,batch_size=self.batch_size,shuffle=True)
        self.val_loader = utils.data.DataLoader(self.val_set,batch_size=self.batch_size)
        self.test_loader = utils.data.DataLoader(self.test_set,batch_size=self.batch_size) 
    

    def prepare_neural_net(self):
        #Load pre trained model
        if self.arch == "resnet":
            self.model = models.resnet18(pretrained=True)
        elif self.arch =="vgg":
            self.model = models.vgg19(pretrained=True) 

        #Freeze training for  these model layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.input_size = self.model.classifier[0].in_features
        output_size = 102
        classifier = nn.Sequential(nn.Linear(self.input_size,self.hidden_units, bias=True),
                            nn.ReLU(),
                            nn.Dropout(p=0.4),      
                            nn.Linear(self.hidden_units,output_size, bias=True),
                            nn.LogSoftmax(dim=1)  )  

        self.model.classifier=classifier 
        
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        self.model = self.model.to(self.device)
         


    def validation(self, model, test_loader, criterion):
        test_loss = 0
        accuracy = 0
        
        for images, labels in test_loader:
            
            images = images.to(self.device) 
            labels = labels.to(self.device) 
            
            output = model.forward(images)
            test_loss += criterion(output, labels).item()
            
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        
        return test_loss, accuracy


    def train(self):
        #Do the training 
        print("Training on device:  ",self.device)

        start_time=time.time()
        track_step=50
        running_loss=0
        for epoch in range(self.EPOCHS):
            self.model.train()
            steps=0 
            
            
            display_header(epoch,self.EPOCHS)
            
            for images,labels in self.train_loader: 
                images=images.to(self.device)
                labels=labels.to(self.device)  
                steps+=1 
                self.optimizer.zero_grad() 

                output = self.model.forward(images)
                loss = self.criterion(output, labels)

                loss.backward()
                self.optimizer.step()
                running_loss+=loss.item()
                
                
                if steps%track_step==0: 
                    self.model.eval()
                    with torch.no_grad():
                        val_loss, accuracy = self.validation(self.model, self.val_loader, self.criterion) 
                    
                    pad=get_padding(steps)
                    print(f"      {epoch+1}       ", # epoch
                        f"      {pad}{steps}/{len(self.train_loader)}       ", # step
                        f"      {running_loss/track_step :.2f}       ",# training loss
                        f"      {val_loss/len(self.test_loader):.2f}       ",# validation loss
                        f"      {accuracy/len(self.test_loader):.2f}      ")# validation accuracy
                    draw_line()
                    self.model.train()
                    running_loss=0
                    
            displayDuration("Epoch "+str(epoch+1),start_time) 
        displayDuration("Training ",start_time) 

        #Save Checkpoint
        self.model.class_to_idx = self.train_set.class_to_idx
        checkpoint = {'name':'Flower Classifier Model',  
              'epoch':self.EPOCHS,
              'optimizer': self.optimizer.state_dict(),
              'input_size': self.input_size,
              'output_size': 102,
              'pretrained_arch': 'vgg19', 
              'learning_rate': self.learning_rate,
              'batch_size': self.batch_size,
              'classifier': self.model.classifier,
              'class_to_idx': self.model.class_to_idx,
              'state_dict':self.model.state_dict()}



        torch.save(checkpoint, self.save_dir+"/"+self.checkpoint)
            
    
        
if __name__=="__main__":

    parser=settings_arg_parser()
    parser.add_argument("data_dir",help="directory of flowers")
    args= parser.parse_args()

    if args.gpu:
        device='cuda'
    else:
        device='cpu'

    print("Data Directory: ",args.data_dir)
    trainer = ModelTrainer(data_dir=args.data_dir,
    checkpoint=args.checkpoint if args.checkpoint else 'my_checkpoint.pth',
    with_device=device,
    arch=args.arch if args.arch else 'vgg',
    hidden_units=args.hidden_units if args.hidden_units else 8192,
    epochs=args.epochs if args.epochs else 5,
    save_dir=args.save_dir if args.save_dir else '')
    trainer.train()
    
        
        