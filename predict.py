"""
This program allows users to make predictions on input images.
__author__ : Arsene I. Muhire
"""
from utils import settings_arg_parser, displayDuration
import numpy as np 
import torch
from torch import utils,nn,optim 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets,transforms,models 
import json 
from PIL import Image

class ModelPredicter():
    def __init__(self,image_abs_path,checkpoint,arch,with_device,cat_to_name):

        self.checkpoint_filename=checkpoint
        self.arch=arch 
        self.device=with_device
        self.image_abs_path=image_abs_path
        with open(cat_to_name, 'r') as f:
            self.cat_to_name = json.load(f)

        self.model= self.load_model_from_checkpoint() 
                



    def load_model_from_checkpoint(self): 
        """
        Loads model checkpoint.
        """
        
        checkpoint = torch.load(self.checkpoint_filename)

        if self.arch == "resnet":
            model = models.resnet18(pretrained=True)
        elif self.arch =="vgg":
            model = models.vgg19(pretrained=True)  
        
        #Freeze training for all these model layers
        for param in model.parameters():
            param.requires_grad = False
            
        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])

        
        return model



    def process_image(self,image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        image = Image.open(image)
        image = image.resize((256,256))
        pad = (256-224)/2
        image = image.crop((pad,pad,256-pad,256-pad))
        image = np.array(image)/255

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std 
        image = image.transpose(2,0,1) 
        return image


    def predict(self,top_k): 
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        ''' 
        
        image_p=self.process_image(self.image_abs_path)
        self.model.eval()
        image = torch.from_numpy(np.expand_dims(image_p,axis=0)).type(torch.FloatTensor).to(self.device)
    
        #k probabilities
        probs = torch.exp(self.model.forward(image))
    
        k_probs = probs.topk(top_k)[0]
        k_labels = probs.topk(top_k)[1] 
        k_probs = np.array(k_probs.detach())[0]
        k_labels = np.array(k_labels.detach())[0] 
        
        idx_to_class = {val: key for key, val in    
                                        self.model.class_to_idx.items()}
        k_labels = [idx_to_class[lb] for lb in k_labels]
        k_flowers = [self.cat_to_name[lb] for lb in k_labels] 
        return k_probs, k_labels, k_flowers 


    def display_top_k_classes(self,top_k): 
        # TODO: Display an image along with the top k classes
        self.model.eval()
        self.model.to(self.device)  
        probs,lbls,flowers = self.predict(top_k)  
        
        print(f"Flower -  Probability")
        for i in range(len(flowers)):
            print(f"{flowers[i]} - {probs[i]:.2f}")

 


if __name__=="__main__": 
     
    args = settings_arg_parser()    
    parser=settings_arg_parser()
    parser.add_argument("input",help="absolute image path")
    args= parser.parse_args()


    if args.gpu:
        device='cuda'
    else:
        device='cpu'


    predicter = ModelPredicter(
        image_abs_path=args.input,
        checkpoint=args.checkpoint if args.checkpoint else 'my_checkpoint.pth',
        arch=args.arch if args.arch else 'vgg',
        with_device=device,
        cat_to_name=args.category_names if args.category_names else 'cat_to_name.json'

    )
    
     
    if args.top_k is not None:
        predicter.display_top_k_classes(top_k=args.top_k) 
    else:
        predicter.display_top_k_classes(1)
    
 