"""
This file contains helper functions.
__author__ : Arsene I. Muhire
"""

import argparse,time

def settings_arg_parser():
    # Define a parser
    parser = argparse.ArgumentParser(description="ModelTrainer Settings") 
    parser.add_argument('--data_directory', 
                        type=str, 
                        help='Folder with images to train the model on' ) 
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Checkpoint save directory' ) 
    parser.add_argument('--arch', 
                        type=str, 
                        help='architecture to be used resnet or vgg') 
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='training learning rate') 
    parser.add_argument('--hidden_units', 
                        type=str, 
                        help='Hidden units') 
    parser.add_argument('--epochs', 
                        type=int, 
                        help='number of epochs') 

    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Point to checkpoint file as str.') 
    
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Choose top K matches as int.')
     
    parser.add_argument('--category_names', 
                        type=str, 
                        help='path to json file with category and their names.')
 
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU device')

    # Parse args
    
    return parser
#Pretty display helper functions
def displayDuration(task,start_time):
    elapsed_secs=time.time()-start_time
    print(task+" duration is {} minutes {:.2f} seconds".format(elapsed_secs//60,elapsed_secs%60))
    
def draw_line():
    print("_"*85) 

def get_padding(steps):
    if steps<10:
        return "  "
    elif steps<100:
        return " "
    else:
        return ""
    

def display_header(epoch,EPOCHS):
    draw_line()
    print(f"                         Epoch ..::{epoch+1} of {EPOCHS} ::..")
    draw_line() 
    print("| EPOCH         ",
          "| STEP          ",
          "| TRAIN LOSS    ",
          "| VAL LOSS      ",
          "| VAL ACCURACY |")
    draw_line()
