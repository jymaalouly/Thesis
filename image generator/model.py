import sys
import os
sys.path.insert(1, '../disentangling-vae-master')
from disvae.utils.modelIO import load_model, load_metadata
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import utils.visualize_1 as vis
from skimage.io import imread
import numpy as np
from torchvision import transforms, datasets
from PIL import Image 
import PIL.ImageOps  

class Traversal:
    def __init__(self):
        model_dir = "C:/Users/Tasiko/Thesis/disentangling-vae-master/results/btcvaescatt-20"
        model = load_model(model_dir)
        self.viz = vis.Visualizer(model=model, model_dir = model_dir,
                 loss_of_interest='kl_loss_')
        self.sample_path= "C:/Users/Tasiko/Thesis/image generator/temp/temp_image/temp_1.png"
    def create_reconstrution(self,slid):
        img = imread(self.sample_path, as_gray=True)
        img = img.reshape(img.shape + (1,)).astype(np.float32) / 255.
        transform=transforms.ToTensor()
        sample = transform(img)
        self.viz.reconstruct_image(sample,slid)
        
    def pre_processing(self, image_path):
        image_file = Image.open(image_path)# open colour image
        image_file = image_file.convert('L')
        image_file_black = PIL.ImageOps.invert(image_file)
        image_file_black = image_file_black.convert('1', dither=Image.NONE) # convert image to black and white
        image_file_black = image_file_black.resize((64, 64), Image.ANTIALIAS)
        image_file_black.save(self.sample_path)