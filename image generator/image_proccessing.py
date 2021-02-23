import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

# select func
class Generate():
    def __init__(self,path,num_images,num_points):
        n = num_points
        self.img = cv2.imread(path);
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY);
        # lets take a sample
        for num_image in range(num_images):
            self.h, self.w = self.img.shape;
            self.sx, self.sy = self.populate();
            samples = []
            for a,b in zip(self.sx,self.sy):
                samples.append([a,b])
            sx_sample = []
            sy_sample = []
            
            for x in range(n):
                tempx = random.choice(samples)
                sx_sample.append(tempx[0])
                sy_sample.append(tempx[1])

            side = None;
            if self.h > self.w:
                side = self.h;
            else:
                side = self.w;

            # make a square graph
            fig, ax = plt.subplots();
            ax.scatter(sx_sample, sy_sample, s = 4);
            ax.set_xlim((0, side));
            ax.set_ylim((0, side));
            x0,x1 = ax.get_xlim();
            y0,y1 = ax.get_ylim();
            ax.set_aspect(abs(x1-x0)/abs(y1-y0));
            fig.savefig(os.getcwd()+"/temp/generated_images/temp_" + str(num_image) + ".png", dpi=100);
            plt.close('all')
        
    def selection(self,value):
        return value**3 >= random.randint(0, 255**3);

    # populate the sample
    def populate(self):


        # go through and populate
        sx = [];
        sy = [];
        for y in range(0, self.h):
            for x in range(0, self.w):
                val = self.img[y, x];

                # use intensity to decide if it gets in
                # replace with what you want this function to look like
                if self.selection(val):
                    sx.append(x);
                    sy.append(self.h - y); # opencv is top-left origin
        return sx, sy;


    # I'm using opencv to pull the image into code, use whatever you like
    # matplotlib can also do something similar, but I'm not familiar with its format


