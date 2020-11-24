import math
import random
import matplotlib.pyplot as plt
import os

mypath = os. getcwd()
output = os. getcwd()+"/output1/"

# number of data points

num_data_points = 25
color = ['r','g','b','c','m','y','k']
marker = ['o','x','s','.','*','D']

# draw the plot

for a in range(0,10):
    x = [random.gauss(0.5, 0.25)  for i in range(num_data_points)]
    y = [random.gauss(0.5, 0.25) for i in range(num_data_points)]
    for b in range(1,800):
        if (b % 50) == 0 :
            for c in range(len(marker)):
                for d in range(len(color)):
                    plt.figure(figsize=(6,6))
                    plt.scatter(x, y, s = b, marker = marker[c], c = color[d])
                    plt.axis([0.0, 1.0, 0.0, 1.0])
                    plt.tight_layout()
                    plt.savefig(output + str(a) + '-' + str(b) + '-' + str(c) + '-' + str(d) + '.png', dpi=21.4)
                    #plt.show()
                    plt.close('all')