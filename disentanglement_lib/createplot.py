import math
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from matplotlib import cm 

mypath = os. getcwd()
output = "/content/Thesis/disentanglement_lib/data/"



# number of data points

num_data_points = 5
marker = ["o","s","v","p","P","*","h","H","X","D","d","^","<",">","8",'$...$']

evenly_spaced_interval = np.linspace(0, 1, 20)
colors = [cm.rainbow(x) for x in evenly_spaced_interval]
graph_features = pd.DataFrame(columns=['size' , 'shape','color'])
# draw the plot

for t in range(1,6):
  x = [random.gauss(0.5, 0.25)  for i in range(num_data_points)]
  y = [random.gauss(0.5, 0.25) for i in range(num_data_points)]
  for b in range(1,3001):
      if (b % 150) == 0 :
          for i, d in enumerate(colors):
            for c in range(len(marker)):
                  name = str(t) + '-size-' + str(b) + '-shape-' + str(c) + '-color-' + str(i) + '.png'
                  plt.figure(figsize=(6,6))
                  plt.scatter(x, y, s = b, marker = marker[c], c = np.array([d]))
                  plt.axis([0.0, 1.0, 0.0, 1.0])
                  plt.tight_layout()
                  plt.savefig( output + "scatt/" + name, dpi=10.7)
                  #plt.show()
                  plt.close('all')
                  graph_features = graph_features.append( {'size':(int(b/100)-1) ,'shape':c, 'color' : i}, ignore_index=True)
  num_data_points += 10
#print(graph_features)
graph_features.to_csv(output + 'output.csv',index = False, header=False)



