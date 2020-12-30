import math
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from matplotlib import cm 
from sklearn import preprocessing


mypath = os.getcwd()
output = "/content/Thesis/disentanglement_lib/data/"

df1 = pd.read_csv("/content/Thesis/disentanglement_lib/csv/AER/CASchools.csv")
counter = 0
# number of data points
num_data_points = 0

marker = ["o","s","v","p","P","h","H","X","D","d"]

evenly_spaced_interval = np.linspace(0, 1, 5)
colors = [cm.rainbow(x) for x in evenly_spaced_interval]
graph_features = pd.DataFrame(columns=['pos' , 'size' , 'shape','color'])
# draw the plot
'''
for f in range(0,25):
  num_data_points += 5
  print(f)
  for a in range(0,10):
    counter += 1
    x = [random.gauss(0.5, 0.25)  for i in range(num_data_points)]
    y = [random.gauss(0.5, 0.25) for i in range(num_data_points)]
    for b in range(1000,3001):
        if (b % 500) == 0 :
            for i, d in enumerate(colors):
              for c in range(len(marker)):
                    name = str(f) + '-pos-' + str(a) + '-size-' + str(b) + '-shape-' + str(c) + '-color-' + str(i) + '.png'
                    plt.figure(figsize=(6,6))
                    plt.scatter(x, y, s = b, marker = marker[c], c = np.array([d]))
                    plt.axis([0.0, 1.0, 0.0, 1.0])
                    plt.tight_layout()
                    plt.savefig( output + "scatt/" + name, dpi=10.7)
                    #plt.show()
                    plt.close('all')
                    graph_features = graph_features.append( {'pos':counter -1 , 'size':(int(b/500)-2) ,'shape':c, 'color' : i}, ignore_index=True)

#print(graph_features)
graph_features.to_csv(output + 'output.csv',index = False, header=False)
'''


for i in range(1,len(df1.columns)):
  if df1.iloc[:, i].dtype.name == 'int64' or df1.iloc[:, i].dtype.name == 'float64':
    x = preprocessing.normalize([df1.iloc[:, i]])
    for t in range (i, len(df1.columns)):
      if df1.iloc[:, t].dtype.name == 'int64' or df1.iloc[:, t].dtype.name == 'float64':
        y = preprocessing.normalize([df1.iloc[:, t]])
        plt.figure(figsize=(6,6))
        plt.scatter(x, y)
        plt.axis([0.0, 1.0, 0.0, 1.0])
        plt.tight_layout()
        plt.savefig( output + "scatt/" + str(i) + '-'+ str(t), dpi=10.7)
        #plt.show()
        plt.close('all')
