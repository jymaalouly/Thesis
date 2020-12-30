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

df = pd.read_csv("/content/Thesis/disentanglement_lib/csv/datasets/AirPassengers.csv", index_col=0)
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
count = 0 
for index1, filename in enumerate(os.listdir('/content/Thesis/disentanglement_lib/csv/datasets')):
    if filename.endswith('.csv'):
      count += 1
      df = pd.read_csv("/content/Thesis/disentanglement_lib/csv/datasets/"+filename, index_col=0)
      for index2, column in enumerate(df):
        if df[column].dtype.name == 'int64' or df[column].dtype.name == 'float64':
          min_max_scaler = preprocessing.MinMaxScaler()
          x_scaled = min_max_scaler.fit_transform(df[[column]])
          x = pd.DataFrame(x_scaled)
          for index3, column1 in enumerate(df):
            if column != column1:
              if df[column1].dtype.name == 'int64' or df[column1].dtype.name == 'float64':
                y_scaled = min_max_scaler.fit_transform(df[[column1]])
                y = pd.DataFrame(y_scaled)
                plt.figure(figsize=(6,6))
                plt.scatter(x, y)
                plt.axis([0.0, 1.0, 0.0, 1.0])
                plt.tight_layout()
                plt.savefig( output + "scatt/" + str(index1) +'-'+ str(index2) + '-'+ str(index3), dpi=10.7)
                #plt.show()
                plt.close('all')
