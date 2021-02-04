import os
import pandas as pd
import math

graph_features = pd.DataFrame(columns=['pos','size' , 'shape','color'])

counter = 0
pos = 0 
for index1, filename in enumerate(sorted(os.listdir('/content/Thesis/disentanglement_lib/data/rdataset/scatt'))): 
  
  x = filename.split('-')
  y = x[6].split('.')
  graph_features = graph_features.append( {'pos': int(x[0]) ,'size': int(x[2]) ,'shape': int(x[4]), 'color' : y[0]}, ignore_index=True)
  
print(graph_features)
graph_features.to_csv('/content/Thesis/disentanglement_lib/data/rdataset/output.csv',index = False, header=False)
print(pos)