import os
import pandas as pd
import math

graph_features = pd.DataFrame(columns=['pos','size' , 'shape','color'])

counter = 0
pos = 0 
for index1, filename in enumerate(sorted(os.listdir('/content/Thesis/disentanglement_lib/data/scatt'))): 
  counter += 1
  if (counter % 181) == 0:
    counter = 1
    pos += 1
  x = filename.split('-')
  y = x[6].split('.')
  graph_features = graph_features.append( {'pos': pos ,'size':int((int(x[2])/50)-1) ,'shape':x[4], 'color' : y[0]}, ignore_index=True)
  
print(graph_features)
graph_features.to_csv('/content/Thesis/disentanglement_lib/output.csv',index = False, header=False)
print(pos)