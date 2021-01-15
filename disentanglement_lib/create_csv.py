import os
import pandas as pd
import math

graph_features = pd.DataFrame(columns=['pos','size' , 'shape','color'])

counter = 0
pos = 0 
for index1, filename in enumerate(sorted(os.listdir('/content/Thesis/disentanglement_lib/data/synthetic_data/scatt'))): 
  x = filename.split('-')
  y = x[8].split('.')
  graph_features = graph_features.append( {'pos': int(x[2]) ,'size':int((int(x[4])/50)-1) ,'shape':x[6], 'color' : y[0]}, ignore_index=True)
  
print(graph_features)
graph_features.to_csv('/content/Thesis/disentanglement_lib/data/synthetic_data/output.csv',index = False, header=False)
print(pos)