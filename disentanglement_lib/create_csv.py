import os
import pandas as pd

graph_features = pd.DataFrame(columns=['size' , 'shape','color'])

for index1, filename in enumerate(os.listdir('/content/Thesis/disentanglement_lib/data/scatt')): 
  x = filename.split('-')
  y = x[6].split('.')
  graph_features = graph_features.append( {'size':int((int(x[2])/200)-1) ,'shape':x[4], 'color' : y[0]}, ignore_index=True)
                
print(graph_features)
graph_features.to_csv('/content/Thesis/disentanglement_lib/output.csv',index = False, header=False)