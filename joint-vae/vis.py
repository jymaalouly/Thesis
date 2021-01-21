from utils.load_model import load
from viz.visualize import Visualizer as Viz
import matplotlib.pyplot as plt
from utils.dataloaders import get_sctterplot_dataloader



dataset_path = '/content/Thesis/disentanglement_lib/data/og/' 
path_to_model_folder = './trained_models/scatterplot/'

model = load(path_to_model_folder)

print(model.latent_spec)

print(model)

# Create a Visualizer for the model

viz = Viz(model)

print(viz)
viz.save_images = True  # Return tensors instead of saving images

data_loader = get_sctterplot_dataloader(64, dataset_path)


# Reconstruct data using Joint-VAE model
recon = viz.reconstructions(data_loader)
