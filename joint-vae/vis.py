from utils.load_model import load
from viz.visualize import Visualizer as Viz
import matplotlib.pyplot as plt
from utils.dataloaders import get_sctterplot_dataloader



dataset_path = '/content/Thesis/disentanglement_lib/data/og/' 
path_to_model_folder = '/content/Thesis/joint-vae/trained_models/scatterplot/'

model = load(path_to_model_folder)
print(model.latent_spec)

print(model)

# Create a Visualizer for the model

viz = Viz(model)
samples = viz.samples()
print(samples)
viz.save_images = True  # Return tensors instead of saving images

traversal = viz.latent_traversal_line(cont_idx=6, size=12)
traversals = viz.all_latent_traversals()

