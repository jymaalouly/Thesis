import torch
from jointvae.models import VAE
from jointvae.training import Trainer
from utils.dataloaders import get_sctterplot_dataloader
from torch import optim


batch_size = 128
lr = 5e-4
epochs = 20
dataset_path = '/content/Thesis/disentanglement_lib/data/real_world_data' 
# Check for cuda
use_cuda = torch.cuda.is_available()

# Load data
data_loader = get_sctterplot_dataloader(batch_size, dataset_path)
img_size = (3, 64, 64)

# Define latent spec and model
latent_spec = {'cont': 16}
model = VAE(img_size=img_size, latent_spec=latent_spec,
            use_cuda=use_cuda)
if use_cuda:
    model.cuda()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define trainer
trainer = Trainer(model, optimizer,
                  cont_capacity=[0.0, 5.0, 25000, 30],
                  disc_capacity=[0.0, 5.0, 25000, 30],
                  use_cuda=use_cuda)

# Train model for 100 epochs
trainer.train(data_loader, epochs)

# Save trained model
torch.save(trainer.model.state_dict(), '/content/Thesis/joint-vae/trained_models/scatterplot/model.pt')
