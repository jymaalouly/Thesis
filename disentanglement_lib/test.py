from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gin.tf
import gin
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
from disentanglement_lib.visualize import visualize_dataset
import tensorflow.compat.v1 as tf
import os
from numpy import loadtxt


path = os.getcwd()
print(path)
path_output = os.path.join(path, "output")
path_vae = os.path.join(path_output, "vae")


overwrite = True
model = ["model.gin"]
path = os.path.join(path_vae, "model")
train.train_with_gin(path, True, model)