# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualization module for disentangled representations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numbers
import os
import cv2
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.utils import results
from disentanglement_lib.visualize import visualize_util
from disentanglement_lib.visualize.visualize_irs import vis_all_interventional_effects
import numpy as np
from scipy import stats
from six.moves import range
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile
import tensorflow_hub as hub
import gin.tf
from PIL import Image



"""Takes trained model from model_dir and visualizes it in output_dir.
Args:
  model_dir: Path to directory where the trained model is saved.
  output_dir: Path to output directory.
  overwrite: Boolean indicating whether to overwrite output directory.
  num_animations: Integer with number of distinct animations to create.
  num_frames: Integer with number of frames in each animation.
  fps: Integer with frame rate for the animation.
  num_points_irs: Number of points to be used for the IRS plots.
"""
model_dir = "/content/Thesis/disentanglement_lib/output/vae/model"
output_dir = "/content/Thesis/disentanglement_lib/test"
overwrite=True
num_animations=5
num_frames=20
fps=10
num_points_irs=10000



def sigmoid(x):
  return stats.logistic.cdf(x)


def tanh(x):
  return np.tanh(x) / 2. + .5


def latent_traversal_1d_multi_dim(generator_fn,
                                  latent_vector,
                                  dimensions=None,
                                  values=None,
                                  transpose=False):
  if latent_vector.ndim != 1:
    raise ValueError("Latent vector needs to be 1-dimensional.")

  if dimensions is None:
    # Default case, use all available dimensions.
    dimensions = np.arange(latent_vector.shape[0])
  elif isinstance(dimensions, numbers.Integral):
    # Check that there are enough dimensions in latent_vector.
    if dimensions > latent_vector.shape[0]:
      raise ValueError("The number of dimensions of latent_vector is less than"
                       " the number of dimensions requested in the arguments.")
    if dimensions < 1:
      raise ValueError("The number of dimensions has to be at least 1.")
    dimensions = np.arange(dimensions)
  if dimensions.ndim != 1:
    raise ValueError("Dimensions vector needs to be 1-dimensional.")

  if values is None:
    # Default grid of values.
    values = np.linspace(-1., 1., num=11)
  elif isinstance(values, numbers.Integral):
    if values <= 1:
      raise ValueError("If an int is passed for values, it has to be >1.")
    values = np.linspace(-1., 1., num=values)
  if values.ndim != 1:
    raise ValueError("Values vector needs to be 1-dimensional.")

  # We iteratively generate the rows/columns for each dimension as different
  # Numpy arrays. We do not preallocate a single final Numpy array as this code
  # is not performance critical and as it reduces code complexity.
  num_values = len(values)
  row_or_columns = []
  for dimension in dimensions:
    # Creates num_values copy of the latent_vector along the first axis.
    latent_traversal_vectors = np.tile(latent_vector, [num_values, 1])
    # Intervenes in the latent space.
    latent_traversal_vectors[:, dimension] = values
    # Generate the batch of images
    images = generator_fn(latent_traversal_vectors)
    # Adds images as a row or column depending whether transpose is True.
    axis = (1 if transpose else 0)
    row_or_columns.append(np.concatenate(images, axis))
  axis = (0 if transpose else 1)
  return np.concatenate(row_or_columns, axis)


# Fix the random seed for reproducibility.
random_state = np.random.RandomState(0)

# Create the output directory if necessary.
if tf.gfile.IsDirectory(output_dir):
  if overwrite:
    tf.gfile.DeleteRecursively(output_dir)
  else:
    raise ValueError("Directory already exists and overwrite is False.")

# Automatically set the proper data set if necessary. We replace the active
# gin config as this will lead to a valid gin config file where the data set
# is present.
# Obtain the dataset name from the gin config of the previous step.
gin_config_file = os.path.join(model_dir, "results", "gin", "train.gin")
gin_dict = results.gin_dict(gin_config_file)
gin.bind_parameter("dataset.name", gin_dict["dataset.name"].replace(
    "'", ""))

# Automatically infer the activation function from gin config.
activation_str = gin_dict["reconstruction_loss.activation"]
if activation_str == "'logits'":
  activation = sigmoid
elif activation_str == "'tanh'":
  activation = tanh

num_pics = 64
module_path = os.path.join(model_dir, "tfhub")

with hub.eval_function_for_module(module_path) as f:

  # Save samples.
  def _decoder(latent_vectors):
    return f(
        dict(latent_vectors=latent_vectors),
        signature="decoder",
        as_dict=True)["images"]

  num_latent = int(gin_dict["encoder.num_latent"])
  num_pics = 32
  
  img_array = cv2.imread("/content/Thesis/disentanglement_lib/9-750-5-1.png")# convert to array
  img_array = img_array.reshape([1, 64, 64, 3]).astype(np.float32) / 255. # add this to our training_data
  random_codes = random_state.normal(0, 1, [1, 3])
  print(random_codes)
  pics = activation(_decoder(random_codes))
  results_dir = os.path.join(output_dir, "sampled")
  if not gfile.IsDirectory(results_dir):
      gfile.MakeDirs(results_dir)
  visualize_util.grid_save_images(pics,
                                  os.path.join(results_dir, "samples.jpg"))
  # Save latent traversals.
  result = f(
      dict(images=img_array),
      signature="gaussian_encoder",
      as_dict=True)
  means = result["mean"]
  logvars = result["logvar"]
  
  pics = activation(
      latent_traversal_1d_multi_dim(_decoder, means[0, :], None))
  file_name = os.path.join(results_dir, "traversals{}.jpg".format(0))
  visualize_util.grid_save_images([pics], file_name)

  #img = Image.fromarray(pics.reshape([64, 64, 3]), 'RGB')
  #img.save('/content/Thesis/disentanglement_lib/my.png')
  #img.show()
  

  #visualize_util.save_image([pics], "/content/Thesis/disentanglement_lib/test")

gin.clear_config()

