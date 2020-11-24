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

"""Shapes3D data set."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import cv2
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf
from numpy import genfromtxt


SCATT_PATH = os.path.join("/content/Thesis/disentanglement_lib/data")



class Scatt(ground_truth_data.GroundTruthData):
  """Shapes3D dataset.

  The data set was originally introduced in "Disentangling by Factorising".

  The ground-truth factors of variation are:
  0 - size(23 different values)
  1 - shape(16 different values)
  2 - color (20 different values)
  """

  def __init__(self):
    join = []
    count = 0
    for img in os.listdir(SCATT_PATH + '/scatt'):
        if img.endswith('.png'):
            count += 1
            img_array = cv2.imread(os.path.join((SCATT_PATH + '/scatt'),img))# convert to array
            join.append(img_array)  # add this to our training_data
        
    images = np.array(join)
    labels = genfromtxt(SCATT_PATH + '/output.csv', delimiter=',')
    n_samples = np.prod(images.shape[0:6])
    self.images = (
        images.reshape([count, 64, 64, 3]).astype(np.float32) / 255.)
    features = labels.reshape([count, 3])
    self.factor_sizes = [23, 16, 20]
    self.latent_factor_indices = list(range(3))
    self.num_total_factors = features.shape[1]
    self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                    self.latent_factor_indices)
    self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
        self.factor_sizes)

  @property
  def num_factors(self):
    return self.state_space.num_latent_factors

  @property
  def factors_num_values(self):
    return self.factor_sizes

  @property
  def observation_shape(self):
    return [64, 64, 3]


  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return self.state_space.sample_latent_factors(num, random_state)

  def sample_observations_from_factors(self, factors, random_state):
    all_factors = self.state_space.sample_all_factors(factors, random_state)
    indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
    return self.images[indices]
