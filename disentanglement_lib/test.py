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
from disentanglement_lib.visualize import visualize_model
import tensorflow.compat.v1 as tf
import os
from numpy import loadtxt


path = "/content/Thesis/disentanglement_lib"

base_path = os.path.join(path, "output")
path_vae = os.path.join(base_path, "vae")


overwrite = True
model = ["model.gin"]
path = os.path.join(path_vae, "model")
train.train_with_gin(path, True, model)




representation_path = os.path.join(path_vae, "representation")
model_path = os.path.join(path_vae, "model")
postprocess_gin = ["postprocess.gin"]  # This contains the settings.
# postprocess.postprocess_with_gin defines the standard extraction protocol.
postprocess.postprocess_with_gin(model_path, representation_path, overwrite,postprocess_gin)

# 4. Compute the Mutual Information Gap (already implemented) for both models.
# ------------------------------------------------------------------------------
# The main evaluation protocol of disentanglement_lib is defined in the
# disentanglement_lib.evaluation.evaluate module. Again, we have to provide a
# gin configuration. We could define a .gin config file; however, in this case
# we show how all the configuration settings can be set using gin bindings.
# We use the Mutual Information Gap (with a low number of samples to make it
# faster). To learn more, have a look at the different scores in
# disentanglement_lib.evaluation.evaluate.metrics and the predefined .gin
# configuration files in
# disentanglement_lib/config/unsupervised_study_v1/metrics_configs/(...).
gin_bindings = [
    "evaluation.evaluation_fn = @mig",
    "dataset.name='auto'",
    "evaluation.random_seed = 0",
    "mig.num_train=1000",
    "discretizer.discretizer_fn = @histogram_discretizer",
    "discretizer.num_bins = 20"
]

result_path = os.path.join(path_vae, "metrics", "mig")
representation_path = os.path.join(path_vae, "representation")
evaluate.evaluate_with_gin(representation_path, result_path, overwrite, gin_bindings=gin_bindings)


# 5. Compute a custom disentanglement metric for both models.
@gin.configurable(
    "custom_metric",
    blacklist=["ground_truth_data", "representation_function", "random_state"])
def compute_custom_metric(ground_truth_data,
                          representation_function,
                          random_state,
                          num_train=gin.REQUIRED,
                          batch_size=16):
  """Example of a custom (dummy) metric.
  Preimplemented metrics can be found in disentanglement_lib.evaluation.metrics.
  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.
  Returns:
    Dict with disentanglement score.
  """
  score_dict = {}

  # This is how to obtain the representations of num_train points along with the
  # ground-truth factors of variation.
  representation, factors_of_variations = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_train, random_state,
      batch_size)
  # We could now compute a metric based on representation and
  # factors_of_variations. However, for the sake of brevity, we just return 1.
  del representation, factors_of_variations
  score_dict["custom_metric"] = 1.
  return score_dict


# To compute the score, we again call the evaluation protocol with a gin
# configuration. At this point, note that for all steps, we have to set a
# random seed (in this case via `evaluation.random_seed`).
gin_bindings = [
    "evaluation.evaluation_fn = @custom_metric",
    "custom_metric.num_train = 100", "evaluation.random_seed = 0",
    "dataset.name='auto'"
]
result_path = os.path.join(path_vae, "metrics", "custom_metric")
evaluate.evaluate_with_gin(representation_path, result_path, overwrite, gin_bindings=gin_bindings)

# 6. Aggregate the results.
# ------------------------------------------------------------------------------
# In the previous steps, we saved the scores to several output directories. We
# can aggregate all the results using the following command.
pattern = os.path.join(base_path,
                       "*/metrics/*/results/aggregate/evaluation.json")
results_path = os.path.join(base_path, "results.json")
aggregate_results.aggregate_results_to_json(
    pattern, results_path)

# 7. Print out the final Pandas data frame with the results.
# ------------------------------------------------------------------------------
# The aggregated results contains for each computed metric all the configuration
# options and all the results captured in the steps along the pipeline. This
# should make it easy to analyze the experimental results in an interactive
# Python shell. At this point, note that the scores we computed in this example
# are not realistic as we only trained the models for a few steps and our custom
# metric always returns 1.
model_results = aggregate_results.load_aggregated_json_results(results_path)
print(model_results)


