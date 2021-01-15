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
from tensorflow.compat.v1 import gfile
import os
from numpy import loadtxt


path = "/content/Thesis/disentanglement_lib/experiment_1/dsprites"

base_path = os.path.join(path, "output")
path_vae = os.path.join(base_path, "beta_vae")

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
if not gfile.IsDirectory(result_path):
    gfile.MakeDirs(result_path)
representation_path = os.path.join(path_vae, "representation")
evaluate.evaluate_with_gin(representation_path, result_path, overwrite, gin_bindings=gin_bindings)


gin_bindings = [
    "evaluation.evaluation_fn = @beta_vae_sklearn",
    "dataset.name='auto'",
    "evaluation.random_seed = 0",
    "beta_vae_sklearn.batch_size=32",
    "beta_vae_sklearn.num_train=100",
    "beta_vae_sklearn.num_eval=100",
    "discretizer.discretizer_fn = @histogram_discretizer",
    "discretizer.num_bins = 20"
]

result_path = os.path.join(path_vae, "metrics", "beta_vae_score")
if not gfile.IsDirectory(result_path):
    gfile.MakeDirs(result_path)
representation_path = os.path.join(path_vae, "representation")
evaluate.evaluate_with_gin(representation_path, result_path, overwrite, gin_bindings=gin_bindings)


gin_bindings = [
    "evaluation.evaluation_fn = @sap_score",
    "dataset.name='auto'",
    "sap_score.num_train=1000",
    "evaluation.random_seed=1",
    "sap_score.num_test=750",
    "sap_score.continuous_factors = False",
    "discretizer.discretizer_fn = @histogram_discretizer",
    "discretizer.num_bins = 20"
]

result_path = os.path.join(path_vae, "metrics", "sap_score")
if not gfile.IsDirectory(result_path):
    gfile.MakeDirs(result_path)
representation_path = os.path.join(path_vae, "representation")
evaluate.evaluate_with_gin(representation_path, result_path, overwrite, gin_bindings=gin_bindings)

gin_bindings = [
    "evaluation.evaluation_fn = @unsupervised_metrics",
    "evaluation.random_seed=1",
    "dataset.name='auto'",
    "unsupervised_metrics.num_train=100",
    "discretizer.discretizer_fn = @histogram_discretizer",
    "discretizer.num_bins = 20"
]

result_path = os.path.join(path_vae, "metrics", "unsupervised_metrics")
if not gfile.IsDirectory(result_path):
    gfile.MakeDirs(result_path)
representation_path = os.path.join(path_vae, "representation")
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


