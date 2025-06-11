This subdirectory holds all work related to the replication of the CHARACTERISING BIAS IN COMPRESSED MODELS paper, and a series of experiments
meant to both general inference and Compression Identified Exemplars (CIE) performance for our base model - ResNet18.

### Results:

The ResNet18 + ViT Encoder architecture reported an increase of 10% on recall and 5% on f1-score on the evaluations. Effectively improving on the original results of the paper.

### Setup:

Please first create & activate a python3 virtual environment in this HW2 directory:

For reference: Python 3.13.2 is the version used to build these experiments

`python3 -m venv venv`

`source venv/bin/activate`

Then, install dependencies found in requirements.txt:

`pip install -r requirements.txt`

Confirm that the dependencies are installed:

`pip list`

### TL:DR:

To observe key project results:

- Model weights can be found on `models/README.md`
- Notebooks containing results on full dataset are found in `src/notebooks/colab/`

### Process

In the following, we're going to reimplement the aforementioned paper. This will require us to create a ResNet-18 base architecture, and train it
on the CelebA so as to acquire similar results to the paper's.

### Experiment List

Additionally, we're going to perform a series of experiments in which we will make modifications to the base model's architecture (and possibly to our data processing as well) and compare the results to our base results. The following is our Experiment List:

- ResNet18 + ViT Encoder hybrid architecture
- Multi-head Latent Attention layer for Parameter-Efficient Fine-Tuning (PEFT)
- Data Augmentation (if time permits)

### Experiment Notebook Structure (BaseLine)

The following module explains the structure of the notebook which we will replicate and modify across our experiments:

- Class defining base ResNet-18 model
- Pick a fixed size batch, # of batches (because we’re dealing with a subset of the full dataset), random seed
- Training implementation of our model. To avoid anomalies while experimenting with different model architectures, we will train our models using cross-validation.
- Record train and validation Loss at each epoch
- Inference cell; here our transformer will perform inference on our validation sample set and its outputs will be saved on a file named validation_inf.txt.
- Record general inference and CIE inference performance
- In the following cell, we create tables to visualize the final performance of the experiment
- Visualize Loss of model across train and validation.
- Additionally, include inference performance metrics (general and CIE)

The purpose of this notebook is to be able to copy it, paste it, and add to it the desired changes to our base ResNet model and, or, data manipulation techniques - e.g. data augmentation. Once the changes are added, we run the entirety of the file and report the observed results.
