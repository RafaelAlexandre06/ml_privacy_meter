# Model Optimization Experiment Guide
# Objective

This series of experiments aims to systematically investigate the impact of hyperparameters and training strategies on the performance of Vision Transformers (ViT) on the CIFAR-10 dataset.

The goal is to build a diverse collection of trained models with different configurations, which will be used in subsequent privacy auditing experiments.

# Prerequisites

The experiments require one of the following environments:

Google Colab with a T4 GPU
An equivalent environment with sufficient GPU resources

Make sure that all required Python dependencies have been installed before running the experiments.

# Running the Experiments

To train a model using the provided configuration, run:

python train_optimized.py --config configs/vit_ffcv.yaml


The configuration file specifies the model architecture, dataset, optimization settings, and other training parameters.

# Experiment Configuration

Experiment settings can be modified through the configuration files under:

configs/


By changing the configuration, you can explore different combinations of model and training parameters, such as:

Learning rate
Batch size
Number of training epochs
Optimizer
Weight decay
Data augmentation
Model architecture and size
Learning-rate scheduling
Other training strategies
# Experimental Goal

Each experiment should produce a trained ViT model with a specific configuration. By systematically varying the training settings, we aim to obtain models with different performance characteristics and training behaviors.

These models will serve as inputs for downstream privacy auditing and analysis.
