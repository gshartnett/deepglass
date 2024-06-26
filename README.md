# deepglass
Deep generative spin-glass models with normalizing flows (https://arxiv.org/abs/2001.00585).

## System Requirements
A GPU is recommended, but not necessary. The required software dependencies may be installed using the setup.py file as described below. This code has been tested on Ubuntu 18.04.2 LTS, Google Colab, and on a p2.xlarge Amazon Web Services (AWS) EC2 instance configured with the following Amazon Machine Image (AMI): Deep Learning AMI (Ubuntu) Version 21.0 (ami-003ce277a8a9c0014).

## Installation
It is recommended to use [conda](https://docs.conda.io/en/latest/) to manage the packages needed to run this code. With conda installed, the packages needed for this repository can be installed by following these commands:

    git clone https://github.com/gshartnett/deepglass
    cd deepglass
    conda create --name deepglass_env python=3.6
    source activate deepglass_env
    python setup.py install
    python -m ipykernel install --user --name deepglass_env --display-name "Python (deepglass)"

## Example use
This repo includes 3 scripts:
  - `SK-parallel-tempering.py`: generates parallel tempering (PT) MCMC samples of the SK model
  - `NVP-trainer-reverseKL.py`: trains deep generative spin-glasses using the reverse KL divergence
  - `NVP-trainer-forwardKL.py`: uses the PT samples to train deep generative spin-glasses using the forward KL divergence

The run-time of the code can be improved with GPUs and parallelization. The default values of the code's parameters are such that the scripts can be run in a few minutes on a laptop with just a CPU (these parameters include the number of spins, the number of disorder replicas, the total number of training iterations, etc). To reproduce the results of our paper https://arxiv.org/abs/2001.00585, we recommend using GPUs as well performing the simulations for different disorder replicas on parallel instances.
