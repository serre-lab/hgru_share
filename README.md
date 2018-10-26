# Learning long-range spatial dependencies with horizontal gated recurrent units

Introducing the hGRU, an approach to learning horizontal connections between computational units. This model is highly effective at capturing dependencies between visual features that span long spatial distances, with only very-deep Residual Networks and state-of-the-art models for per-pixel prediction rivaling the performance of a *single hGRU layer* on the tasks that we investigate in [our manuscript](https://arxiv.org/abs/1805.08315), set to appear at NIPS, 2018.

## Instructions
The code is structured to work based on directory paths described in `config.py`. Change `self.data_root` and `self.project_directory` to match your local configuration.

Classes in `dataset_processing` describe datasets that you will use with your models. The project expects TFRecords, and placeholders are depreciated.

Model scripts in the main directory have the function, `experiment_params`. This describes the experiment parameters for your project, such as learning rates, datasets, and batch sizes. Once this is set, you can run any of the models in the main directory. For example: `CUDA_VISIBLE_DEVICES=0 python hgru.py`.

- hgru.py. A one-layer hgru with a gabor-kernel feedforward drive.
- hgru_bn.py. A one-layer batchnorm hGRU with a gabor-kernel feedforward drive.
- hgru_bn_relu.py. A one-layer batchnorm hGRU with a ReLU nonlinearity, which constrains H^(1) to inhibition and H^(2) to excitation. This model additionally uses two horizontal kernels, W_1 and W_2. It uses the standard gabor-kernel feedforward drive.
- multiplicative_lesion.py. An hGRU with lesions to its multiplicative horizontal interactions (alpha and omega).
- additive_lesion.py. An hGRU with lesions to its additive horizontal interactions (mu and kappa).
- ff_5.py. A 5-layer feedforward control model (referred to as "large kernels" in the manuscript.).

