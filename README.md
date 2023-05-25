# Dr-Fairness: Dynamic Data Ratio Adjustment for Fair Training on Real and Generated Data

**Authors: Yuji Roh, Weili Nie, De-An Huang, Steven Euijong Whang, Arash Vahdat, and Anima Anandkumar**

This repo contains codes used in the TMLR 2023 paper: [Dr-Fairness: Dynamic Data Ratio Adjustment for Fair Training on Real and Generated Data](https://openreview.net/forum?id=TyBd56VK7z).

*Abstract: Fair visual recognition has become critical for preventing demographic disparity. A major cause of model unfairness is the imbalanced representation of different groups in training data. Recently, several works aim to alleviate this issue using generated data. However, these approaches often use generated data to obtain similar amounts of data across groups, which is not optimal for achieving high fairness due to different learning difficulties and generated data qualities across groups. To address this issue, we propose a novel adaptive sampling approach that leverages both real and generated data for fairness. We design a bilevel optimization that finds the optimal data sampling ratios among groups and between real and generated data while training a model. The ratios are dynamically adjusted considering both the model's accuracy as well as its fairness. To efficiently solve our non-convex bilevel optimization, we propose a simple approximation to the solution given by the implicit function theorem. Extensive experiments show that our framework achieves state-of-the-art fairness and accuracy on the CelebA and ImageNet People Subtree datasets. We also observe that our method adaptively relies less on the generated data when it has poor quality. Our work shows the importance of using generated data together with real data for improving model fairness.*

## Setting

### General setting
The program needs PyTorch, PyTorch libraries, and CUDA for simulating Dr-Fairness on the CelebA dataset.
The directory contains a total of 7 files: 1 README and 6 python files.
To run the code, we need to prepare the original CelebA dataset (Liu et al., ICCV15) and generated dataset. 
We can put the data paths as input arguments for the main program.

### How to create generated data
In general, any synthetic data, including data from deep generative models, can be considered generated data in Dr-Fairness. Here, the key role of the generated data in algorithmic fairness is supporting the limited subset of the real data. In this paper, we assume we can get group-specific
generated data by using conditional image generation techniques (Nie et al., 2021; Dhariwal & Nichol, 2021).

**The generative model used for CelebA:** We use a StyleGAN-based controllable generation method called [LACE (Nie et al., 2021)](https://nvlabs.github.io/LACE/), which can synthesize images for each (label y, group z)-class. 
LACE is a controllable generation method that uses an energy-based model (EBM) in the latent space of a pre-trained generative model such as StyleGAN2 (Karras et al., 2020). We consider StyleGAN2 pre-trained on the CelebA-HQ dataset as our base generative model. In LACE, we first need to train the latent classifiers in the w-space of StyleGAN2, each of which corresponds to an energy function for an individual attribute in the EBM formulation (see Eq. (4) in (Nie et al., 2021)). Next, for each combination of attribute values (e.g., age=‘young’, gender=‘female’, smile=‘true’, glasses=‘true’, and haircolor=‘black’), we use the ordinal differential equation (ODE) sampler in the latent space to sample the corresponding images. 

**The number of generated samples**: In CelebA, we consider 5 attributes for the controllable generation: age (young and old), gender (male and female), smile (true and false), glasses (true and false), and haircolor (black, blond, and others). Thus, these 5 attributes yield 48 class combinations (i.e., 2^4 × 3). We generate a total of 96k samples, where there are 2k samples for each attribute combination (e.g., 2k samples for (age=‘young’, gender=‘female’, smile=‘true’, glasses=‘true’, and haircolor=‘black’)).

Please refer to more details on data generation in Section B.3 of our paper. We note that Dr-Fairness can also work with generated data that are created differently (e.g., using other generation models or changing the number of samples). 

### How to use the prepared generated data
We now explain how to set the directory of generated data. In the generated data path (e.g., PATH/TO/GEN/DATA/), we first make attribute combination folders, which are named by listing the attribute values connecting by underbars (e.g., PATH/TO/GEN/DATA/old_male_nosmile_glasses_blackhair). Then, in each attribute combination folder, we save the corresponding generated samples. When we follow the above section's instructions for CelebA, there will be a total of 48 folders under the path PATH/TO/GEN/DATA/, and 2k images are saved in each folder. These generated images will be loaded from preprocessing_celeba.py. We note that one can change the directory setting with the modification in preprocessing_celeba.py.

## Simulation
To simulate the algorithm, please use the **train_celeba_ours.py file**.

The train_celeba_ours.py file will load the data and train the models.
We can run this main code via the following (example) command:
```
$ python train_celeba_ours.py --batch_size 128 --fairness eqodds --data_path PATH/TO/REAL/DATA/ --gen_path PATH/TO/GEN/DATA/ --k 20 --n_classes 1 --y age --z gender
```

We can set other input arguments, including total_epochs, cuda_device, and save_path. 
Please see all possible arguments and their default values in the code.

The program first loads real and generated datasets and then serves both datasets and the initialized model to the training function.
In the training function, the model is iteratively updated based on the mini-batches given by the data loader.
Here, the data loader uses Dr-Fairness as the sampler.
The intermediate models will be saved in the ./intermediate_models/ directory.
Note that the specific functionalities are defined in other python files (e.g., DrFairness.py and preprocessing_celeba.py).

## Other details
The other five python files are DrFairness.py, preprocessing_celeba.py, models.py, customdataset.py, and ema.py.

The DrFairness.py contains a sampler class for the lambda and mu adjustment and batch selection of DrFairness.
The preprocessing_celeba.py contains functions for preprocessing both real and generated data. 
- Note that the preprocessing functions are designed for the original CelebA dataset (Liu et al., ICCV15) and our generated dataset described in the paper. Please refer to details on data generation in Section B.3.
- For the original CelebA dataset, we follow the preprocessing steps in (Ramaswamy et al., CVPR21). 
- For the generated dataset, users can change the preprocessing functions to support their own generated data.
The models.py contains various model architectures, including ResNet50.
The customdataset.py contains three classes that define different types of datasets to support customized sampling in the data loader.
The ema.py contains the class for the exponential moving average.

Detailed explanations about each component have been written in the codes as comments.

## Acknowledgements
This repo in part utilizes codes from the following:
- https://github.com/NVlabs/LSGM
- https://github.com/yuji-roh/fairbatch
- https://github.com/princetonvisualai/gan-debiasing

## License
This work is made available under the Nvidia Source Code License-NC. Please check the LICENSE file.

If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

This work may be used non-commercially, meaning for research or evaluation purposes only. For business inquiries, please contact researchinquiries@nvidia.com.

## Reference
```
@article{roh2023drfairness,
title={Dr-Fairness: Dynamic Data Ratio Adjustment for Fair Training on Real and Generated Data},
author={Yuji Roh and Weili Nie and De-An Huang and Steven Euijong Whang and Arash Vahdat and Anima Anandkumar},
journal={Transactions on Machine Learning Research},
year={2023},
url={https://openreview.net/forum?id=TyBd56VK7z}
}
```
