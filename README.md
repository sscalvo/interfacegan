# Finding StyleGAN Latent Space Boundary Vectors 
### Based on: InterFaceGAN - Interpreting the Latent Space of GANs for Semantic Face Editing


![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![pytorch 1.1.0](https://img.shields.io/badge/pytorch-1.1.0-green.svg?style=plastic)
![TensorFlow 1.12.2](https://img.shields.io/badge/tensorflow-1.12.2-green.svg?style=plastic)
![sklearn 0.21.2](https://img.shields.io/badge/sklearn-0.21.2-green.svg?style=plastic)

![image](./docs/assets/results.png)
**Figure:** *High-quality facial attributes editing results with InterFaceGAN.*

This repository is an adaptation of the original InterFaceGAN in which we try to find the boundary vector for any specific feature (in this case, we focus on *eye color*, but you can easily adapt to the feature of your wish) 

## How to calculate a boundary vector associated to a semantic (eyes color) feature?

![image](./docs/assets/VGG_crop.png)

**Figure:** *Cropping faces to obtain 224x224 eyes images. The corresponding W vector is saved.*

 - Choose whatever feature you want to force StyleGAN to move towards (in this case we chose *eyes color*)
 - Generate between 10k and 50k images to train a classifier for your chosen feature: Since we are dealing with eyes, we used ```custom_generate_data.py``` (a custom version of ```generate_data.py``` that generates the faces and crops both eyes joining them together into a single 224x224 image). Use this colab notebook to generate 10k of such images: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10pS-aIIUrBJJI3HyNAj2x9xh1wGWIzVa?usp=sharing)

- Manually classify this images into your desired semantics (in our case, *light, brown, black, sunglass*)

![image](./docs/assets/classes.jpg)

- ..and zip it. We will use it to train a classifier neural network.

![image](./docs/assets/labeled_dataset.png)

 - Now you are ready to follow instrucctions of book 2 (we will extract image features using a VGG network, and use them to train our DNN)
 
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18HuLJuez6eYHL9yfjcbkB_kcw4c34NnJ?usp=sharing)
 
 If you want to skip book 2, just download the already [trained dnn model](https://drive.google.com/file/d/1YWb3ptS9cuZ9d5obUxthj1nlwiNZjHtT/view?usp=sharing)



## GAN Models Used (Prior Work)

Before going into details, we would like to first introduce the two state-of-the-art GAN models used in this work, which are ProgressiveGAN (Karras *el al.*, ICLR 2018) and StyleGAN (Karras *et al.*, CVPR 2019). These two models achieve high-quality face synthesis by learning unconditional GANs. For more details about these two models, please refer to the original papers, as well as the official implementations.

ProgressiveGAN:
  [[Paper](https://arxiv.org/pdf/1710.10196.pdf)]
  [[Code](https://github.com/tkarras/progressive_growing_of_gans)]

StyleGAN:
  [[Paper](https://arxiv.org/pdf/1812.04948.pdf)]
  [[Code](https://github.com/NVlabs/stylegan)]

## Code Instruction

### Generative Models

A GAN-based generative model basically maps the latent codes (commonly sampled from high-dimensional latent space, such as standart normal distribution) to photo-realistic images. Accordingly, a base class for generator, called `BaseGenerator`, is defined in `models/base_generator.py`. Basically, it should contains following member functions:

- `build()`: Build a pytorch module.
- `load()`: Load pre-trained weights.
- `convert_tf_model()` (Optional): Convert pre-trained weights from tensorflow model.
- `sample()`: Randomly sample latent codes. This function should specify what kind of distribution the latent code is subject to.
- `preprocess()`: Function to preprocess the latent codes before feeding it into the generator.
- `synthesize()`: Run the model to get synthesized results (or any other intermediate outputs).
- `postprocess()`: Function to postprocess the outputs from generator to convert them to images.

We have already provided following models in this repository:

- ProgressiveGAN:
  - A clone of official tensorflow implementation: `models/pggan_tf_official/`. This clone is only used for converting tensorflow pre-trained weights to pytorch ones. This conversion will be done automitally when the model is used for the first time. After that, tensorflow version is not used anymore.
  - Pytorch implementation of official model (just for inference): `models/pggan_generator_model.py`.
  - Generator class derived from `BaseGenerator`: `models/pggan_generator.py`.
  - Please download the official released model trained on CelebA-HQ dataset and place it in folder `models/pretrain/`.
- StyleGAN:
  - A clone of official tensorflow implementation: `models/stylegan_tf_official/`. This clone is only used for converting tensorflow pre-trained weights to pytorch ones. This conversion will be done automitally when the model is used for the first time. After that, tensorflow version is not used anymore.
  - Pytorch implementation of official model (just for inference): `models/stylegan_generator_model.py`.
  - Generator class derived from `BaseGenerator`: `models/stylegan_generator.py`.
  - Please download the official released models trained on CelebA-HQ dataset and FF-HQ dataset and place them in folder `models/pretrain/`.
  - Support synthesizing images from $\mathcal{Z}$ space, $\mathcal{W}$ space, and extended $\mathcal{W}$ space (18x512).
  - Set truncation trick and noise randomization trick in `models/model_settings.py`. Among them, `STYLEGAN_RANDOMIZE_NOISE` is highly recommended to set as `False`. `STYLEGAN_TRUNCATION_PSI = 0.7` and `STYLEGAN_TRUNCATION_LAYERS = 8` are inherited from official implementation. Users can customize their own models. NOTE: These three settings will NOT affect the pre-trained weights.
- Customized model:
  - Users can do experiments with their own models by easily deriving new class from `BaseGenerator`.
  - Before used, new model should be first registered in `MODEL_POOL` in file `models/model_settings.py`.

### Utility Functions

We provide following utility functions in `utils/manipulator.py` to make InterFaceGAN much easier to use.

- `train_boundary()`: This function can be used for boundary searching. It takes pre-prepared latent codes and the corresponding attributes scores as inputs, and then outputs the normal direction of the separation boundary. Basically, this goal is achieved by training a linear SVM. The returned vector can be further used for semantic face editing.
- `project_boundary()`: This function can be used for conditional manipulation. It takes a primal direction and other conditional directions as inputs, and then outputs a new normalized direction. Moving latent code along this new direction will manipulate the primal attribute yet barely affect the conditioned attributes. NOTE: For now, at most two conditions are supported.
- `linear_interpolate()`: This function can be used for semantic face editing. It takes a latent code and the normal direction of a particular semantic boundary as inputs, and then outputs a collection of manipulated latent codes with linear interpolation. These interpolation can be used to see how the synthesis will vary if moving the latent code along the given direction.

### Tools

- `generate_data.py`: This script can be used for data preparation. It will generate a collection of syntheses (images are saved for further attribute prediction) as well as save the input latent codes.

- `train_boundary.py`: This script can be used for boundary searching.

- `edit.py`: This script can be usd for semantic face editing.

## Usage

We take ProgressiveGAN model trained on CelebA-HQ dataset as an instance.

### Prepare data

```bash
NUM=10000
python generate_data.py -m pggan_celebahq -o data/pggan_celebahq -n "$NUM"
```

### Predict Attribute Score

Get your own predictor for attribute `$ATTRIBUTE_NAME`, evaluate on all generated images, and save the inference results as `data/pggan_celebahq/"$ATTRIBUTE_NAME"_scores.npy`. NOTE: The save results should be with shape `($NUM, 1)`.

### Search Semantic Boundary

```bash
python train_boundary.py \
    -o boundaries/pggan_celebahq_"$ATTRIBUTE_NAME" \
    -c data/pggan_celebahq/z.npy \
    -s data/pggan_celebahq/"$ATTRIBUTE_NAME"_scores.npy
```

### Compute Conditional Boundary (Optional)

This step is optional. It depends on whether conditional manipulation is needed. Users can use function `project_boundary()` in file `utils/manipulator.py` to compute the projected direction.

## Boundaries Description

We provided following boundaries in folder `boundaries/`. The boundaries can be more accurate if stronger attribute predictor is used.

- ProgressiveGAN model trained on CelebA-HQ dataset:
  - Single boundary:
    - `pggan_celebahq_pose_boundary.npy`: Pose.
    - `pggan_celebahq_smile_boundary.npy`: Smile (expression).
    - `pggan_celebahq_age_boundary.npy`: Age.
    - `pggan_celebahq_gender_boundary.npy`: Gender.
    - `pggan_celebahq_eyeglasses_boundary.npy`: Eyeglasses.
    - `pggan_celebahq_quality_boundary.npy`: Image quality.
  - Conditional boundary:
    - `pggan_celebahq_age_c_gender_boundary.npy`: Age (conditioned on gender).
    - `pggan_celebahq_age_c_eyeglasses_boundary.npy`: Age (conditioned on eyeglasses).
    - `pggan_celebahq_age_c_gender_eyeglasses_boundary.npy`: Age (conditioned on gender and eyeglasses).
    - `pggan_celebahq_gender_c_age_boundary.npy`: Gender (conditioned on age).
    - `pggan_celebahq_gender_c_eyeglasses_boundary.npy`: Gender (conditioned on eyeglasses).
    - `pggan_celebahq_gender_c_age_eyeglasses_boundary.npy`: Gender (conditioned on age and eyeglasses).
    - `pggan_celebahq_eyeglasses_c_age_boundary.npy`: Eyeglasses (conditioned on age).
    - `pggan_celebahq_eyeglasses_c_gender_boundary.npy`: Eyeglasses (conditioned on gender).
    - `pggan_celebahq_eyeglasses_c_age_gender_boundary.npy`: Eyeglasses (conditioned on age and gender).
- StyleGAN model trained on CelebA-HQ dataset:
  - Single boundary in $\mathcal{Z}$ space:
    - `stylegan_celebahq_pose_boundary.npy`: Pose.
    - `stylegan_celebahq_smile_boundary.npy`: Smile (expression).
    - `stylegan_celebahq_age_boundary.npy`: Age.
    - `stylegan_celebahq_gender_boundary.npy`: Gender.
    - `stylegan_celebahq_eyeglasses_boundary.npy`: Eyeglasses.
  - Single boundary in $\mathcal{W}$ space:
    - `stylegan_celebahq_pose_w_boundary.npy`: Pose.
    - `stylegan_celebahq_smile_w_boundary.npy`: Smile (expression).
    - `stylegan_celebahq_age_w_boundary.npy`: Age.
    - `stylegan_celebahq_gender_w_boundary.npy`: Gender.
    - `stylegan_celebahq_eyeglasses_w_boundary.npy`: Eyeglasses.

- StyleGAN model trained on FF-HQ dataset:
  - Single boundary in $\mathcal{Z}$ space:
    - `stylegan_ffhq_pose_boundary.npy`: Pose.
    - `stylegan_ffhq_smile_boundary.npy`: Smile (expression).
    - `stylegan_ffhq_age_boundary.npy`: Age.
    - `stylegan_ffhq_gender_boundary.npy`: Gender.
    - `stylegan_ffhq_eyeglasses_boundary.npy`: Eyeglasses.
  - Conditional boundary in $\mathcal{Z}$ space:
    - `stylegan_ffhq_age_c_gender_boundary.npy`: Age (conditioned on gender).
    - `stylegan_ffhq_age_c_eyeglasses_boundary.npy`: Age (conditioned on eyeglasses).
    - `stylegan_ffhq_eyeglasses_c_age_boundary.npy`: Eyeglasses (conditioned on age).
    - `stylegan_ffhq_eyeglasses_c_gender_boundary.npy`: Eyeglasses (conditioned on gender).
  - Single boundary in $\mathcal{W}$ space:
    - `stylegan_ffhq_pose_w_boundary.npy`: Pose.
    - `stylegan_ffhq_smile_w_boundary.npy`: Smile (expression).
    - `stylegan_ffhq_age_w_boundary.npy`: Age.
    - `stylegan_ffhq_gender_w_boundary.npy`: Gender.
    - `stylegan_ffhq_eyeglasses_w_boundary.npy`: Eyeglasses.

## BibTeX

```bibtex
@inproceedings{shen2020interpreting,
  title     = {Interpreting the Latent Space of GANs for Semantic Face Editing},
  author    = {Shen, Yujun and Gu, Jinjin and Tang, Xiaoou and Zhou, Bolei},
  booktitle = {CVPR},
  year      = {2020}
}
```

```bibtex
@article{shen2020interfacegan,
  title   = {InterFaceGAN: Interpreting the Disentangled Face Representation Learned by GANs},
  author  = {Shen, Yujun and Yang, Ceyuan and Tang, Xiaoou and Zhou, Bolei},
  journal = {TPAMI},
  year    = {2020}
}
```
