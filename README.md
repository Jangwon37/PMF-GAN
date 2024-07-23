# Stabilized GAN Models Training with Kernel-Histogram Transformation and Probability Mass Function Distance

## Article information
Seo, J., Hwang, H. S., Lee, M., & Seok, J. (2024). Stabilized GAN Models Training with Kernel-Histogram Transformation and Probability Mass Function Distance. Applied Soft Computing, 112003.
[https://doi.org/10.1016/j.asoc.2024.112003](https://doi.org/10.1016/j.asoc.2024.112003)

![figure1](https://github.com/Jangwon37/PMF-GAN/assets/99333410/f77ad3c8-bd9e-45a8-9812-fe2c124386d6)

## Structure of directory
```
├── code
│   ├── train.py
│   ├── sample.py
│   ├── utils
│   │   ├── utility.py
│   │   └── score.py
│   ├── data
│   │   └──data-loader.py
│   ├── calc
│   │   └──statistical_analysis.py
│   └── models
│       ├── loss_model.py
│       └── model.py
├── requierments.txt
└── result
```

## Requirements
+ Linux and Windows are supported.
+ 64-bit Python 3.8 and PyTorch 1.10.1.
+ CUDA toolkit 11.1 or later. Use at least version 11.1 
+ Python libraries:
    ```
    pip install -r requirements.txt
    ```

## Getting started
```
python train.py

optional arguments:
--n_epochs                number of epochs of training [default value is 256]
--batch_size              size of the batches [default value is 128]
--n_bin                   histogram's bin [default value is 3]
--dataset                 Enter the dataset you want the model to train on [default value is cifar10]

The code executes when you select the model number below.
1. WGAN-GP
2. LSGAN-GP
3. Euclidean
4. Chi2
5. Chybyshev
6. Manhattan
7. SquaredChord
```

## Experiments and results
![figure3](https://github.com/Jangwon37/PMF-GAN/assets/99333410/cf8e024a-ac00-4daf-ba56-2359200c2a9b)
Displaying the training results for WGAN, LSGAN, SphereGAN, and PMF-GANs on the AFHQ, CelebA, LSUN, CIFAR-10, and CIFAR-100 datasets. The training was conducted for 100,000 iterations, with the number of bins in PMF-GANs set to 3.
