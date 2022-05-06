### Combined paired-unpaired GAN training for underwater image enhancement (UIE)
[Sonia Cromp](https://github.com/soCromp), University of Wisconsin-Madison Computer Sciences <br>
See [here](resource/cromp_cv_proposal.pdf) for the proposal and [here](resource/cv_midreport.pdf) for the midterm report.

## Introduction

Underwater image enhancement (UIE) is the problem of improving the visual quality of images, from the color to the clarity and sharpness. UIE has a broad range of applications, from archaeological and biological research to sunken ship recovery. However, UIE also faces several challenges. It is difficult to obtain large datasets required to train typical state-of-the-art machine learning models, and a dataset dedicated to one location (such as the Mariana trench) may not generalize well to other locations (such as a shallow lake). <br>

One line of prior work focuses on applying unpaired learning techniques to UIE. In a typical, paired UIE learning regime, one trains a model to translate from an unclear image to a clear image using a dataset consisting of *n* unclear images and their corresponding *n* clear images that have been manually enhanced. In unpaired learning, techniques are designed to learn from a dataset of *n* unclear images and a seperate dataset of *m* clear images. In this way, the model learns to the distinguishing characteristics of the source/unclear and the target/clear domains without any examples of one scene in both domains.

In the present work, I propose UW PUP GAN (UnderWater / University of Wisconsin) Paired-or-UnPaired Generative Adversarial Network, which fuses paired and unpaired learning techniques. The anticipated use-case is a situation where relatively large quantities of unpaired unclear and clear examples are available, along with a smaller paired dataset. For instance, perhaps a research group has a limited time or budget to manually enhance a small number of unclear images and wishes to aid the learning of this dataset with unpaired images. To my knowledge, this is the first UIE work to combine paired and unpaired learning.

UW PUP GAN is inspired by a combination of the paired and unpaired versions of FUnIE-GAN, which in turn also draws inspiration from CycleGAN. In the procress of this project, I created a PyTorch implementation of FUnIE-GAN that is available here LINK.

## Approach
Overview of the paired training regime: <br>
<p align="center">
<img src="imgs/Paired.png" style="width:500px;"/>
</p>

Overview of the unpaired training regime: <br>
<p align="center">
<img src="imgs/Unpaired.png" style="width:500px;"/>
</p>

Combining paired and unpaired: <br>
<p align="center">
<img src="imgs/PUP.png" style="width:500px;"/>
</p>

## Experiments

## Results
<img src="imgs/imgs2.png" style="width:300px;"/>

## Future work

**Acknowledgement**

## Sources
