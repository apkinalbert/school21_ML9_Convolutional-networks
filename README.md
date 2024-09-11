# Convolutional Neural Networks

Summary: this project is an introduction to convolutional neural networks and image classification problems.

💡 [Tap here](https://new.oprosso.net/p/4cb31ec3f47a4596bc758ea1861fb624) **to leave your feedback on the project**. It's anonymous and will help our team make your educational experience better. We recommend completing the survey immediately after the project.

## Contents

1. [Chapter I. Preamble](#chapter-i-preamble)
2. [Chapter II. Introduction](#chapter-ii-introduction) \
    2.1. [Convolution operation](#convolution-operation) \
    2.2. [Pooling operation](#pooling-operation) \
    2.3. [First CNN architecture: LeNet](#first-cnn-architecture-lenet) \
    2.4. [CNN usage examples](#cnn-usage-examples) \
    2.5. [Modern CNN architectures](#modern-cnn-architectures) \
    2.6. [Augmentations](#augmentations) \
    2.7. [MixUp & CutMix](#mixup--cutmix) \
    2.8. [Test-Time-Augmentations](#test-time-augmentations)
3. [Chapter III. Goal](#chapter-iii-goal)
4. [Chapter IV. Instructions](#chapter-iv-instructions)
5. [Chapter V. Task](#chapter-v-task)
6. [Chapter VI. Bonus part](#chapter-vi-bonus-part)

## Chapter I. Preamble

We have already discussed different algorithms to handle regression or classification problems with tabular data as input. But for various AI problems (like image or text classification) using such algorithms is suboptimal or even impossible. Now we are going to learn the first type of algorithms devoted to work with spatial data: Convolutional Neural Networks (CNN).

## Chapter II. Introduction

The models that we have discussed so far remain appropriate options when we are solving tabular problems. Tabular data instance is represented by vector **x** and corresponding label **y** (scalar or vector) where components of **x** don’t have any structure: we can permute components of **x** and it would not break any interactions between features and target. Training on a dataset with permuted features will produce the same model (the same up to the random seed of the algorithm) as learning with features in original order.

Sometimes, we need more flexible instruments to process data with inner structure. One such example of data is images: one pixel contains a small amount of information, but pixel values are correlated and form composite objects. In cat/dog image classification tasks cat or dog can be in any part of the input.

![Cat](misc/images/cat.png)

[Source](https://aiplanet.com/learn/getting-started-with-deep-learning/convolutional-neural-networks/267/cnn-transfer-learning-data-augmentation)

We can’t come up with a traditional “tabular” approach in that case because shuffling features (of the input) will break the model. Another way – to introduce complex hand-crafted features and use any tabular classifier. For different input data (cats/dogs, medical data, space imaging, etc.) you need to design a different set of hand-crafted features. It is an option but from previous lessons we remember that neural networks shine in guided feature creation: inputs transform via learnable functions into representations convenient for classification/regression problems. So it makes sense to design special neural networks devoted to effectively working with such types of data: invariant to translations of input.

We call such networks Convolutional Neural Networks or CNNs. This [source](http://d2l.ai/chapter_convolutional-neural-networks/why-conv.html) contains more motivations to use CNN.

### Convolution operation

Imagine we use image pixels as input for MLP. Usual dense layer of the neural network will treat each pixel with its own weight.

![Convolution operation](misc/images/convolution_operation.png)

Objects of interest (for example, cats) can appear anywhere on the image. It makes sense to design modules that respond similarly to the same patch, regardless of where it appears in the image. We want the neurons to “fire” in any location that contains a cat.

![Convolution operation](misc/images/convolution_operation_2.png)

On the image below we apply a 3x3 convolution layer: multiply patch of image on corresponding weights in 3x3 window, add bias term, then slide the window by 1 pixel (number of pixels to shift the window is called **stride**). This convolution has 10 learnable parameters.

Using a linear layer for the whole image will result in usage of 82 parameters: 8 times more of learnable parameters. This [resource](http://d2l.ai/chapter_convolutional-neural-networks/why-conv.html#constraining-the-mlp) shows that convolutions are basically constrained MLP layers: we use less parameters and have a better resulting model quality. This is called **parameter sharing**. Convolution layers are good examples of parameter sharing: different parts of input interact with the same set of parameters. It also can be shown that convolutions are equivalent to linear layers with matrices of special form ([Toeplitz matrices](https://en.wikipedia.org/wiki/Toeplitz_matrix)). Parameter sharing is used in various NN architectures, we will see more examples of that later.

We can use different weights for different channels of input image: separate parameters for Red, Green, Blue channels.

![Multiple maps](misc/images/multiple_maps.png)

Output of the convolutional layer is something that reminds an image: 3D object with height, width, and number of channels as dimensions. Stacking multiple layers of convolutions allows to capture longer-range features of the image (in a way similar to higher level vision in nature).

![io tensor](misc/images/io_tensor.png)

There are different types of convolutions, this [source](https://www.youtube.com/watch?v=shVKhOmT0HE&t=3403s) describes all the details related to convolution variations.

### Pooling operation

First (or the earliest) layers of the NN are focused on local regions (layers don’t pay attention to contents of the image in distant regions). This is called **the locality principle**. Learned local representations are aggregated to make predictions for the whole image. For aggregation we use other kinds of layers called **pooling layers**. Pooling layers consist of a fixed-shape window (for example 3x3) that slides over the image (according to its stride), computing a single output for each location. Almost all the time we use **max** (take max of elements inside the window) or **average** (average elements in the window) pooling.

![pooling](misc/images/pooling.png)

Pooling layer doesn’t have any learnable parameters.

### First CNN architecture: LeNet

In the end of our neural network we flatten the resulting tensor and apply one or more dense layers. The first convolutional network in history, LeNet (1998), follows this approach:

![LeNet](misc/images/lenet.png)

LeNet’s history is described in this [source](https://en.wikipedia.org/wiki/LeNet).

We learn optimal weights of convolution and dense layers using the standard Backpropagation algorithm. We receive a NN model with learnable (convolutional) feature extractors from the image and MLP classifier at the end.

Convolutional and pooling layers also can be adapted to 1D (time series, texts) and 3D objects (brain scans, videos). Next image illustrates a typical 1D CNN model.

![2D and 1D](misc/images/2d_and_1d.png)

To sum up, **Convolutional neural networks (CNNs)** can be defined as any neural networks that contain convolutional-pooling layers.

### Transfer Learning

As any neural network, CNNs are prone to overfitting and require a large amount of training data. For many applications, the huge amount of training data is expensive and not available. **Transfer learning** is a technique to train the network on a larger data set from a related (or, sometimes, even unrelated) domain. Once the network parameters have converged an additional training step is performed using the in-domain training data to fine-tune the network weights. This approach allows CNN models to be applied to almost any problem with a small training set.

The most used dataset used for pretraining CNNs is [ImageNet](https://en.wikipedia.org/wiki/ImageNet).

### CNN usage examples

CNNs are so flexible that we can use them for solving a wide variety of problems. Convolutional networks shine in problems based on grid-like structures: image/video classification, time-series forecasting, audio classification, etc.

Audio classification, for example, is “solved” via transforming audio to image-like structure.

![Audio classification](misc/images/audio_classification.png)

Convolutions were used to win NFL competition on kaggle: participants were challenged with prediction of  the yardage gained on the play in american football game.

![football](misc/images/football.png)

CNNs are considered to be most effective in problems with strong local patterns.

### Modern CNN architectures

This [guide](http://d2l.ai/chapter_convolutional-modern/index.html) gives a comprehensive overview of modern design CNN approaches.

A lot of ideas related to CNN’s performance improvement consist of ways to ease the learning process.

One of the groundbreaking ideas is skip-connection (or residual connection). Residual connection helps CNN to propagate gradients better when using backprop.

![Residual connection](misc/images/residual_connection.png)

This trick allows us to build networks consisting of tens and hundreds of layers: and more layers generally mean better accuracy. Residual connections are used not only in CNNs but in more other architectures.

Another way to propagate information from previous layers is to use dense blocks: instead of addition we concatenate output of the current layer and outputs of all previous layers and use this as input for the next convolutional layer. So, convolutional level 5 takes as input concatenated results of layer 3, layer 2, etc.

![Dense blocks](misc/images/dense_blocks.png)

This [source](https://www.youtube.com/watch?v=shVKhOmT0HE&t=3403s) has good illustrations of numerous modern CNN models.

### Augmentations

To increase the generalization ability of CNN, sometimes we transform training examples in a way that doesn’t corrupt the original target signal. This transformation is called **augmentation**.

Common choices for augmentations are:

- Flipping (horizontal or vertical)
- Cropping (random, central, etc.)
- Blur
- Cutout (zero-out rectangle region of the image)
- Change contrast of the image

For example, in bird classification task we can use following augmentations:

![Bird classification](misc/images/bird_classification.png)

Training with augmentations reduces overfitting of the CNN model.

This [source](http://d2l.ai/chapter_computer-vision/image-augmentation.html#image-augmentation) contains a good overview of augmentations for image problems.

### MixUp & CutMix

There are augmentation approaches that involve mixing different samples. The most famous are MixUp and CutMix.

MixUp is basically a way to force neural networks to learn how to interpolate "between" training instances. To perform MixUp we have to compute convex combination on 2 different samples and corresponding targets:

```math
\begin {align*}
    \tilde{x} &= \lambda x_i + (1 - \lambda) x_j, \quad \text {where } x_i, x_j \text { are raw input vectors} \\
    \tilde{y} &= \lambda y_i + (1 - \lambda) y_j, \quad \text {where } y_i, y_j \text { are one-hot label encodings}
\end {align*}
```

We can use MixUp with any type of data that allows straightforward computation of convex combinations with instances. Following picture illustrates the effect of the MixUp on a toy problem.

![Toy problem](misc/images/toy_problem.png)

Blue shading indicates $`p(y=1|x)`$ green means class 0 and orange means class 1.

CutMix in essence proposes the following augmentation strategy: patches are cut and pasted among training images where the ground truth labels are also mixed proportionally to the area of the patches.

![CutMix](misc/images/CutMix.png)

### Test-Time-Augmentations

It is possible to apply augmentations (that make sense in real-life problems) on inference samples and average predictions. For example, in cat/dog classification we can crop, flip or slightly rotate our image, predict and average to compute the final prediction for a particular sample.

![Augmentations](misc/images/augmentations.png)

## Chapter III. Goal

The goal of this task is to get a deep understanding of the base models for image classification (CNN modules, LeNet, ResNet, DenseNet, etc.).

## Chapter IV. Instructions

This project will only be evaluated by humans. You are free to organize and name
your files as you desire.

- Here and further we use Python 3 as the only correct version of Python.
- For training deep learning algorithms you can try Google Colab. It offers kernels

    (Runtime) with GPU for free that is faster than CPU for such tasks.
- The norm is not applied to this project. Nevertheless, you are asked to be clear
and structured in the conception of your source code.
- Store the datasets in the subfolder data.

## Chapter V. Task

1. Download [Zindi gesture data](https://zindi.africa/competitions/kenyan-sign-language-classification-challenge/data). Research the data for some time. Make sure to get rid of duplicated samples. Are there any wrongly labeled examples? Think what augmentations would be optimal in the current task? Design train-validation split. Use random 33% of the whole data as the validation set.
2. Write custom pytorch dataset class for image and target retrieval: read images using OpenCV2, for target extraction use pandas. Ensure that the dataset class has the correct API for retrieving samples. Create pytorch DataLoader for your datasets.
3. Recreate the LeNet architecture. Use predefined levels from pytorch library: Conv2d, Pooling, Linear. You can use the Dropout layer to increase the quality of your model.
4. Design a basic train-validation loop: iterate over train dataset, batch by batch, update parameters of the network and check quality of the model using validation set. [Here](https://github.com/pytorch/examples/blob/main/mnist/main.py) you can find a comprehensive example of a basic training-validation pipeline (you can copy-paste it at first, then modify). As a loss for your model use Binary Cross Entropy, as a metric use ROC AUC score. Receive ROC AUC higher than 0.75.
5. Pick any vision model or backbone (resnet18 is recommended as a baseline) from this [library](https://github.com/rwightman/pytorch-image-models). Change the head of your model to a linear layer with one output. Train the model for 2-4 epochs (iterations, traversals over the whole train dataset).
You must be able to receive ROC AUC higher than 0.9. You also advised to play with other backbones to have better results.
6. Apply different augmentations from the [albumentations](https://albumentations.ai/) library and check if they improve validation score.
7. Implement MixUp and CutMix augmentations and test them in your pipeline; check if they improve validation score.

## Chapter VI. Bonus part

1. Add Test-Time-Augmentation (TTA) option to your code. For example, try to average predictions of horizontally flipped and original images. What is your uplift in terms of ROC AUC score?

>Please leave feedback on the project in the [feedback form.](https://forms.yandex.ru/cloud/646b47abc09c022a752404b0/) 
