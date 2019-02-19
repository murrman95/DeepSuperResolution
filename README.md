# Super Resolution Image Reconstruction using a Deep Learning Architecture

[INF573](http://www.enseignement.polytechnique.fr/informatique/INF573/) Image Analysis (2018-2019, M. Renaud Keriven, [Ecole Polytechnique](https://www.polytechnique.edu/)) - *Final project* - **Guillaume Dufau, Christopher Murray**

## Description

Implementation of a very deep CNN model for image super resolution, and small adaptation for multi-images input.

Please find in a pdf format the report for this project.

Here is the paper we based our work on, and added few features (like multi-image input option):
Accurate Image Super-Resolution Using Very Deep Convolutional Networks
[arXiv link](https://arxiv.org/pdf/1511.04587.pdf)

## Getting Started

The project is composed of two main Python files, run.py and test.py. 
The former train the CNN using the given images, the latter save test images in concatenated format allowing comparison between ground truth and generated image (in SR folder).

Fill free to modify the config.py file in order to use multi-image input. You can also save advancement during training.

### Prerequisites

Libraries used:
* Tensorflow
* Scikit-learn
* matplotlib
* OpenCV 2

## Authors

* **Guillaume Dufau** - [GuillaumeDufau](https://github.com/GuillaumeDufau)
* **Christopher Murray** - [murrman95](https://github.com/murrman95)
