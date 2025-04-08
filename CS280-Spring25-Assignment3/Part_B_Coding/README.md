# Homework3

Name:

Student ID: 

This assignment is due on **Thursday, April 28 2025** at 11:59pm GMT+8.

## Instructions

- [Homework3](#homework3)
  - [Instructions](#instructions)
    - [Goals](#goals)
    - [Setup](#setup)
    - [Quick Start](#quick-start)
    - [✅Q1: Image Captioning with Vanilla RNNs (26 points)](#q1-image-captioning-with-vanilla-rnns-26-points)
    - [✅Q2: Image Captioning with LSTMs (22 points)](#q2-image-captioning-with-lstms-22-points)
    - [✅Q3: Image Captioning with Transformers (22 points)](#q3-image-captioning-with-transformers-22-points)
    - [✅Q4: Network Visualization: Saliency maps, Class Visualization, and Fooling Images (10 points)](#q4-network-visualization-saliency-maps-class-visualization-and-fooling-images-10-points)
    - [Submitting your work](#submitting-your-work)

### Goals

In this assignment, you will implement recurrent neural networks and apply them to image captioning on the Microsoft COCO data. You will also explore methods for visualizing the features of a pretrained model on ImageNet. 

The goals of this assignment are as follows:

- Understand the architecture of recurrent neural networks (RNNs) and how they operate on sequences by sharing weights over time.
- Understand and implement Vanilla RNNs, Long-Short Term Memory (LSTM) and Transformer networks.
- Understand how to combine convolutional neural nets and recurrent nets to implement an image captioning system.
- Explore various applications of image gradients, including saliency maps, fooling images, class visualizations.

### Setup

You should be able to use your setup from assignments 1 and 2.

**Ensure you have followed the [setup instructions](https://cs231n.github.io/setup-instructions/) before proceeding.**

**Install Packages**. Once you have the starter code, activate your environment and run `pip install -r requirements.txt`.

### Quick Start
Copy codes into your beginning of jupyter notebook and run them.

```python
# set workspace and get datasets
from google.colab import drive
drive.mount('/content/drive')
FOLDERNAME = "cs231n/assignments/assignment3/"
assert FOLDERNAME is not None, "[!] Enter the foldername."
import sys
sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))
%cd /content/drive/My\ Drive/$FOLDERNAME/cs231n/datasets/
!bash get_datasets.sh
%cd /content/drive/My\ Drive/$FOLDERNAME
```

**Download data**. Next, you will need to download the COCO captioning data, a pretrained SqueezeNet model (for TensorFlow), and a few ImageNet validation images. Run the following from the `assignment3` directory:

```bash
cd cs231n/datasets
./get_datasets.sh
```

**Start Jupyter Server**. After you've downloaded the data, you can start the Jupyter server from the `assignment3` directory by executing `jupyter notebook` in your terminal.

Complete each notebook, then once you are done, go to the [submission instructions](#submitting-your-work).

### ✅Q1: Image Captioning with Vanilla RNNs (26 points)

The notebook `RNN_Captioning.ipynb` will walk you through the implementation of an image captioning system on MS-COCO using vanilla recurrent networks.

### ✅Q2: Image Captioning with LSTMs (22 points)

The notebook `LSTM_Captioning.ipynb` will walk you through the implementation of Long-Short Term Memory (LSTM) RNNs, and apply them to image captioning on MS-COCO.

### ✅Q3: Image Captioning with Transformers (22 points)

The notebook `Transformer_Captioning.ipynb` will walk you through the implementation of a Transformer model and apply it to image captioning on COCO.

### ✅Q4: Network Visualization: Saliency maps, Class Visualization, and Fooling Images (10 points)

The notebook `NetworkVisualization-PyTorch.ipynb` will introduce the pretrained SqueezeNet model, compute gradients with respect to images, and use them to produce saliency maps and fooling images.

### Submitting your work

**Important**. Please make sure that the submitted notebooks have been run and the cell outputs are visible.

Once you have completed all notebooks and filled out the necessary code, there are **_two_** steps you must follow to submit your assignment:

**1.** You must have (a) `nbconvert` installed with Pandoc and Tex support and (b) `PyPDF2` installed to successfully convert your notebooks to a PDF file. Please follow these [installation instructions](https://nbconvert.readthedocs.io/en/latest/install.html#installing-nbconvert) to install (a) and run `pip install PyPDF2` to install (b). If you are, for some inexplicable reason, unable to successfully install the above dependencies, you can manually convert each jupyter notebook to HTML (`File -> Download as -> HTML (.html)`), save the HTML page as a PDF.

**2.** Submit the PDFs for each ipynb to gradescope. Upload the zip folder to pan.shanghaitech.edu.cn.
