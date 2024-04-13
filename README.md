# Enhanced Experts with Uncertainty-Aware Routing for Multimodal Sentiment Analysis

The project is a partial implementation of the paper "Enhanced Experts with Uncertainty-Aware Routing for Multimodal Sentiment Analysis"
This project is implemented using the PyTorch framework.


## Dataset preparation

Download the dataset and place it in the "datasets" folder.
The data used in the paper can be obtained through the following links: [MIB](https://github.com/TmacMai/Multimodal-Information-Bottleneck)

## Backbone Model
Download the pre-trained bert and put it in the "prebert" folder.

## Instructions on running experiments

To run our method on CMU-MOSI:

```
python train.py
```


To test our model on MVSA-Single:
```
python evaluate.py
```
