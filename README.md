This is a repository for research on indoor localization based on wireless fingerprinting techniques.

# Intorduction
 - **Background**
 
   The most widely used positioning system is global positioning system (GPS). However, the radio signals of satellites are too weak to penetrate buildings and walls, thus GPS is not applicable in indoor environment.
 - **Indoor localization**
 
   Typical indoor localization algorithms mainly include trilateration or triangulation and maximum likelihood techniques. The common ranging methods are as follows: Time of Arrival (TOA), Time Difference of Arrival (TDOA), Angle of Arrival (AOA) and Received Signal Strength Indicator (RSSI) ranging method.
 - **Received signal strengths (RSS)**

   Due to complex indoor environment and the limits of access points (APs), the positioning technology of (RSSI) is the main technology of WiFi localization.
 - **Problems**
 
   Due to complex indoor environment, RSSs are easily affected by multipath effect, cofrequency radio interference, the absorption of human body and other ambient changes. Therefore, RSSs fluctuate with time even at a fixed location, which causes inaccurate positioning results. Meanwhile, the positions estimated by WiFi fingerprinting method are independent and discontinuous in real-time tracking.

# Methodology
**1. Experimental Data**

- UJIIndoorLoc dataset
- Total 19937 samples separated in training, validation, and test data.
- Each scan in the database contains 529 attributes. 

**2. stacked Auto Encoder (SAE)**

Since the DNN architecture is not optimized enough, we can use stacked auto encoders for this task and provide raw measurements at DNN input. Stacked auto encoders (SAE) are parts of the deep network used to reduce the dimensionality of the input data by learning the reduced representation of the original data during unsupervised training.

![](images/1.jpg)

# **Methodology**

-   Implement [a new program](./python/bf_classification.py), which calculates accuracies separately for building and floor classification, to investigate the hierarchical nature of the classification problem at hand; the deep-learning-based place recognition system described in the key paper<sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup> does not take into account this and carries out classification based on flattened labels (i.e., (building, floor) -> 'building-floor'). We are now considering two options to guarantee 100% accuracy for the building classification:
    -   Hierarchical classifier with a tree structure and multiple classifiers and data sets, which is a conventional approach and a reference for this investigation.
    -   One classifier with a weighted loss function<sup><a id="fnr.2" class="footref" href="#fn.2">2</a></sup>. In our case, however, the loss function does not give a closed-form gradient function, which forces us to use evolutionary algorithms (e.g., [genetic algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm)) for training of neural network weights or [multi-label classification with different class weights](https://github.com/fchollet/keras/issues/741) (i.e., higher weights for buildings in our case).


# **Results**

-   Today, we further simplified the building/floor classification system by removing a hidden layer from the classifier (therefore no dropout), resulting in the configuration of '520-64-4-13' (including input and output layers) with loss=7.050603e-01 and accuracy=9.234923e-01 ([results](./results/indoor_localization_deep_learning_out_20170815-203448.org)). This might mean that the 4-dimensional data from the SAE encoder (64-4) can be linearly separable. Due to training of SAE encoder weights for the combined system, however, it needs further investigation.


# **Conclusion**

-   We investigated whether a couple of strong RSSs in a fingerprint dominate the classification performance in building/floor classification. After many trials with different configurations, we could obtain more than 90% accuracies with the stacked-autoencoder (SAE) having 64-4-64 hidden layers (i.e., just 4 dimension) and the classifier having just one 128-node hidden layer ([results](./results/indoor_localization_deep_learning_out_20170814-184009.org)). This implies that a small number of RSSs from access points (APs) deployed in a building/floor can give enough information for the building/floor classification; the localization on the same floor, by the way, would be quite different, where RSSs from possibly many APs have a significant impact on the localization performance.


# 2017-08-13

-   We finally obtained [more than 90% accuracies](./results/indoor_localization_deep_learning.org) from [this version](./python/indoor_localization_deep_learning.py), which are comparable to the results of the key paper <sup><a id="fnr.1.100" class="footref" href="#fn.1">1</a></sup> based on the [UJIIndoorLoc Data Set](https://archive.ics.uci.edu/ml/datasets/ujiindoorloc); refer to the [multi-class clarification example](https://keras.io/getting-started/sequential-model-guide/#compilation) for classifier parameter settings.
-   We [replace the activation functions of the hidden-layer from 'tanh' to 'relu'](./python/indoor_localization-2.ipynb) per the second answer to [this question](https://stats.stackexchange.com/questions/218542/which-activation-function-for-output-layer) ([results](./results/indoor_localization-2_20170813.csv)). Compared to the case with 'tanh', however, the results seem to not improve (a bit in line with the gut-feeling suggestions from [this](https://datascience.stackexchange.com/questions/10048/what-is-the-best-keras-model-for-multi-class-classification)).


# 2017-08-12

-   We first tried [a feed-forward classifier with just one hidden layer](./python/indoor_localization-1.ipynb) per the comments from [this](https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw) ([results](./results/indoor_localization-1_20170812.csv)). (\* *nh*: number of hidden layer nodes, *dr*: [dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) rate, *loss*: [categorical cross-entropy](http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#theano.tensor.nnet.nnet.categorical_crossentropy), *acc*: accuracy \*).

## Footnotes

<sup><a id="fn.1" class="footnum" href="#fnr.1">1</a></sup> M. Nowicki and J. Wietrzykowski, "Low-effort place recognition with WiFi fingerprints using deep learning," arXiv:1611.02049v2 [cs.RO] [(arXiv)](https://arxiv.org/abs/1611.02049v2)

<sup><a id="fn.2" class="footnum" href="#fnr.2">2</a></sup> T. Yamashita et al., "Cost-alleviative learning for deep convolutional neural network-based facial part labeling," *IPSJ Transactions on Computer Vision and Applications*, vol. 7, pp. 99-103, 2015. [(DOI)](http://doi.org/10.2197/ipsjtcva.7.99)
