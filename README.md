# Taint
## Andy Chen

## Introduction

This is an exploratory project primarily inspired by two ideas:
* Current deep learning models are, in a sense, fragile. Imagine a trained 
image classifier on animals (it can figure out what animal is in a given 
picture), with excellent performance on a test set. If we take a correctly 
identified image of a lion and tweak the pixels in a strategic way (so that 
to a human, the image looks roughly the same), the classifier might actually 
think that the new image has an owl instead! These tweaked pictures are called 
_adversarial examples_, and they demonstrate how unstable certain machine 
learning models can be.
* Human-created concepts are also fragile; the boundary between opposing 
concepts such as "good" and "evil" is a fine one. For instance, imagine 
we saw Bryce offer to pay for Clay's beer in a convenience store; at first
glance, we may see this deed as a generous one. However, if we notice in 
Clay's voice that he is actually somewhat reluctant to buy that beer, then
we might wonder if Bryce is forcing Clay to buy it, changing our perception
of the situation. In this example, certain nuances completely change the 
way we perceive, or _classify_, the scenario in our heads. However, an AI
may not be able to pick up on these nuances, which may cause conflicts between
AI and people.

This project is devoted to finding adversarial examples for letters of the 
alphabet. Suppose we have an accurate classifier of those letters. If we 
have correctly identified images of the letters "G", "O", "O", and "D", 
then we can find adversarial examples that the model classifies as "E", "V",
"I", and "L". This process is an example of a _targeted attack_; we're fooling
the classifier to 

## Requirements
* Python 3.6 (currently, Tensorflow does not support later versions of Python)
* Pip


## How to Use
First, install the required Python 3 libraries with pip:
```
pip install -r requirements.txt
```
Then, download the EMNIST letters dataset; visit the Kaggle dataset page and
download the `emnist-letters-train.csv` and `emnist-letters-train.csv` files. 

Now, from the repository root directory, run the following command:
```
python src/emnist.py emnist-letters-train.csv emnist-letters-test.csv
```
This will train an alphabet classifier model, which we will attempt to "fool"
with a Generative Adversarial Network


# TODO
* Take custom image input and convert to same format as in EMNIST
* Save the Tensorflow model for future use
* Implement AdvGAN detailed in 
[_Generating Adversarial Examples with Adversarial Networks_](https://arxiv.org/pdf/1801.02610.pdf)
* Refine classifier architecture to improve its accuracy
* Comment functions more thoroughly



