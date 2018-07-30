# Taint
## Andy Chen

An exploratory project primarily inspired by two ideas:
* Current deep learning models are, in a sense, fragile. Imagine a trained 
image classifier on animals, with excellent performance on a test set. If we 
take a correctly identified image of a lion and tweak the pixels in a strategic
way (so that to a human, the image looks roughly the same), the classifier 
might actually think that the new image has an owl instead! These tweaked 
pictures are called _adversarial examples_, and they demonstrate how 
unstable certain machine learning models can be.
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
"I", and "L". 


# TODO
* Process EMNIST data
* Decide on classifier architecture and implement it
* Implement AdvGAN detailed in 
[_Generating Adversarial Examples with Adversarial Networks_](https://arxiv.org/pdf/1801.02610.pdf)



