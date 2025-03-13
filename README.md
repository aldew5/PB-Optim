# Optimizing PAC-Bayes 
Using natural gradient optimizers to optimize PAC-Bayes bounds. This will be a flexible framework for optimizing different structured posteriors which has applications in many other domains.

To run noisy KFAC on kfactored posterior for 10 epochs:

``python3 train.py --optimizer noisy-kfac --approx kfac --precision float64 --epoch 10``

