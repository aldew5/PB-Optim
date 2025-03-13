# Optimizing PAC-Bayes 
Using natural gradient optimizers to optimize PAC-Bayes bounds. This will be a flexible framework for optimizing different structured posteriors which has applications in many other domains.


You must first run 
``python3 train_mlp.py``
with ``SAVE_WEIGHTS=TRUE`` in  ``config.py`` which will train a 2 layer MLP and save its weights. We will then postprocess these weights to 
compute a PAC-Bayes bound by training the corresponding SNN.

To run noisy KFAC on kfactored posterior for 10 epochs:

``python3 train.py --optimizer noisy-kfac --approx kfac --precision float64 --epoch 10``

