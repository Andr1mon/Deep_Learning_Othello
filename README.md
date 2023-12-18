# DeepLearning2PlayOthello


This package contains files as follows:

game.py : This code is used for playing Othello game between two models. The game would be run two times with different colors for each player (different starting player) and it generates a GIF file as the log of each game.

training_[x].py : this code is provided as an example of dataset loader and training a model.
 
networks_[your_student_number].py : This file contains your networks. You can define and compare several models with different class name in this file. You should change the name of this files to your student number, which is unique.

utils.py	: this file contains functions that will be used as the rule and different steps of games.


To prepare all dependencies, it is needed to install all packages in requirement by:
	pip install -r requirements.txt
Or
	pip install scikit-learn pandas tabulate matplotlib h5py scipy tqdm torch torchvision
