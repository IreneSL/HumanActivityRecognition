# README #

### What about this repository? ###

* This project tries to recognize types of body movements based on biometrics signals (linear acceleration, angular velocity, etc.) using supervised learning algorithms (Naive Bayes, SVM and Decision Trees)
* Uses data from a public data repository (**UCI Machine Learning Repository**): <https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones>
* It's built on Python 2.7.10

### How do I get set up? ###

* It's possible to install dependencies by running `pip install -r requirements.txt`. If you don't have `pip`, you can visit [this page](https://pip.pypa.io/en/stable/installing/)
* Now, you are can run the main script by tipping `python activityRecognition.py` on console

### Functions

* Data pre-processing / Feature evaluation
* Feature computation
* Boxplots
* Dimensionality reduction
    * Feature selection (Variance threshold)
    * Feature extraction (PCA)
* Classification (Supervised learning)
    * Stratified sampling
    * Naive Bayes, SVC and Decision Tree

### Who do I talk to? ###

* irene.sanchezl93@gmail.com