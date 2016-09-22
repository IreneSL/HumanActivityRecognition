#!/usr/bin/python
# -*- coding: utf-8 -*-
import parametersConfig
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics, preprocessing
from sklearn.tree import DecisionTreeClassifier

class supervisedLearning():

	def dataClassSelector(self,data,activity):
		""" Picks data linked to a body movement

		Keywords arguments:
		data -- entire dataset
		activity -- activity key
		"""
		selectedRows = data[np.where(data[:,0] == activity)]
		return selectedRows

	def trainingAndTestDataChooser(self,data,trainingRatio):
		""" Split data into train and test

		Keywords arguments:
		data -- entire dataset
		trainingRatio -- ratio for split between training and testing sets
		"""
		trainSize = int(len(data) * trainingRatio)
		trainSet = data[:trainSize,:,]
		testSet = data[trainSize:data.shape[0]:,]
		return [trainSet, testSet]

	def standardTrainingAndPrediction(self,model,trainingFeatures,trainingTarget,testingFeatures,testingTarget):
		""" Standard format for training and prediction

		Keywords arguments:
		model -- algorithm model
		trainingFeatures -- features for training
		trainingTarget -- target activities for training
		testingFeatures -- features for testing
		testingTarget -- target activities for testing
		"""
		model.fit(trainingFeatures,trainingTarget)
		expected = testingTarget
		predicted = model.predict(testingFeatures)
		return (expected,predicted)
	
	def gaussianNaiveBayes(self,trainingFeatures,trainingTarget,testingFeatures,testingTarget):
		""" Gaussian Naive Bayes algorithm. 
			The likelihood of the features is assumed to be Gaussian (mean: 0; standard deviation: 1)

		Keywords arguments:
		trainingFeatures -- features for training
		trainingTarget -- target activities for training
		testingFeatures -- features for testing
		testingTarget -- target activities for testing
		"""
		print "- 1. Gaussian Naive Bayes"
		model = GaussianNB()
		expected, predicted = self.standardTrainingAndPrediction(model,trainingFeatures,trainingTarget,testingFeatures,testingTarget)
		self.metricsCalculator(expected,predicted)
		return None

	def supportVectorClassification(self,trainingFeatures,trainingTarget,testingFeatures,testingTarget,library):
		""" Support Vector Classification algorithm 

		Keywords arguments:
		trainingFeatures -- features for training
		trainingTarget -- target activities for training
		testingFeatures -- features for testing
		testingTarget -- target activities for testing
		"""
		print "- 2. SVM"
		# 'SVC' library implments differents kinds of kernels (polynomial, linear, Gaussian RBF, etc.)
		if library == 'SVC':
			kernels = ['rbf', 'linear', 'poly', 'sigmoid']
			for kernel in kernels:
				print "	Kernel", kernel
				model = SVC(cache_size=parametersConfig.cacheSize, kernel=kernel, verbose=True)
				expected, predicted = self.standardTrainingAndPrediction(model,trainingFeatures,trainingTarget,testingFeatures,testingTarget)
				self.metricsCalculator(expected,predicted)
		# 'LinearSVC library is specific to linear kernel, but is much more efficient than 'SVC' library
		elif library == 'LinearSVC': 
			model = LinearSVC(verbose=True)
			expected, predicted = self.standardTrainingAndPrediction(model,trainingFeatures,trainingTarget,testingFeatures,testingTarget)
			self.metricsCalculator(expected,predicted)
		return None

	def decisionTree(self,trainingFeatures,trainingTarget,testingFeatures,testingTarget):
		""" Decision Tree Classifier

		Keywords arguments:
		trainingFeatures -- features for training
		trainingTarget -- target activities for training
		testingFeatures -- features for testing
		testingTarget -- target activities for testing
		"""
		print "- 3. Decision Tree"
		model = DecisionTreeClassifier(min_samples_leaf=parametersConfig.minSamplesLeaf, random_state=0)
		expected, predicted = self.standardTrainingAndPrediction(model,trainingFeatures,trainingTarget,testingFeatures,testingTarget)
		self.metricsCalculator(expected,predicted)
		return None

	def metricsCalculator(self,expected,predicted):
		""" Computes accuracy, precision and recall classification score

		Keywords arguments:
		expected -- real activities
		predicted -- activities predicted by model
		"""
		print "+ Accuracy classification score:", metrics.accuracy_score(expected,predicted)
		print(metrics.classification_report(expected,predicted))
		return None

