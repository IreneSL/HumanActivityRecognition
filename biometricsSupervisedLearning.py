#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics, preprocessing
from sklearn.tree import DecisionTreeClassifier
import parametersConfig

class supervisedLearning():
	
	def gaussianNaiveBayes(self,trainingFeatures,trainingTarget,testingFeatures,testingTarget):
		''' Gaussian Naive Bayes algorithm. 
			The likelihood of the features is assumed to be Gaussian (mean: 0; standard deviation: 1)
		'''
		print "Gaussian Naive Bayes"
		model = GaussianNB()
		model.fit(trainingFeatures,trainingTarget)
		expected = testingTarget
		predicted = model.predict(testingFeatures)
		self.metricsCalculator(expected,predicted)
		return None


	def supportVectorClassification(self,trainingFeatures,trainingTarget,testingFeatures,testingTarget,library):
		''' Support Vector Classification '''
		print "SVM"
		if library == 'SVC':
			kernels = ['rbf', 'linear', 'poly', 'sigmoid']
			for kernel in kernels:
				print "	Kernel", kernel
				model = SVC(cache_size=parametersConfig.cacheSize, verbose=True, kernel=kernel)
				model.fit(trainingFeatures,trainingTarget)
				expected = testingTarget
				predicted = model.predict(testingFeatures)
				self.metricsCalculator(expected,predicted)
		elif library == 'LinearSVC': # 'LinearSVC is much more efficient'
			model = LinearSVC(verbose=True)
			model.fit(trainingFeatures,trainingTarget)
			expected = testingTarget
			predicted = model.predict(testingFeatures)
			self.metricsCalculator(expected,predicted)
		return None


	def decisionTree(self,trainingFeatures,trainingTarget,testingFeatures,testingTarget):
		''' Decision Tree Classifier '''
		print "Decision Tree"
		model = DecisionTreeClassifier(min_samples_leaf=parametersConfig.minSamplesLeaf, random_state=0)
		model.fit(trainingFeatures,trainingTarget)
		expected = testingTarget
		predicted = model.predict(testingFeatures)
		self.metricsCalculator(expected,predicted)
		return None

	def metricsCalculator(self,expected,predicted):
		''' Computes accuracy classification score, compute confusion matrix to evaluate the accuracy of a classification
			and build a text report showing the main classification metrics
		'''
		print(metrics.accuracy_score(expected,predicted))
		print(metrics.classification_report(expected,predicted))
		print(metrics.confusion_matrix(expected,predicted))
		return None

