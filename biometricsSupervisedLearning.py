#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

class supervisedLearning():
	
	def naiveBayes(self,trainingFeatures,trainingTarget,testingFeatures,testingTarget):
		print "Naive Bayes"
		model = GaussianNB()
		model.fit(trainingFeatures,trainingTarget)
		expected = testingTarget
		predicted = model.predict(testingFeatures)
		self.metricsCalculator(expected,predicted)
		return None

	def metricsCalculator(self,expected,predicted):
		print(metrics.classification_report(expected,predicted))
		print(metrics.confusion_matrix(expected,predicted))
		return None
