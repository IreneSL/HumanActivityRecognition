#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import parametersConfig, dataInspector, featuresManager, dimensionalityManager, biometricsSupervisedLearning
from datetime import datetime
from sklearn import preprocessing

def activityRecognition():

	DI = dataInspector.dataAnalysis()
	# Loading data
	data = DI.dataLoader('data/HAR_UCI_RAW.txt',',',['userID','activity','experimentID','xAcceleration','yAcceleration','zAcceleration','xAngVelocity','yAngVelocity', 'zAngVelocity'], (int,int,int,float,float,float,float,float,float))
	# Selecting certain activities
	data = DI.activitiesSelector(data, [1,2,3,4,5,6])
	DI.dataSummarizer(data,['Walking', 'Walking upstairs', 'Walking downstairs', 'Sitting', 'Standing', 'Laying'])

	# New features computation
	FM = featuresManager.featuresAgent()
	data = FM.featuresComputation(data, parametersConfig.windowSize)

	# Feautures headers
	headers = parametersConfig.headers
	# Features formats
	formats = ['int'] + (['float'] * (data.shape[1] - 1))
	# Setting headers and formats (for more useful data management)
	dt = {'names':headers, 'formats': formats}
	dataHeaders = np.zeros(data.shape[0], dtype=dt)
	for columnNumber in range(0,data.shape[1]):
		dataHeaders[headers[columnNumber]] = data[:,columnNumber]
	
	# Creating boxplots
	#DI.boxplotPrinter(data,headers[7:len(headers)-1])
	#DI.boxplotPrinter(dataHeaders,headers[7:len(headers)-1])

	del dataHeaders # Saving memory

	# Reducing dimensionality
	MD = dimensionalityManager.manageDimensionality()
	# Feature selection
	featuresSelected = MD.featureSelection(data[:,1:data.shape[1]]) # Not using 'dataHeaders' variable for format pruposes
																	# Don't select activity labels
	# Feature extraction
	featuresExtracted = MD.featureExtraction(featuresSelected) 

	# Adding labels to transformed data
	labels = data[:,0].reshape(data.shape[0],1)
	data = np.column_stack((labels,featuresExtracted))

	del featuresSelected, featuresExtracted # Saving memory

	# Splitting between training and test datasets
	# Stratified sampling (Training: 0.8; Testing: 0.2)
	trainSet = np.empty((0,data.shape[1]))
	testSet = np.empty((0,data.shape[1]))
	for i in range(1,7):
		dataChoosed = DI.trainingAndTestDataChooser(DI.dataClassSelector(data, i), 0.8)
		trainSet = np.append(trainSet, dataChoosed[0], axis=0)
		scaledTrainSet = preprocessing.scale(trainSet) # For SVM purposes (data needs to be scaled)
		testSet = np.append(testSet, dataChoosed[1], axis= 0)
		scaledTestSet = preprocessing.scale(testSet) # For SVM purposes (data needs to be scaled)

	# Train
	trainSetFeatures = trainSet[:,1:trainSet.shape[1]] # Depedent variables
	trainSetTarget = trainSet[:,0] # Independent variable: activity (walking, sitting, laying, etc.)
	scaledTrainSetFeatures = scaledTrainSet[:,1:scaledTrainSet.shape[1]] #
	# Test
	testSetFeatures = testSet[:,1:testSet.shape[1]]
	testSetTarget = testSet[:,0]
	scaledTestSetFeatures = scaledTestSet[:,1:scaledTestSet.shape[1]]

	# ---------------- 3. Machine learning algorithms ----------------
	BSL = biometricsSupervisedLearning.supervisedLearning()
	# 3.1 Naive Bayes
	BSL.gaussianNaiveBayes(trainSetFeatures,trainSetTarget,testSetFeatures,testSetTarget)
	# SVM
	BSL.supportVectorClassification(scaledTrainSetFeatures,trainSetTarget,scaledTestSetFeatures,testSetTarget,parametersConfig.SVMLibrary)
	# Decision Trees
	BSL.decisionTree(trainSetFeatures,trainSetTarget,testSetFeatures,testSetTarget)

if __name__ == '__main__':
	activityRecognition()