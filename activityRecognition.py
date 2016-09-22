#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import parametersConfig, dataInspector, featuresManager, dimensionalityManager, biometricsSupervisedLearning
from sklearn import preprocessing

def activityRecognition():

	# 1 ---------------------------- Pre-processing / Feature evaluation  ----------------------------
	print "\n 1 ---------------------------- Pre-processing / Feature evaluation  ----------------------------"
	DI = dataInspector.dataAnalysis()
	# Loading data
	print "**** Loading data ****"
	data = DI.dataLoader('data/HAR_UCI_RAW.txt',',',['userID','activity', 'experimentID','xAcceleration', \
		'yAcceleration','zAcceleration','xAngVelocity','yAngVelocity', 'zAngVelocity'], \
		(int,int,int,float,float,float,float,float,float))
	# Selecting certain activities (walking, walking upstairs, walking downstais, standing and laying)
	print "**** Choosing data linked walking, walking upstairs, walking downstais, standing and laying activities **** "
	data = DI.activitiesSelector(data, [1,2,3,4,5,6])
	# Checkif dataset is balanced or imbalanced
	print " **** Activities distribution **** "
	DI.dataSummarizer(data,['Walking', 'Walking upstairs', 'Walking downstairs', 'Sitting', 'Standing', 'Laying'])

	# 2 ------------------------------------- Feature computation ------------------------------------
	print "\n 2 ------------------------------------- Feature computation ------------------------------------"
	FM = featuresManager.featuresAgent()
	# New features computation
	print "**** New features computation ****"
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
	# Creating boxplots for feautures computed
	print "**** Creating boxplots for feautures computed ****"
	DI.boxplotPrinter(dataHeaders,headers[7:len(headers)-1])

	del dataHeaders # Saving memory

	# 3 ---------------------------------- Dimensionality reduction ----------------------------------
	print "\n 3 ---------------------------------- Dimensionality reduction ----------------------------------"
	MD = dimensionalityManager.manageDimensionality()
	# Feature selection (Removing all low-variance features)
	print "**** Features selection ****"
	featuresSelected = MD.featureSelection(data[:,1:data.shape[1]],headers[1:len(data)]) 
	# Feature extraction (PCA)
	print "**** Features extraction ****"
	featuresExtracted = MD.featureExtraction(featuresSelected) 
	# Adding labels to transformed data
	labels = data[:,0].reshape(data.shape[0],1)
	data = np.column_stack((labels,featuresExtracted))

	del featuresSelected, featuresExtracted # Saving memory

	# 4 ---------------------------- Classification (Supervised Learning) ----------------------------
	print "\n 4 ---------------------------- Classification (Supervised Learning) ----------------------------"
	BSL = biometricsSupervisedLearning.supervisedLearning()
	# Splitting between training and test datasets
	# Stratified sampling (Training: 0.8; Testing: 0.2)
	print "**** Splitting between training and test datasets. Stratified sampling (Training: 0.8; Testing: 0.2) ****"
	trainSet = np.empty((0,data.shape[1]))
	testSet = np.empty((0,data.shape[1]))
	# For each activity, 80% are for training and 20% for testing
	for i in range(1,7):
		dataChoosed = BSL.trainingAndTestDataChooser(BSL.dataClassSelector(data, i), 0.8)
		trainSet = np.append(trainSet, dataChoosed[0], axis=0)
		testSet = np.append(testSet, dataChoosed[1], axis= 0)
		# For SVM purposes (data needs to be scaled)
		scaledTrainSet = preprocessing.scale(trainSet)
		scaledTestSet = preprocessing.scale(testSet) 

	# Train set
	trainSetFeatures = trainSet[:,1:trainSet.shape[1]] # Depedent variables 
	trainSetTarget = trainSet[:,0] # Independent variable: activity (walking, sitting, laying, etc.)
	# For SVM purposes (data needs to be scaled)
	scaledTrainSetFeatures = scaledTrainSet[:,1:scaledTrainSet.shape[1]]
	# Test set
	testSetFeatures = testSet[:,1:testSet.shape[1]] # Depedent variables 
	testSetTarget = testSet[:,0] # Independent variable: activity (walking, sitting, laying, etc.)
	# For SVM purposes (data needs to be scaled)
	scaledTestSetFeatures = scaledTestSet[:,1:scaledTestSet.shape[1]]

	# Using differents algorithms
	# Naive Bayes
	print "**** Supervised Learning Algorithms****"
	BSL.gaussianNaiveBayes(trainSetFeatures,trainSetTarget,testSetFeatures,testSetTarget)
	# SVC
	BSL.supportVectorClassification(scaledTrainSetFeatures,trainSetTarget,scaledTestSetFeatures,testSetTarget,parametersConfig.SVMLibrary)
	# Decision Trees
	BSL.decisionTree(trainSetFeatures,trainSetTarget,testSetFeatures,testSetTarget)

if __name__ == '__main__':
	activityRecognition()