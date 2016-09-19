#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import parametersConfig, dataInspector, featuresManager, dimensionalityManager

#class BiometricsSupervisedLearning():

def activityRecognition():

	DI = dataInspector.dataAnalysis()
	# Loading data
	data = DI.dataLoader('data/HAR_UCI_RAW.txt',',',['userID','activity','experimentID','xAcceleration','yAcceleration','zAcceleration','xAngVelocity','yAngVelocity', 'zAngVelocity'], (int,int,int,float,float,float,float,float,float))
	# Selecting certain activities
	dataFiltered = DI.activitiesSelector(data, [1,2,3,4,5,6])
	DI.dataSummarizer(dataFiltered,['Walking', 'Walking upstairs', 'Walking downstairs', 'Sitting', 'Standing', 'Laying'])

	# New features computation
	FM = featuresManager.featuresAgent()
	dataWithAddedFeatures = FM.featuresComputation(dataFiltered, parametersConfig.windowSize)

	# Feautures headers
	headers = parametersConfig.headers

	# Features formats
	formats = ['int'] + (['float'] * (dataWithAddedFeatures.shape[1] - 1))
	# Setting headers and formats (for more useful data management)
	dt = {'names':headers, 'formats': formats}
	dataWithAddedFeaturesAndHeaders = np.zeros(dataWithAddedFeatures.shape[0], dtype=dt)
	for columnNumber in range(0,dataWithAddedFeatures.shape[1]):
		dataWithAddedFeaturesAndHeaders[headers[columnNumber]] = dataWithAddedFeatures[:,columnNumber]

	data = dataWithAddedFeaturesAndHeaders

	# Saving memory
	savingMemory([dataFiltered,dataWithAddedFeaturesAndHeaders])
	
	# Creating boxplots
	DI.boxplotPrinter(data,headers[7:len(headers)-1])

	# Reducing dimensionality
	MD = dimensionalityManager.manageDimensionality()
	# Feature selection
	dataFeatureSelection = MD.featureSelection(dataWithAddedFeatures)
	# Feature extraction
	dataTransformed = MD.featureExtraction(dataFeatureSelection[:,1:]) # Not using 'data' variable for format pruposes
	# Adding labels to transformed data
	#dataTransformedWithLabels = np.array((dataWithAddedFeatures[:,0], dataTransformed[:,0]))

	'''# Splitting between training and test datasets
	# Stratified sampling (Training: 0.8; Testing: 0.2)
	trainSet = np.empty((0,dataTransformedWithLabels.shape[1]))
	testSet = np.empty((0,dataTransformedWithLabels.shape[1]))
	for i in range(1,6):
		dataChoosed = DI.trainingAndTestDataChooser(DI.dataClassSelector(dataTransformedWithLabels, i), 0.8)
		trainSet = np.append(trainSet, dataChoosed[0], axis= 0)
		testSet = np.append(testSet, dataChoosed[1], axis= 0)

	print "train"
	print trainSet.shape
	print trainSet
	print "test"
	print testSet.shape
	print testSet'''


def savingMemory(variables):
	for variable in variables:
		del variable
	
if __name__ == '__main__':
	activityRecognition()