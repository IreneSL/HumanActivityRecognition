#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import parametersConfig
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import os

class dataInspector():

	def dataLoader(self, inputFile, delimiter, headers, dataTypes):
		''' Loads a file as data '''
		data = np.genfromtxt(inputFile, delimiter=delimiter, names=headers, dtype=dataTypes)
		return data

	def variablesSelector(self, data, variablesToChoose):
		dataChosenVariables = data[variablesToChoose]
		return dataChosenVariables

	def activitiesSelector(self, data, activities):
		activityInstruction = "data[np.where("
		counter = 0
		for activity in activities:
			counter += 1
			activityInstruction += "(data['activity']==" + str(activity)
			if counter == len(activities):
				activityInstruction += "))]"
			else:
				activityInstruction += ") | "

		dataChosenActivities = eval(activityInstruction)
		return dataChosenActivities

	def dataSummarizer(self,data,activitiesNames):
		''' Indicates which percentage of data is about a certain movement '''
		counter = 0
		for activity in range(1,7):
			selectedRows = data[np.where(data['activity'] == activity)]
			classPercentage = round(float(selectedRows.shape[0])/float(int(data.shape[0])),2) * 100
			print '-', activitiesNames[counter], ':', classPercentage, '%'
			counter += 1
		return None

	def dataClassSelector(self,data,column):
		''' Pick data linked to a body movement '''
		selectedRows = data[np.where(data[:,1] == column)]
		print "Longitud datos iniciales"
		print data.shape[0]
		print "Longitud datos seleccionados"
		print selectedRows.shape[0]
		return selectedRows

	def trainingAndTestDataChooser(self,data,trainingRatio):
		''' Split data into train and test '''
		trainSize = int(len(data) * trainingRatio)
		print "Size train", trainSize
		trainSet = data[:trainSize,:,]
		testSet = data[trainSize:data.shape[0]:,]
		return [trainSet, testSet]

	def boxplotPrinter(self, data, features):
		activities = np.unique(data['activity'])
		for activity in activities:
			activityData = data[np.where(data['activity'] == activity)]
			for feature in features:
				featureValues = activityData[feature]
				figure = plt.figure(1, figsize=(9, 6))
				ax = figure.add_subplot(111)
				bp = ax.boxplot(featureValues, patch_artist=True)
				## change outline color, fill color and linewidth of the boxes
				for box in bp['boxes']:
					# change outline color
					box.set( color='#7570b3', linewidth=2)
					# change fill color
					box.set( facecolor = '#1b9e77' )
				## change color and linewidth of the whiskers
				for whisker in bp['whiskers']:
					whisker.set(color='#7570b3', linewidth=2)
				## change color and linewidth of the caps
				for cap in bp['caps']:
					cap.set(color='#7570b3', linewidth=2)
				## change color and linewidth of the medians
				for median in bp['medians']:
					median.set(color='#b2df8a', linewidth=2)
				## change the style of fliers and their fill
				for flier in bp['fliers']:
					flier.set(marker='o', color='#e7298a', alpha=0.5)
				## Custom x-axis labels
				ax.set_xticklabels([feature])
				## Remove top axes and right axes ticks
				ax.get_yaxis().tick_left()

				directory = 'img/'+'activity_'+str(activity)
				if not os.path.exists(directory):
					os.makedirs(directory)
				figure.savefig(directory+'/feature_'+feature,bbox_inches='tight')
				figure.clf()

	def variableDescriptor(self,data,headers):
		for header in headers:
			print "- Column", header
			print "		+ Max", np.max(data[header])
			print "		+ Min", np.min(data[header])
			print "------------"


class featureGenerator():

	def featuresComputation(self,data,windowSize):
		# Existing features
		activitiesList, xAccList, yAccList, zAccList, xAngVelList, yAngVelList, zAngVelList = [], [], [], [], [], [], []
		# New features
		# 1. Means
		xAccMeanList, yAccMeanList, zAccMeanList, xAngVelMeanList, yAngVelMeanList, zAngVelMeanList = [], [], [], [], [], []
		# 2. Medians
		xAccMedianList, yAccMedianList, zAccMedianList, xAngVelMedianList, yAngVelMedianList, zAngVelMedianList = [], [], [], [], [], []
		# 3. Max
		xAccMaxList, yAccMaxList, zAccMaxList, xAngVelMaxList, yAngVelMaxList, zAngVelMaxList = [], [], [], [], [], []
		# 4. Min
		xAccMinList, yAccMinList, zAccMinList, xAngVelMinList, yAngVelMinList, zAngVelMinList = [], [], [], [], [], []
		# 5. MinMax (difference between maximum and minimum)
		xAccMinMaxList, yAccMinMaxList, zAccMinMaxList, xAngVelMinMaxList, yAngVelMinMaxList, zAngVelMinMaxList = [], [], [], [], [], []
		# 6. Standard deviation
		xAccStdList, yAccStdList, zAccStdList, xAngVelStdList, yAngVelStdList, zAngVelStdList = [], [], [], [], [], []
		# 7. First quartile
		xAcc1QList, yAcc1QList, zAcc1QList, xAngVel1QList, yAngVel1QList, zAngVel1QList = [], [], [], [], [], []
		# 8. Third quartile
		xAcc3QList, yAcc3QList, zAcc3QList, xAngVel3QList, yAngVel3QList, zAngVel3QList = [], [], [], [], [], []
		# 9. Average time between peaks
		xAccTBMaxPList, yAccTBMaxPList, zAccTBMaxPList, xAngVelTBMaxPList, yAngVelTBMaxPList, zAngVelTBMaxPList = [], [], [], [], [], []
		xAccTBMinPList, yAccTBMinPList, zAccTBMinPList, xAngVelTBMinPList, yAngVelTBMinPList, zAngVelTBMinPList = [], [], [], [], [], []
		# 10. Peak frecuency
		xAccPKMaxList, yAccPKMaxList, zAccPKMaxList, xAngVelPKMaxList, yAngVelPKMaxList, zAngVelPKMaxList = [], [], [], [], [], []
		xAccPKMinList, yAccPKMinList, zAccPKMinList, xAngVelPKMinList, yAngVelPKMinList, zAngVelPKMinList = [], [], [], [], [], []
		# 11. Correlations between signals axes
		corAccXYList, corAccXZList, corAccYZList, corAngVelXYList, corAngVelXZList, corAngVelYZList = [], [], [], [], [], []

		users = np.unique(data['userID'])
		for user in users:
			print "- USER", user
			userData = data[np.where(data['userID'] == user)]
			experiments = np.unique(userData['experimentID'])
			for experiment in experiments:
				print "	+ EXPERIMENT", experiment
				userExperimentData = userData[np.where(userData['experimentID'] == experiment)]
				activities = np.unique(userExperimentData['activity'])
				for activity in activities:
					print "		· ACTIVITY", activity
					userExperimentActivityData = userExperimentData[np.where(userExperimentData['activity'] == activity)]
					print "			* Rows", userExperimentActivityData.shape
					if userExperimentActivityData.shape[0] >= windowSize:
						sample = userExperimentActivityData[0:windowSize]
					else:
						sample = userExperimentActivityData

					# Existing features
					activitiesList = self.appendToList(activitiesList,userExperimentActivityData['activity'])
					xAccList = self.appendToList(xAccList,userExperimentActivityData['xAcceleration'])
					yAccList = self.appendToList(yAccList,userExperimentActivityData['yAcceleration'])
					zAccList = self.appendToList(zAccList,userExperimentActivityData['zAcceleration'])
					xAngVelList = self.appendToList(xAngVelList,userExperimentActivityData['xAngVelocity'])
					yAngVelList = self.appendToList(yAngVelList,userExperimentActivityData['yAngVelocity'])
					zAngVelList = self.appendToList(zAngVelList,userExperimentActivityData['zAngVelocity'])
					# New features
					# 1. Means
					xAccMeanList = self.appendToList(xAccMeanList, [np.mean(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					yAccMeanList = self.appendToList(yAccMeanList, [np.mean(sample['yAcceleration'])] * userExperimentActivityData.shape[0])
					zAccMeanList = self.appendToList(zAccMeanList, [np.mean(sample['zAcceleration'])] * userExperimentActivityData.shape[0])
					xAngVelMeanList = self.appendToList(xAngVelMeanList, [np.mean(sample['xAngVelocity'])] * userExperimentActivityData.shape[0])
					yAngVelMeanList = self.appendToList(yAngVelMeanList, [np.mean(sample['yAngVelocity'])] * userExperimentActivityData.shape[0])
					zAngVelMeanList = self.appendToList(zAngVelMeanList, [np.mean(sample['zAngVelocity'])] * userExperimentActivityData.shape[0])
					# 2. Medians
					xAccMedianList = self.appendToList(xAccMedianList, [np.median(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					yAccMedianList = self.appendToList(yAccMedianList, [np.median(sample['yAcceleration'])] * userExperimentActivityData.shape[0])
					zAccMedianList = self.appendToList(zAccMedianList, [np.median(sample['zAcceleration'])] * userExperimentActivityData.shape[0])
					xAngVelMedianList = self.appendToList(xAngVelMedianList, [np.median(sample['xAngVelocity'])] * userExperimentActivityData.shape[0])
					yAngVelMedianList = self.appendToList(yAngVelMedianList, [np.median(sample['yAngVelocity'])] * userExperimentActivityData.shape[0])
					zAngVelMedianList = self.appendToList(zAngVelMedianList, [np.median(sample['zAngVelocity'])] * userExperimentActivityData.shape[0])
					# 3. Max
					xAccMaxList = self.appendToList(xAccMaxList, [np.max(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					yAccMaxList = self.appendToList(yAccMaxList, [np.max(sample['yAcceleration'])] * userExperimentActivityData.shape[0])
					zAccMaxList = self.appendToList(zAccMaxList, [np.max(sample['zAcceleration'])] * userExperimentActivityData.shape[0])
					xAngVelMaxList = self.appendToList(xAngVelMaxList, [np.max(sample['xAngVelocity'])] * userExperimentActivityData.shape[0])
					yAngVelMaxList = self.appendToList(yAngVelMaxList, [np.max(sample['yAngVelocity'])] * userExperimentActivityData.shape[0])
					zAngVelMaxList = self.appendToList(zAngVelMaxList, [np.max(sample['zAngVelocity'])] * userExperimentActivityData.shape[0])
					# 4. Min
					xAccMinList = self.appendToList(xAccMinList, [np.min(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					yAccMinList = self.appendToList(yAccMinList, [np.min(sample['yAcceleration'])] * userExperimentActivityData.shape[0])
					zAccMinList = self.appendToList(zAccMinList, [np.min(sample['zAcceleration'])] * userExperimentActivityData.shape[0])
					xAngVelMinList = self.appendToList(xAngVelMinList, [np.min(sample['xAngVelocity'])] * userExperimentActivityData.shape[0])
					yAngVelMinList = self.appendToList(yAngVelMinList, [np.min(sample['yAngVelocity'])] * userExperimentActivityData.shape[0])
					zAngVelMinList = self.appendToList(zAngVelMinList, [np.min(sample['zAngVelocity'])] * userExperimentActivityData.shape[0])
					# 5. MinMax
					xAccMinMaxList = self.appendToList(xAccMinMaxList, [self.calculateMinMax(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					yAccMinMaxList = self.appendToList(yAccMinMaxList, [self.calculateMinMax(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					zAccMinMaxList = self.appendToList(zAccMinMaxList, [self.calculateMinMax(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					xAngVelMinMaxList = self.appendToList(xAngVelMinMaxList, [self.calculateMinMax(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					yAngVelMinMaxList = self.appendToList(yAngVelMinMaxList, [self.calculateMinMax(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					zAngVelMinMaxList = self.appendToList(zAngVelMinMaxList, [self.calculateMinMax(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					# 6. Standard deviation
					xAccStdList = self.appendToList(xAccStdList, [np.std(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					yAccStdList = self.appendToList(yAccStdList, [np.std(sample['yAcceleration'])] * userExperimentActivityData.shape[0])
					zAccStdList = self.appendToList(zAccStdList, [np.std(sample['zAcceleration'])] * userExperimentActivityData.shape[0])
					xAngVelStdList = self.appendToList(xAngVelStdList, [np.std(sample['xAngVelocity'])] * userExperimentActivityData.shape[0])
					yAngVelStdList = self.appendToList(yAngVelStdList, [np.std(sample['yAngVelocity'])] * userExperimentActivityData.shape[0])
					zAngVelStdList = self.appendToList(zAngVelStdList, [np.std(sample['zAngVelocity'])] * userExperimentActivityData.shape[0])
					# 7. First quartile
					xAcc1QList = self.appendToList(xAcc1QList, [self.calculateQuartile(sample['xAcceleration'],25)] * userExperimentActivityData.shape[0])
					yAcc1QList = self.appendToList(yAcc1QList, [self.calculateQuartile(sample['yAcceleration'],25)] * userExperimentActivityData.shape[0])
					zAcc1QList = self.appendToList(zAcc1QList, [self.calculateQuartile(sample['zAcceleration'],25)] * userExperimentActivityData.shape[0])
					xAngVel1QList = self.appendToList(xAngVel1QList, [self.calculateQuartile(sample['xAngVelocity'],25)] * userExperimentActivityData.shape[0])
					yAngVel1QList = self.appendToList(yAngVel1QList, [self.calculateQuartile(sample['yAngVelocity'],25)] * userExperimentActivityData.shape[0])
					zAngVel1QList = self.appendToList(zAngVel1QList, [self.calculateQuartile(sample['zAngVelocity'],25)] * userExperimentActivityData.shape[0])
					# 8. Third quartile
					xAcc3QList = self.appendToList(xAcc3QList, [self.calculateQuartile(sample['xAcceleration'],75)] * userExperimentActivityData.shape[0])
					yAcc3QList = self.appendToList(yAcc3QList, [self.calculateQuartile(sample['yAcceleration'],75)] * userExperimentActivityData.shape[0])
					zAcc3QList = self.appendToList(zAcc3QList, [self.calculateQuartile(sample['zAcceleration'],75)] * userExperimentActivityData.shape[0])
					xAngVel3QList = self.appendToList(xAngVel3QList, [self.calculateQuartile(sample['xAngVelocity'],75)] * userExperimentActivityData.shape[0])
					yAngVel3QList = self.appendToList(yAngVel3QList, [self.calculateQuartile(sample['yAngVelocity'],75)] * userExperimentActivityData.shape[0])
					zAngVel3QList = self.appendToList(zAngVel3QList, [self.calculateQuartile(sample['zAngVelocity'],75)] * userExperimentActivityData.shape[0])
					# 9.1 Average time between maxima peaks
					xAccTBMaxPList = self.appendToList(xAccTBMaxPList, [self.calculateAverageTimeBetweenPeaks(sample['xAcceleration'])[0]] * userExperimentActivityData.shape[0])
					yAccTBMaxPList = self.appendToList(yAccTBMaxPList, [self.calculateAverageTimeBetweenPeaks(sample['yAcceleration'])[0]] * userExperimentActivityData.shape[0])
					zAccTBMaxPList = self.appendToList(zAccTBMaxPList, [self.calculateAverageTimeBetweenPeaks(sample['zAcceleration'])[0]] * userExperimentActivityData.shape[0])
					xAngVelTBMaxPList = self.appendToList(xAngVelTBMaxPList, [self.calculateAverageTimeBetweenPeaks(sample['xAngVelocity'])[0]] * userExperimentActivityData.shape[0])
					yAngVelTBMaxPList = self.appendToList(yAngVelTBMaxPList, [self.calculateAverageTimeBetweenPeaks(sample['yAngVelocity'])[0]] * userExperimentActivityData.shape[0])
					zAngVelTBMaxPList = self.appendToList(zAngVelTBMaxPList, [self.calculateAverageTimeBetweenPeaks(sample['zAngVelocity'])[0]] * userExperimentActivityData.shape[0])
					# 9.2 Average time between minima peaks
					xAccTBMinPList = self.appendToList(xAccTBMinPList, [self.calculateAverageTimeBetweenPeaks(sample['xAcceleration'])[1]] * userExperimentActivityData.shape[0])
					yAccTBMinPList = self.appendToList(yAccTBMinPList, [self.calculateAverageTimeBetweenPeaks(sample['yAcceleration'])[1]] * userExperimentActivityData.shape[0])
					zAccTBMinPList = self.appendToList(zAccTBMinPList, [self.calculateAverageTimeBetweenPeaks(sample['zAcceleration'])[1]] * userExperimentActivityData.shape[0])
					xAngVelTBMinPList = self.appendToList(xAngVelTBMinPList, [self.calculateAverageTimeBetweenPeaks(sample['xAngVelocity'])[1]] * userExperimentActivityData.shape[0])
					yAngVelTBMinPList = self.appendToList(yAngVelTBMinPList, [self.calculateAverageTimeBetweenPeaks(sample['yAngVelocity'])[1]] * userExperimentActivityData.shape[0])
					zAngVelTBMinPList = self.appendToList(zAngVelTBMinPList, [self.calculateAverageTimeBetweenPeaks(sample['zAngVelocity'])[1]] * userExperimentActivityData.shape[0])
					# 10.1 Maxima peak frecuency
					xAccPKMaxList = self.appendToList(xAccPKMaxList, [self.calculatePeakFrecuency(sample['xAcceleration'])[0]] * userExperimentActivityData.shape[0])
					yAccPKMaxList = self.appendToList(yAccPKMaxList, [self.calculatePeakFrecuency(sample['yAcceleration'])[0]] * userExperimentActivityData.shape[0])
					zAccPKMaxList = self.appendToList(zAccPKMaxList, [self.calculatePeakFrecuency(sample['zAcceleration'])[0]] * userExperimentActivityData.shape[0])
					xAngVelPKMaxList = self.appendToList(xAngVelPKMaxList, [self.calculatePeakFrecuency(sample['xAngVelocity'])[0]] * userExperimentActivityData.shape[0])
					yAngVelPKMaxList = self.appendToList(yAngVelPKMaxList, [self.calculatePeakFrecuency(sample['yAngVelocity'])[0]] * userExperimentActivityData.shape[0])
					zAngVelPKMaxList = self.appendToList(zAngVelPKMaxList, [self.calculatePeakFrecuency(sample['zAngVelocity'])[0]] * userExperimentActivityData.shape[0])
					# 10.2 Minima peak frecuency
					xAccPKMinList = self.appendToList(xAccPKMinList, [self.calculatePeakFrecuency(sample['xAcceleration'])[1]] * userExperimentActivityData.shape[0])
					yAccPKMinList = self.appendToList(yAccPKMinList, [self.calculatePeakFrecuency(sample['yAcceleration'])[1]] * userExperimentActivityData.shape[0])
					zAccPKMinList = self.appendToList(zAccPKMinList, [self.calculatePeakFrecuency(sample['zAcceleration'])[1]] * userExperimentActivityData.shape[0])
					xAngVelPKMinList = self.appendToList(xAngVelPKMinList, [self.calculatePeakFrecuency(sample['xAngVelocity'])[1]] * userExperimentActivityData.shape[0])
					yAngVelPKMinList = self.appendToList(yAngVelPKMinList, [self.calculatePeakFrecuency(sample['yAngVelocity'])[1]] * userExperimentActivityData.shape[0])
					zAngVelPKMinList = self.appendToList(zAngVelPKMinList, [self.calculatePeakFrecuency(sample['zAngVelocity'])[1]] * userExperimentActivityData.shape[0])
					# 11. Correlations between signals axes
					corAccXYList = self.appendToList(corAccXYList, [np.corrcoef(sample['xAcceleration'],sample['yAcceleration'])[0,1]] * userExperimentActivityData.shape[0])
					corAccXZList = self.appendToList(corAccXZList, [np.corrcoef(sample['xAcceleration'],sample['zAcceleration'])[0,1]] * userExperimentActivityData.shape[0])
					corAccYZList = self.appendToList(corAccYZList, [np.corrcoef(sample['yAcceleration'],sample['zAcceleration'])[0,1]] * userExperimentActivityData.shape[0])
					corAngVelXYList = self.appendToList(corAngVelXYList, [np.corrcoef(sample['xAngVelocity'],sample['yAngVelocity'])[0,1]] * userExperimentActivityData.shape[0])
					corAngVelXZList = self.appendToList(corAngVelXZList, [np.corrcoef(sample['xAngVelocity'],sample['zAngVelocity'])[0,1]] * userExperimentActivityData.shape[0])
					corAngVelYZList = self.appendToList(corAngVelYZList, [np.corrcoef(sample['yAngVelocity'],sample['zAngVelocity'])[0,1]] * userExperimentActivityData.shape[0])
				
			print "------------"

		finalDataSet = np.column_stack((activitiesList,xAccList,yAccList,zAccList,xAngVelList,yAngVelList,zAngVelList, \
			xAccMeanList,yAccMeanList,zAccMeanList,xAngVelMeanList,yAngVelMeanList,zAngVelMeanList, \
			xAccMedianList,yAccMedianList,zAccMedianList,xAngVelMedianList,yAngVelMedianList,zAngVelMedianList, \
			xAccMaxList,yAccMaxList,zAccMaxList,xAngVelMaxList,yAngVelMaxList,zAngVelMaxList, \
			xAccMinList,yAccMinList,zAccMinList,xAngVelMinList,yAngVelMinList,zAngVelMinList, \
			xAccMinMaxList,yAccMinMaxList,zAccMinMaxList,xAngVelMinMaxList,yAngVelMinMaxList,zAngVelMinMaxList, \
			xAccStdList,yAccStdList,zAccStdList,xAngVelStdList,yAngVelStdList,zAngVelStdList, \
			xAcc1QList,yAcc1QList,zAcc1QList,xAngVel1QList,yAngVel1QList,zAngVel1QList, \
			xAcc3QList,yAcc3QList,zAcc3QList,xAngVel3QList,yAngVel3QList,zAngVel3QList, \
			xAccTBMaxPList,yAccTBMaxPList,zAccTBMaxPList,xAngVelTBMaxPList,yAngVelTBMaxPList,zAngVelTBMaxPList, \
			xAccTBMinPList,yAccTBMinPList,zAccTBMinPList,xAngVelTBMinPList,yAngVelTBMinPList,zAngVelTBMinPList, \
			xAccPKMaxList,yAccPKMaxList,zAccPKMaxList,xAngVelPKMaxList,yAngVelPKMaxList,zAngVelPKMaxList, \
			xAccPKMinList,yAccPKMinList,zAccPKMinList,xAngVelPKMinList,yAngVelPKMinList,zAngVelPKMinList, \
			corAccXYList,corAccXZList,corAccYZList,corAngVelXYList,corAngVelXZList,corAngVelYZList))

		print "New dataset", finalDataSet
		print "Shape", finalDataSet.shape
		print "Row example (first observation)", finalDataSet[1]

		return finalDataSet

	def appendToList(self,list,elements):
		for element in elements:
			list.append(element)
		return list

	def calculateMinMax(self,data):
		minMax = np.max(data) - np.min(data)
		return minMax

	def calculateQuartile(self,data,number):
		percentile = np.percentile(data,number)
		return percentile

	def calculateAverageTimeBetweenPeaks(self,data):
		data = np.asarray(data)
		# Local maxima
		maxPositions = argrelextrema(data, np.greater)
		observationsBetweenMaxs = np.ediff1d(maxPositions)
		meanMax = np.mean(observationsBetweenMaxs) * (1/float(parametersConfig.frecuencyRate))
		# Local minima
		minPositions = argrelextrema(data, np.greater)
		observationsBetweenMins = np.ediff1d(minPositions)
		meanMin = np.mean(observationsBetweenMins) * (1/float(parametersConfig.frecuencyRate))
		return (meanMax,meanMin)

	def calculatePeakFrecuency(self,data):
		# Local maxima
		maxPositions = argrelextrema(data, np.greater)
		maxPeakFrecuency = len(maxPositions[0])
		# Local minima
		minPositions = argrelextrema(data, np.greater)
		minPeakFrecuency = len(minPositions[0])
		return (maxPeakFrecuency,minPeakFrecuency)

#class dimensionalityDimensionalizer():

#class BiometricsSupervisedLearning():

def activityRecognition():
	DI = dataInspector()
	data = DI.dataLoader('data/HAR_UCI_RAW.txt',',',['userID','activity','experimentID','xAcceleration','yAcceleration','zAcceleration','xAngVelocity','yAngVelocity', 'zAngVelocity'], (int,int,int,float,float,float,float,float,float))
	dataFiltered = DI.activitiesSelector(data, [1,2,3,4,5,6])
	DI.dataSummarizer(dataFiltered,['Walking', 'Walking upstairs', 'Walking downstairs', 'Sitting', 'Standing', 'Laying'])

	FG = featureGenerator()
	dataWithAddedFeatures = FG.featuresComputation(dataFiltered, parametersConfig.windowSize)

	headers = ['activity','xAcc','yAcc','zAcc','xAngVel','yAngVel','zAngVel', \
			'xAccMean','yAccMean','zAccMean','xAngVelMean','yAngVelMean','zAngVelMean', \
			'xAccMedian','yAccMedian','zAccMedian','xAngVelMedian','yAngVelMedian','zAngVelMedian', \
			'xAccMax','yAccMax','zAccMax','xAngVelMax','yAngVelMax','zAngVelMax', \
			'xAccMin','yAccMin','zAccMin','xAngVelMin','yAngVelMin','zAngVelMin', \
			'xAccMinMax','yAccMinMax','zAccMinMax','xAngVelMinMax','yAngVelMinMax','zAngVelMinMax', \
			'xAccStd','yAccStd','zAccStd','xAngVelStd','yAngVelStd','zAngVelStd', \
			'xAcc1Q','yAcc1Q','zAcc1Q','xAngVel1Q','yAngVel1Q','zAngVel1Q', \
			'xAcc3Q','yAcc3Q','zAcc3Q','xAngVel3Q','yAngVel3Q','zAngVel3Q', \
			'xAccTBMaxP','yAccTBMaxP','zAccTBMaxP','xAngVelTBMaxP','yAngVelTBMaxP','zAngVelTBMaxP', \
			'xAccTBMinP','yAccTBMinP','zAccTBMinP','xAngVelTBMinP','yAngVelTBMinP','zAngVelTBMinP', \
			'xAccPKMax','yAccPKMax','zAccPKMax','xAngVelPKMax','yAngVelPKMax','zAngVelPKMax', \
			'xAccPKMin','yAccPKMin','zAccPKMin','xAngVelPKMin','yAngVelPKMin','zAngVelPKMin', \
			'corAccXY','corAccXZ','corAccYZ','corAngVelXY','corAngVelXZ','corAngVelYZ']

	formats = ['int'] + (['float'] * (dataWithAddedFeatures.shape[1] - 1))

	dt = {'names':headers, 'formats': formats}
	dataWithAddedFeaturesAndHeaders = np.zeros(dataWithAddedFeatures.shape[0], dtype=dt)

	for columnNumber in range(0,dataWithAddedFeatures.shape[1]):
		print headers[columnNumber]
		print dataWithAddedFeatures[:,columnNumber]
		dataWithAddedFeaturesAndHeaders[headers[columnNumber]] = dataWithAddedFeatures[:,columnNumber]

	# Saving memory
	data = dataWithAddedFeaturesAndHeaders
	del dataFiltered, dataWithAddedFeatures, dataWithAddedFeaturesAndHeaders
	
	# Boxplots
	DI.boxplotPrinter(data,headers[7:len(headers)-1])
	
	
if __name__ == '__main__':
	activityRecognition()