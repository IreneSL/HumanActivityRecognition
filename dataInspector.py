#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt

class dataAnalysis():

	def dataLoader(self, inputFile, delimiter, headers, dataTypes):
		''' Loads a file as data '''
		data = np.genfromtxt(inputFile, delimiter=delimiter, names=headers, dtype=dataTypes)
		return data

	def variablesSelector(self, data, variablesToChoose):
		dataChosenVariables = data[variablesToChoose]
		return dataChosenVariables

	def activitiesSelector(self, data, activities):
		''' Select data linked certain activities '''
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
		selectedRows = data[np.where(data[:,0] == column)]
		print "Initial data length"
		print data.shape[0]
		print "Selected data length"
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
		''' Plot features boxplots charts '''
		activities = np.unique(data['activity'])
		for activity in activities:
			activityData = data[np.where(data['activity'] == activity)]
			for feature in features:
				featureValues = activityData[feature]
				figure = plt.figure(1, figsize=(9, 6))
				ax = figure.add_subplot(111)
				bp = ax.boxplot(featureValues, patch_artist=True)
				# change outline color, fill color and linewidth of the boxes
				for box in bp['boxes']:
					# change outline color
					box.set( color='#7570b3', linewidth=2)
					# change fill color
					box.set( facecolor = '#1b9e77' )
				# change color and linewidth of the whiskers
				for whisker in bp['whiskers']:
					whisker.set(color='#7570b3', linewidth=2)
				# change color and linewidth of the caps
				for cap in bp['caps']:
					cap.set(color='#7570b3', linewidth=2)
				# change color and linewidth of the medians
				for median in bp['medians']:
					median.set(color='#b2df8a', linewidth=2)
				# change the style of fliers and their fill
				for flier in bp['fliers']:
					flier.set(marker='o', color='#e7298a', alpha=0.5)
				# Custom x-axis labels
				ax.set_xticklabels([feature])
				# Remove top axes and right axes ticks
				ax.get_yaxis().tick_left()

				directory = 'img/'+'activity_'+str(activity)
				if not os.path.exists(directory):
					os.makedirs(directory)
				figure.savefig(directory+'/feature_'+feature,bbox_inches='tight')
				figure.clf()