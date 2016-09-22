#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt

class dataAnalysis():

	def dataLoader(self, inputFile, delimiter, headers, dataTypes):
		""" Loads data from file 

		Keywords arguments:
		inputFile -- file name
		delimiter -- delimiter that separates values from same row
		headers -- list with columns headers
		dataTypes -- list with data type for each colum
		"""
		data = np.genfromtxt(inputFile, delimiter=delimiter, names=headers, dtype=dataTypes)
		return data

	def activitiesSelector(self, data, activities):
		""" Selects data linked certain activities

		Keywords arguments:
		data -- entire dataset
		activities -- list with activities keys
		"""
		# The command is composed activity by activity
		activityInstruction = "data[np.where("
		counter = 0
		for activity in activities:
			counter += 1
			activityInstruction += "(data['activity']==" + str(activity)
			# If there are no more activities, close command
			if counter == len(activities):
				activityInstruction += "))]"
			# If there are more activies, add an 'or' clause
			else:
				activityInstruction += ") | "

		dataChosenActivities = eval(activityInstruction)
		return dataChosenActivities

	def dataSummarizer(self,data,activitiesNames):
		""" Indicates which percentage of data is about a certain movement 

		Keywords arguments:
		data -- entire dataset
		activitiesNames -- list with activities names
		"""
		counter = 0
		for activity in range(1,7):
			selectedRows = data[np.where(data['activity'] == activity)]
			classPercentage = round(float(selectedRows.shape[0])/float(int(data.shape[0])),2) * 100
			print '-', activitiesNames[counter], ':', classPercentage, '%'
			counter += 1
		return None

	def boxplotPrinter(self, data, features):
		""" Plot features boxplots charts 

		Keywords arguments:
		data -- dataset
		features -- list with variables to plot
		"""
		# List with differents activities
		activities = np.unique(data['activity'])
		# For each activity...
		for activity in activities:
			print "- Plotting features linked to activity", activity, '...'
			# ... select data linked with that activity
			activityData = data[np.where(data['activity'] == activity)]
			# For each feature...
			for feature in features:
				# ... select data linked with that feature 
				featureValues = activityData[feature]
				# Plotting figure
				figure = plt.figure(1, figsize=(9, 6))
				ax = figure.add_subplot(111)
				bp = ax.boxplot(featureValues, patch_artist=True)
				# Change outline color, fill color and linewidth of the boxes
				for box in bp['boxes']:
					# Change outline color
					box.set( color='#7570b3', linewidth=2)
					# Change fill color
					box.set( facecolor = '#1b9e77' )
				# Change color and linewidth of the whiskers
				for whisker in bp['whiskers']:
					whisker.set(color='#7570b3', linewidth=2)
				# Change color and linewidth of the caps
				for cap in bp['caps']:
					cap.set(color='#7570b3', linewidth=2)
				# Change color and linewidth of the medians
				for median in bp['medians']:
					median.set(color='#b2df8a', linewidth=2)
				# Change the style of fliers and their fill
				for flier in bp['fliers']:
					flier.set(marker='o', color='#e7298a', alpha=0.5)
				# Custom x-axis labels
				ax.set_xticklabels([feature])
				# Remove top axes and right axes ticks
				ax.get_yaxis().tick_left()

				# If path doesn't exist, let's create it
				directory = 'img/boxplots/activity_'+str(activity)
				if not os.path.exists(directory):
					os.makedirs(directory)
				# Saving plot
				figure.savefig(directory+'/feature_'+feature,bbox_inches='tight')
				figure.clf()
		return None