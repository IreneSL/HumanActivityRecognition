#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import parametersConfig
from scipy.signal import argrelextrema

class featuresAgent():

	def featuresComputation(self,data,windowSize):
		''' Generate new features from original data '''
		# Existing features
		activitiesList, xAccList, yAccList, zAccList, xAngVelList, yAngVelList, zAngVelList = [], [], [], [], [], [], []
		# New features
		# 1. Means
		xAccMeanList, yAccMeanList, zAccMeanList, xAngVelMeanList, yAngVelMeanList, zAngVelMeanList = [], [], [], [], [], []
		# 2. Medians
		xAccMedianList, yAccMedianList, zAccMedianList, xAngVelMedianList, yAngVelMedianList, zAngVelMedianList = [], [], [], [], [], []
		# 3. MinMax (difference between maximum and minimum)
		xAccMinMaxList, yAccMinMaxList, zAccMinMaxList, xAngVelMinMaxList, yAngVelMinMaxList, zAngVelMinMaxList = [], [], [], [], [], []
		# 4. Standard deviation
		xAccStdList, yAccStdList, zAccStdList, xAngVelStdList, yAngVelStdList, zAngVelStdList = [], [], [], [], [], []
		# 5. First quartile
		xAcc1QList, yAcc1QList, zAcc1QList, xAngVel1QList, yAngVel1QList, zAngVel1QList = [], [], [], [], [], []
		# 6. Third quartile
		xAcc3QList, yAcc3QList, zAcc3QList, xAngVel3QList, yAngVel3QList, zAngVel3QList = [], [], [], [], [], []
		# 7. Interquartile range (Q3 - Q1) (It is not affected by outliers)
		xAccIQRList, yAccIQRList, zAccIQRList, xAngVelIQRList, yAngVelIQRList, zAngVelIQRList = [], [], [], [], [], []
		# 8. Average time between peaks 
		xAccTBMaxPList, yAccTBMaxPList, zAccTBMaxPList, xAngVelTBMaxPList, yAngVelTBMaxPList, zAngVelTBMaxPList = [], [], [], [], [], []
		xAccTBMinPList, yAccTBMinPList, zAccTBMinPList, xAngVelTBMinPList, yAngVelTBMinPList, zAngVelTBMinPList = [], [], [], [], [], []
		# 9. Peak frecuency
		xAccPKMaxList, yAccPKMaxList, zAccPKMaxList, xAngVelPKMaxList, yAngVelPKMaxList, zAngVelPKMaxList = [], [], [], [], [], []
		xAccPKMinList, yAccPKMinList, zAccPKMinList, xAngVelPKMinList, yAngVelPKMinList, zAngVelPKMinList = [], [], [], [], [], []
		# 10. Positives and negatives peaks
		xAccMaxPosPeaksList, yAccMaxPosPeaksList, zAccMaxPosPeaksList, xAngVelMaxPosPeaksList, yAngVelMaxPosPeaksList, zAngVelMaxPosPeaksList = [], [], [], [], [], []
		xAccMaxNegPeaksList, yAccMaxNegPeaksList, zAccMaxNegPeaksList, xAngVelMaxNegPeaksList, yAngVelMaxNegPeaksList, zAngVelMaxNegPeaksList = [], [], [], [], [], []
		xAccMinPosPeaksList, yAccMinPosPeaksList, zAccMinPosPeaksList, xAngVelMinPosPeaksList, yAngVelMinPosPeaksList, zAngVelMinPosPeaksList = [], [], [], [], [], []
		xAccMinNegPeaksList, yAccMinNegPeaksList, zAccMinNegPeaksList, xAngVelMinNegPeaksList, yAngVelMinNegPeaksList, zAngVelMinNegPeaksList = [], [], [], [], [], []
		# 11. Zero crossings
		xAccZCList, yAccZCList, zAccZCList, xAngVelZCList, yAngVelZCList, zAngVelZCList = [], [], [], [], [], []
		# 12. Correlations between signals axes
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
					# 3. MinMax
					xAccMinMaxList = self.appendToList(xAccMinMaxList, [self.calculateMinMax(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					yAccMinMaxList = self.appendToList(yAccMinMaxList, [self.calculateMinMax(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					zAccMinMaxList = self.appendToList(zAccMinMaxList, [self.calculateMinMax(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					xAngVelMinMaxList = self.appendToList(xAngVelMinMaxList, [self.calculateMinMax(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					yAngVelMinMaxList = self.appendToList(yAngVelMinMaxList, [self.calculateMinMax(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					zAngVelMinMaxList = self.appendToList(zAngVelMinMaxList, [self.calculateMinMax(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					# 4. Standard deviation
					xAccStdList = self.appendToList(xAccStdList, [np.std(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					yAccStdList = self.appendToList(yAccStdList, [np.std(sample['yAcceleration'])] * userExperimentActivityData.shape[0])
					zAccStdList = self.appendToList(zAccStdList, [np.std(sample['zAcceleration'])] * userExperimentActivityData.shape[0])
					xAngVelStdList = self.appendToList(xAngVelStdList, [np.std(sample['xAngVelocity'])] * userExperimentActivityData.shape[0])
					yAngVelStdList = self.appendToList(yAngVelStdList, [np.std(sample['yAngVelocity'])] * userExperimentActivityData.shape[0])
					zAngVelStdList = self.appendToList(zAngVelStdList, [np.std(sample['zAngVelocity'])] * userExperimentActivityData.shape[0])
					# 5. First quartile
					xAcc1QList = self.appendToList(xAcc1QList, [self.calculateQuartile(sample['xAcceleration'],25)] * userExperimentActivityData.shape[0])
					yAcc1QList = self.appendToList(yAcc1QList, [self.calculateQuartile(sample['yAcceleration'],25)] * userExperimentActivityData.shape[0])
					zAcc1QList = self.appendToList(zAcc1QList, [self.calculateQuartile(sample['zAcceleration'],25)] * userExperimentActivityData.shape[0])
					xAngVel1QList = self.appendToList(xAngVel1QList, [self.calculateQuartile(sample['xAngVelocity'],25)] * userExperimentActivityData.shape[0])
					yAngVel1QList = self.appendToList(yAngVel1QList, [self.calculateQuartile(sample['yAngVelocity'],25)] * userExperimentActivityData.shape[0])
					zAngVel1QList = self.appendToList(zAngVel1QList, [self.calculateQuartile(sample['zAngVelocity'],25)] * userExperimentActivityData.shape[0])
					# 6. Third quartile
					xAcc3QList = self.appendToList(xAcc3QList, [self.calculateQuartile(sample['xAcceleration'],75)] * userExperimentActivityData.shape[0])
					yAcc3QList = self.appendToList(yAcc3QList, [self.calculateQuartile(sample['yAcceleration'],75)] * userExperimentActivityData.shape[0])
					zAcc3QList = self.appendToList(zAcc3QList, [self.calculateQuartile(sample['zAcceleration'],75)] * userExperimentActivityData.shape[0])
					xAngVel3QList = self.appendToList(xAngVel3QList, [self.calculateQuartile(sample['xAngVelocity'],75)] * userExperimentActivityData.shape[0])
					yAngVel3QList = self.appendToList(yAngVel3QList, [self.calculateQuartile(sample['yAngVelocity'],75)] * userExperimentActivityData.shape[0])
					zAngVel3QList = self.appendToList(zAngVel3QList, [self.calculateQuartile(sample['zAngVelocity'],75)] * userExperimentActivityData.shape[0])
					# 7. Interquartile range
					xAccIQRList = self.appendToList(xAccIQRList, [self.calculateInterquartileRange(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					yAccIQRList = self.appendToList(yAccIQRList, [self.calculateInterquartileRange(sample['yAcceleration'])] * userExperimentActivityData.shape[0])
					zAccIQRList = self.appendToList(zAccIQRList, [self.calculateInterquartileRange(sample['zAcceleration'])] * userExperimentActivityData.shape[0])
					xAngVelIQRList = self.appendToList(xAngVelIQRList, [self.calculateInterquartileRange(sample['xAngVelocity'])] * userExperimentActivityData.shape[0])
					yAngVelIQRList = self.appendToList(yAngVelIQRList, [self.calculateInterquartileRange(sample['yAngVelocity'])] * userExperimentActivityData.shape[0])
					zAngVelIQRList = self.appendToList(zAngVelIQRList, [self.calculateInterquartileRange(sample['zAngVelocity'])] * userExperimentActivityData.shape[0])
					# 8.1 Average time between maxima peaks
					xAccTBMaxPList = self.appendToList(xAccTBMaxPList, [self.calculateAverageTimeBetweenPeaks(sample['xAcceleration'])[0]] * userExperimentActivityData.shape[0])
					yAccTBMaxPList = self.appendToList(yAccTBMaxPList, [self.calculateAverageTimeBetweenPeaks(sample['yAcceleration'])[0]] * userExperimentActivityData.shape[0])
					zAccTBMaxPList = self.appendToList(zAccTBMaxPList, [self.calculateAverageTimeBetweenPeaks(sample['zAcceleration'])[0]] * userExperimentActivityData.shape[0])
					xAngVelTBMaxPList = self.appendToList(xAngVelTBMaxPList, [self.calculateAverageTimeBetweenPeaks(sample['xAngVelocity'])[0]] * userExperimentActivityData.shape[0])
					yAngVelTBMaxPList = self.appendToList(yAngVelTBMaxPList, [self.calculateAverageTimeBetweenPeaks(sample['yAngVelocity'])[0]] * userExperimentActivityData.shape[0])
					zAngVelTBMaxPList = self.appendToList(zAngVelTBMaxPList, [self.calculateAverageTimeBetweenPeaks(sample['zAngVelocity'])[0]] * userExperimentActivityData.shape[0])
					# 8.2 Average time between minima peaks
					xAccTBMinPList = self.appendToList(xAccTBMinPList, [self.calculateAverageTimeBetweenPeaks(sample['xAcceleration'])[1]] * userExperimentActivityData.shape[0])
					yAccTBMinPList = self.appendToList(yAccTBMinPList, [self.calculateAverageTimeBetweenPeaks(sample['yAcceleration'])[1]] * userExperimentActivityData.shape[0])
					zAccTBMinPList = self.appendToList(zAccTBMinPList, [self.calculateAverageTimeBetweenPeaks(sample['zAcceleration'])[1]] * userExperimentActivityData.shape[0])
					xAngVelTBMinPList = self.appendToList(xAngVelTBMinPList, [self.calculateAverageTimeBetweenPeaks(sample['xAngVelocity'])[1]] * userExperimentActivityData.shape[0])
					yAngVelTBMinPList = self.appendToList(yAngVelTBMinPList, [self.calculateAverageTimeBetweenPeaks(sample['yAngVelocity'])[1]] * userExperimentActivityData.shape[0])
					zAngVelTBMinPList = self.appendToList(zAngVelTBMinPList, [self.calculateAverageTimeBetweenPeaks(sample['zAngVelocity'])[1]] * userExperimentActivityData.shape[0])
					# 9.1 Maxima peak frecuency
					xAccPKMaxList = self.appendToList(xAccPKMaxList, [self.calculatePeakFrecuency(sample['xAcceleration'])[0]] * userExperimentActivityData.shape[0])
					yAccPKMaxList = self.appendToList(yAccPKMaxList, [self.calculatePeakFrecuency(sample['yAcceleration'])[0]] * userExperimentActivityData.shape[0])
					zAccPKMaxList = self.appendToList(zAccPKMaxList, [self.calculatePeakFrecuency(sample['zAcceleration'])[0]] * userExperimentActivityData.shape[0])
					xAngVelPKMaxList = self.appendToList(xAngVelPKMaxList, [self.calculatePeakFrecuency(sample['xAngVelocity'])[0]] * userExperimentActivityData.shape[0])
					yAngVelPKMaxList = self.appendToList(yAngVelPKMaxList, [self.calculatePeakFrecuency(sample['yAngVelocity'])[0]] * userExperimentActivityData.shape[0])
					zAngVelPKMaxList = self.appendToList(zAngVelPKMaxList, [self.calculatePeakFrecuency(sample['zAngVelocity'])[0]] * userExperimentActivityData.shape[0])
					# 9.2 Minima peak frecuency
					xAccPKMinList = self.appendToList(xAccPKMinList, [self.calculatePeakFrecuency(sample['xAcceleration'])[1]] * userExperimentActivityData.shape[0])
					yAccPKMinList = self.appendToList(yAccPKMinList, [self.calculatePeakFrecuency(sample['yAcceleration'])[1]] * userExperimentActivityData.shape[0])
					zAccPKMinList = self.appendToList(zAccPKMinList, [self.calculatePeakFrecuency(sample['zAcceleration'])[1]] * userExperimentActivityData.shape[0])
					xAngVelPKMinList = self.appendToList(xAngVelPKMinList, [self.calculatePeakFrecuency(sample['xAngVelocity'])[1]] * userExperimentActivityData.shape[0])
					yAngVelPKMinList = self.appendToList(yAngVelPKMinList, [self.calculatePeakFrecuency(sample['yAngVelocity'])[1]] * userExperimentActivityData.shape[0])
					zAngVelPKMinList = self.appendToList(zAngVelPKMinList, [self.calculatePeakFrecuency(sample['zAngVelocity'])[1]] * userExperimentActivityData.shape[0])
					# 10.1 Positives max peaks
					xAccMaxPosPeaksList = self.appendToList(xAccMaxPosPeaksList, [self.calculatePeaksGroups(sample['xAcceleration'])[0]] * userExperimentActivityData.shape[0])
					yAccMaxPosPeaksList = self.appendToList(yAccMaxPosPeaksList, [self.calculatePeaksGroups(sample['yAcceleration'])[0]] * userExperimentActivityData.shape[0])
					zAccMaxPosPeaksList = self.appendToList(zAccMaxPosPeaksList, [self.calculatePeaksGroups(sample['zAcceleration'])[0]] * userExperimentActivityData.shape[0])
					xAngVelMaxPosPeaksList = self.appendToList(xAngVelMaxPosPeaksList, [self.calculatePeaksGroups(sample['xAngVelocity'])[0]] * userExperimentActivityData.shape[0])
					yAngVelMaxPosPeaksList = self.appendToList(yAngVelMaxPosPeaksList, [self.calculatePeaksGroups(sample['yAngVelocity'])[0]] * userExperimentActivityData.shape[0])
					zAngVelMaxPosPeaksList = self.appendToList(zAngVelMaxPosPeaksList, [self.calculatePeaksGroups(sample['zAngVelocity'])[0]] * userExperimentActivityData.shape[0])
					# 10.2 Negatives max peaks
					xAccMaxNegPeaksList = self.appendToList(xAccMaxNegPeaksList, [self.calculatePeaksGroups(sample['xAcceleration'])[1]] * userExperimentActivityData.shape[0])
					yAccMaxNegPeaksList = self.appendToList(yAccMaxNegPeaksList, [self.calculatePeaksGroups(sample['yAcceleration'])[1]] * userExperimentActivityData.shape[0])
					zAccMaxNegPeaksList = self.appendToList(zAccMaxNegPeaksList, [self.calculatePeaksGroups(sample['zAcceleration'])[1]] * userExperimentActivityData.shape[0])
					xAngVelMaxNegPeaksList = self.appendToList(xAngVelMaxNegPeaksList, [self.calculatePeaksGroups(sample['xAngVelocity'])[1]] * userExperimentActivityData.shape[0])
					yAngVelMaxNegPeaksList = self.appendToList(yAngVelMaxNegPeaksList, [self.calculatePeaksGroups(sample['yAngVelocity'])[1]] * userExperimentActivityData.shape[0])
					zAngVelMaxNegPeaksList = self.appendToList(zAngVelMaxNegPeaksList, [self.calculatePeaksGroups(sample['zAngVelocity'])[1]] * userExperimentActivityData.shape[0])
					# 10.3 Positives min peaks
					xAccMinPosPeaksList = self.appendToList(xAccMinPosPeaksList, [self.calculatePeaksGroups(sample['xAcceleration'])[2]] * userExperimentActivityData.shape[0])
					yAccMinPosPeaksList = self.appendToList(yAccMinPosPeaksList, [self.calculatePeaksGroups(sample['yAcceleration'])[2]] * userExperimentActivityData.shape[0])
					zAccMinPosPeaksList = self.appendToList(zAccMinPosPeaksList, [self.calculatePeaksGroups(sample['zAcceleration'])[2]] * userExperimentActivityData.shape[0])
					xAngVelMinPosPeaksList = self.appendToList(xAngVelMinPosPeaksList, [self.calculatePeaksGroups(sample['xAngVelocity'])[2]] * userExperimentActivityData.shape[0])
					yAngVelMinPosPeaksList = self.appendToList(yAngVelMinPosPeaksList, [self.calculatePeaksGroups(sample['yAngVelocity'])[2]] * userExperimentActivityData.shape[0])
					zAngVelMinPosPeaksList = self.appendToList(zAngVelMinPosPeaksList, [self.calculatePeaksGroups(sample['zAngVelocity'])[2]] * userExperimentActivityData.shape[0])
					# 10.4 Negative min peaks
					xAccMinNegPeaksList = self.appendToList(xAccMinNegPeaksList, [self.calculatePeaksGroups(sample['xAcceleration'])[3]] * userExperimentActivityData.shape[0])
					yAccMinNegPeaksList = self.appendToList(yAccMinNegPeaksList, [self.calculatePeaksGroups(sample['yAcceleration'])[3]] * userExperimentActivityData.shape[0])
					zAccMinNegPeaksList = self.appendToList(zAccMinNegPeaksList, [self.calculatePeaksGroups(sample['zAcceleration'])[3]] * userExperimentActivityData.shape[0])
					xAngVelMinNegPeaksList = self.appendToList(xAngVelMinNegPeaksList, [self.calculatePeaksGroups(sample['xAngVelocity'])[3]] * userExperimentActivityData.shape[0])
					yAngVelMinNegPeaksList = self.appendToList(yAngVelMinNegPeaksList, [self.calculatePeaksGroups(sample['yAngVelocity'])[3]] * userExperimentActivityData.shape[0])
					zAngVelMinNegPeaksList = self.appendToList(zAngVelMinNegPeaksList, [self.calculatePeaksGroups(sample['zAngVelocity'])[3]] * userExperimentActivityData.shape[0])
					# 11. Zero crossings
					xAccZCList = self.appendToList(xAccZCList, [self.zeroCrossings(sample['xAcceleration'])] * userExperimentActivityData.shape[0])
					yAccZCList = self.appendToList(yAccZCList, [self.zeroCrossings(sample['yAcceleration'])] * userExperimentActivityData.shape[0])
					zAccZCList = self.appendToList(zAccZCList, [self.zeroCrossings(sample['zAcceleration'])] * userExperimentActivityData.shape[0])
					xAngVelZCList = self.appendToList(xAngVelZCList, [self.zeroCrossings(sample['xAngVelocity'])] * userExperimentActivityData.shape[0])
					yAngVelZCList = self.appendToList(yAngVelZCList, [self.zeroCrossings(sample['yAngVelocity'])] * userExperimentActivityData.shape[0])
					zAngVelZCList =  self.appendToList(zAngVelZCList, [self.zeroCrossings(sample['zAngVelocity'])] * userExperimentActivityData.shape[0])
					# 12. Correlations between signals axes
					corAccXYList = self.appendToList(corAccXYList, [np.corrcoef(sample['xAcceleration'],sample['yAcceleration'])[0,1]] * userExperimentActivityData.shape[0])
					corAccXZList = self.appendToList(corAccXZList, [np.corrcoef(sample['xAcceleration'],sample['zAcceleration'])[0,1]] * userExperimentActivityData.shape[0])
					corAccYZList = self.appendToList(corAccYZList, [np.corrcoef(sample['yAcceleration'],sample['zAcceleration'])[0,1]] * userExperimentActivityData.shape[0])
					corAngVelXYList = self.appendToList(corAngVelXYList, [np.corrcoef(sample['xAngVelocity'],sample['yAngVelocity'])[0,1]] * userExperimentActivityData.shape[0])
					corAngVelXZList = self.appendToList(corAngVelXZList, [np.corrcoef(sample['xAngVelocity'],sample['zAngVelocity'])[0,1]] * userExperimentActivityData.shape[0])
					corAngVelYZList = self.appendToList(corAngVelYZList, [np.corrcoef(sample['yAngVelocity'],sample['zAngVelocity'])[0,1]] * userExperimentActivityData.shape[0])
				
			print "------------"

		finalDataSet = np.column_stack((activitiesList, \
			xAccMeanList,yAccMeanList,zAccMeanList,xAngVelMeanList,yAngVelMeanList,zAngVelMeanList, \
			xAccMedianList,yAccMedianList,zAccMedianList,xAngVelMedianList,yAngVelMedianList,zAngVelMedianList, \
			xAccMinMaxList,yAccMinMaxList,zAccMinMaxList,xAngVelMinMaxList,yAngVelMinMaxList,zAngVelMinMaxList, \
			xAccStdList,yAccStdList,zAccStdList,xAngVelStdList,yAngVelStdList,zAngVelStdList, \
			xAcc1QList,yAcc1QList,zAcc1QList,xAngVel1QList,yAngVel1QList,zAngVel1QList, \
			xAcc3QList,yAcc3QList,zAcc3QList,xAngVel3QList,yAngVel3QList,zAngVel3QList, \
			xAccIQRList, yAccIQRList, zAccIQRList, xAngVelIQRList, yAngVelIQRList, zAngVelIQRList, \
			xAccTBMaxPList,yAccTBMaxPList,zAccTBMaxPList,xAngVelTBMaxPList,yAngVelTBMaxPList,zAngVelTBMaxPList, \
			xAccTBMinPList,yAccTBMinPList,zAccTBMinPList,xAngVelTBMinPList,yAngVelTBMinPList,zAngVelTBMinPList, \
			xAccPKMaxList,yAccPKMaxList,zAccPKMaxList,xAngVelPKMaxList,yAngVelPKMaxList,zAngVelPKMaxList, \
			xAccPKMinList,yAccPKMinList,zAccPKMinList,xAngVelPKMinList,yAngVelPKMinList,zAngVelPKMinList, \
			xAccMaxPosPeaksList, yAccMaxPosPeaksList, zAccMaxPosPeaksList, xAngVelMaxPosPeaksList, yAngVelMaxPosPeaksList, zAngVelMaxPosPeaksList, \
			xAccMaxNegPeaksList, yAccMaxNegPeaksList, zAccMaxNegPeaksList, xAngVelMaxNegPeaksList, yAngVelMaxNegPeaksList, zAngVelMaxNegPeaksList, \
			xAccMinPosPeaksList, yAccMinPosPeaksList, zAccMinPosPeaksList, xAngVelMinPosPeaksList, yAngVelMinPosPeaksList, zAngVelMinPosPeaksList, \
			xAccMinNegPeaksList, yAccMinNegPeaksList, zAccMinNegPeaksList, xAngVelMinNegPeaksList, yAngVelMinNegPeaksList, zAngVelMinNegPeaksList, \
			xAccZCList, yAccZCList, zAccZCList, xAngVelZCList, yAngVelZCList, zAngVelZCList, \
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

	def calculateInterquartileRange(self,data):
		firstQuartile = np.percentile(data,25)
		thirdQuartile = np.percentile(data,75)
		IQR = thirdQuartile - firstQuartile
		return IQR

	def calculateAverageTimeBetweenPeaks(self,data):
		data = np.asarray(data)
		# Local maxima
		maxPositions = argrelextrema(data, np.greater)
		observationsBetweenMaxs = np.ediff1d(maxPositions)
		meanMax = np.mean(observationsBetweenMaxs) * (1/float(parametersConfig.frecuencyRate))
		# Local minima
		minPositions = argrelextrema(data, np.less)
		observationsBetweenMins = np.ediff1d(minPositions)
		meanMin = np.mean(observationsBetweenMins) * (1/float(parametersConfig.frecuencyRate))
		return (meanMax,meanMin)

	def calculatePeakFrecuency(self,data):
		# Local maxima
		maxPositions = argrelextrema(data, np.greater)
		maxPeakFrecuency = len(maxPositions[0])
		# Local minima
		minPositions = argrelextrema(data, np.less)
		minPeakFrecuency = len(minPositions[0])
		return (maxPeakFrecuency,minPeakFrecuency)

	def calculatePeaksGroups(self,data):
		# Local maxima
		maxPositions = argrelextrema(data, np.greater)[0]	
		maxPeaks = data[maxPositions]
		positiveMaxPeaks = maxPeaks[(maxPeaks>0)].size
		negativeMaxPeaks = maxPeaks[(maxPeaks<0)].size
		# Local minima
		minPositions = argrelextrema(data, np.less)
		minPeaks = data[minPositions]
		positiveMinPeaks = minPeaks[(minPeaks>0)].size
		negativeMinPeaks = minPeaks[(minPeaks<0)].size
		return (positiveMaxPeaks,negativeMaxPeaks,positiveMinPeaks,negativeMinPeaks)

	def zeroCrossings(self,data):
		zero_crossings = len(np.where(np.diff(np.sign(data)))[0])
		return zero_crossings