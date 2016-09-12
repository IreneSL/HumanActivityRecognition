#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

class DataAnalyzer():

	def dataLoader(self,inputFile,delimiter):
		''' Loads a file as data '''
		data = np.genfromtxt(inputFile, delimiter=delimiter)
		return data

	def dataSummarizer(self,data,name,column):
		''' Indicates which percentage of data is about a certain movement '''
		selectedRows = data[np.where(data[:,1] == column)]
		classPercentage = round(float(selectedRows.shape[0])/float(int(data.shape[0])),2) * 100
		print '-', name, ':', classPercentage, '%'

	def boxPlotPrinter(self,data,columns,groupByClass):
		''' Prints boxplots about certain dependent variables and movements chosen '''
		for variable in columns:
			if groupByClass:
				for movement in range(1,13):
					dataMovement = data[np.where(data[:,1] == movement)]
					variableValues = dataMovement[:,variable-1]
					figure = plt.boxplot(variableValues)
					plt.savefig('img/boxplot_variable-'+str(variable)+'_movement-'+str(movement)+'.png')
					plt.close()
			else:
				variableValues = data[:,variable-1]
				figure = plt.boxplot(variableValues)
				plt.savefig('img/boxplot-variable-'+str(variable)+'.png')
				plt.close()
			

	def checkCollinearity(self,data):
		return None

	def dataClassSelector(self,data,column):
		''' Pick data linked to a body movement '''
		selectedRows = data[np.where(data[:,1] == column)]
		print "Longitud datos iniciales"
		print data.shape[0]
		print "Longitud datos seleccionados"
		print selectedRows.shape[0]
		return selectedRows

	def featuresChooser(self,data,columns):
		''' Choose dependent variables '''
		trainingData = data[:,columns]
		return trainingData

	def independentVariableChooser(self, data, column):
		''' Choose indepenent variable '''
		targetData = data[:,column]
		return targetData

	def trainingAndTestDataChooser(self,data,trainingRatio):
		''' Split data into train and test '''
		trainSize = int(len(data) * trainingRatio)
		print "Size train", trainSize
		trainSet = data[:trainSize,:,]
		testSet = data[trainSize:data.shape[0]:,]
		return [trainSet, testSet]

class BiometricsSupervisedLearning():
		 
	def naiveBayes(self,trainingFeatures,trainingTarget,testingFeatures,testingTarget):
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

def main():
	DA = DataAnalyzer()
	# Loading data
	data = DA.dataLoader('data/HAR_UCI_RAW.txt',',')

	# ---------- 1. Dataset info -------------------------------------
	# 1.1 Percentage of each type of activity
	'''DA.dataSummarizer(data,'Walking', 1)
	DA.dataSummarizer(data,'Walking upstairs', 2)
	DA.dataSummarizer(data,'Walking downstairs', 3)
	DA.dataSummarizer(data,'Sitting', 4)
	DA.dataSummarizer(data,'Standing', 5)
	DA.dataSummarizer(data,'Laying', 6)
	DA.dataSummarizer(data,'Stand to sit', 7)
	DA.dataSummarizer(data,'Sit to stand', 8)
	DA.dataSummarizer(data,'Sit to lie', 9)
	DA.dataSummarizer(data,'Lie to stand', 10)
	DA.dataSummarizer(data,'Stand to lie', 11)
	DA.dataSummarizer(data,'Lie to stand', 12)'''

	# 1.2 Analyzing outliers
	#DA.boxPlotPrinter(data,[4,5,6,7,8,9],False)	# We see a lot of outliers (at every selected variable). 
													# Perhaps it's more appropiate look at outliers grouping by kind of body movement.

	#DA.boxPlotPrinter(data,[4,5,6,7,8,9],True) 	# Although we see many outliers in almost variables, it's possible to appreciate 
													# there are less variability at variables linked to acceleration (4,5,6) in certain 
													# calm movements (9,10,11,12)

													# I think in this case, because motion oscillations (however great) could be decisive, 
													# the correct choice is not to remove outliers. 

	# 1.3 Checking collinearity between linear acceleration and angular velocity
	# I think this doesn't have any sense. Linear acceleartion is about the 
	# speed variation in time, while angular velocity is about the change of
	# angle in time.


	# ---------------- 2. Choosing data ------------------------------
	# 2.1 Stratified sampling (Training: 0.8; Testing: 0.2)
	trainSet = np.empty((0,data.shape[1]))
	testSet = np.empty((0,data.shape[1]))
	for i in range(1,13):
		dataChoosed = DA.trainingAndTestDataChooser(DA.dataClassSelector(data, i), 0.8)
		trainSet = np.append(trainSet, dataChoosed[0], axis= 0)
		testSet = np.append(testSet, dataChoosed[1], axis= 0)

	# 2.1 Features selection
	# Train
	trainSetFeatures = DA.featuresChooser(trainSet,(3,4,5,6,7,8)) # Depedent variables: linear acceleration and angular velocity
	trainSetTarget = DA.independentVariableChooser(trainSet,1) # Independent variable: activity (walking, sitting, laying, etc.)
	# Test
	testSetFeatures = DA.featuresChooser(testSet,(3,4,5,6,7,8))
	testSetTarget = DA.independentVariableChooser(testSet,1)

	# ---------------- 3. Machine learning algorithms ----------------
	BSL = BiometricsSupervisedLearning()
	# 3.1 Naive Bayes
	BSL.naiveBayes(trainSetFeatures,trainSetTarget,testSetFeatures,testSetTarget)
	
if __name__ == '__main__':
	main()