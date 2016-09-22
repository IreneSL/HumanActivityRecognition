#!/usr/bin/python
# -*- coding: utf-8 -*-
import parametersConfig
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import VarianceThreshold
import numpy as np

class manageDimensionality():

	def featureSelection(self,data,headers):
		""" Removing all features whose variance doesn’t meet some threshold

		Keywords arguments:
		data -- entire dataset (except labels)
		headers -- list with data headers (for knowing which features have been chosen)
		"""
		print "- Removing all features whose variance doesn’t meet", parametersConfig.thresholdVariance, "threshold"
		print "	+ Old shape", data.shape
		sel = VarianceThreshold(threshold=(parametersConfig.thresholdVariance * (1 - parametersConfig.thresholdVariance)))
		newDataset = sel.fit_transform(data)
		print "	+ New shape", newDataset.shape
		featuresSelectedHeaders = []
		featuresSelected = sel.get_support(True)
		for feature in featuresSelected:
			featuresSelectedHeaders.append(headers[feature])
		print " + Features selected", featuresSelectedHeaders
		return newDataset

	def featureExtraction(self,data):
		""" PCA (Principal Component Analysis)

		Keywords arguments:
		data -- entire dataset (except labels)
		"""
		print "- PCA (Choose components to explain", parametersConfig.minExplainedVarianceRatio, "variance of data)"
		pca = PCA()
		pca.fit(data)
		explained_variance_ratio = 0
		component = 1
		# Chooses more components until they explain, at least, 'minExplainedVarianceRatio'
		while (explained_variance_ratio < parametersConfig.minExplainedVarianceRatio):
			explained_variance_ratio = sum(pca.explained_variance_ratio_[0:component])
			print "	+ Components:", component
			print "	+ Explained variance ratio:", explained_variance_ratio
			component +=1
			print "------------"

		pca = PCA(n_components=component-1)
		pca.fit(data)
		print "	+ Final explained variance", explained_variance_ratio
		newDataset = pca.transform(data)
		return newDataset
