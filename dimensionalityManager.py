#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import parametersConfig


class manageDimensionality():

	def featureSelection(self,data):
		''' Removing all low-variance features '''
		print "Old shape", data.shape
		sel = VarianceThreshold(threshold=(parametersConfig.thresholdVariance * (1 - parametersConfig.thresholdVariance)))
		newDataset = sel.fit_transform(data)
		print "New shape", newDataset.shape
		print "Features selected", sel.get_support(True)
		return newDataset

	def featureExtraction(self,data):
		''' PCA '''
		pca = PCA()
		pca.fit(data)
		explained_variance_ratio = 0
		component = 1
		print "Variance by component", pca.explained_variance_ratio_

		while (explained_variance_ratio < parametersConfig.minExplainedVarianceRatio):
			explained_variance_ratio = sum(pca.explained_variance_ratio_[0:component])
			print "Components:", component
			print "Explained variance ratio:", explained_variance_ratio
			component +=1

		pca = PCA(n_components=component-1)
		pca.fit(data)
		print "Final explained variance", pca.explained_variance_ratio_
		newDataset = pca.transform(data)
		return newDataset
