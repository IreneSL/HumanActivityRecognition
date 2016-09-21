windowSize = 250 # 5 seconds
frecuencyRate = 50 # 50 Hz
minExplainedVarianceRatio = 0.9
thresholdVariance = 0.8
headers = ['activity', \
		'xAccMean','yAccMean','zAccMean','xAngVelMean','yAngVelMean','zAngVelMean', \
		'xAccMedian','yAccMedian','zAccMedian','xAngVelMedian','yAngVelMedian','zAngVelMedian', \
		'xAccMinMax','yAccMinMax','zAccMinMax','xAngVelMinMax','yAngVelMinMax','zAngVelMinMax', \
		'xAccStd','yAccStd','zAccStd','xAngVelStd','yAngVelStd','zAngVelStd', \
		'xAcc1Q','yAcc1Q','zAcc1Q','xAngVel1Q','yAngVel1Q','zAngVel1Q', \
		'xAcc3Q','yAcc3Q','zAcc3Q','xAngVel3Q','yAngVel3Q','zAngVel3Q', \
		'xAccIQR', 'yAccIQR', 'zAccIQR', 'xAngVelIQR', 'yAngVelIQR', 'zAngVelIQR', \
		'xAccTBMaxP','yAccTBMaxP','zAccTBMaxP','xAngVelTBMaxP','yAngVelTBMaxP','zAngVelTBMaxP', \
		'xAccTBMinP','yAccTBMinP','zAccTBMinP','xAngVelTBMinP','yAngVelTBMinP','zAngVelTBMinP', \
		'xAccPKMax','yAccPKMax','zAccPKMax','xAngVelPKMax','yAngVelPKMax','zAngVelPKMax', \
		'xAccPKMin','yAccPKMin','zAccPKMin','xAngVelPKMin','yAngVelPKMin','zAngVelPKMin', \
		'xAccMaxPosPeaks', 'yAccMaxPosPeaks', 'zAccMaxPosPeaks', 'xAngVelMaxPosPeaks', 'yAngVelMaxPosPeaks', 'zAngVelMaxPosPeaks', \
		'xAccMaxNegPeaks', 'yAccMaxNegPeaks', 'zAccMaxNegPeaks', 'xAngVelMaxNegPeaks', 'yAngVelMaxNegPeaks', 'zAngVelMaxNegPeaks', \
		'xAccMinPosPeaks', 'yAccMinPosPeaks', 'zAccMinPosPeaks', 'xAngVelMinPosPeaks', 'yAngVelMinPosPeaks', 'zAngVelMinPosPeaks', \
		'xAccMinNegPeaks', 'yAccMinNegPeaks', 'zAccMinNegPeaks', 'xAngVelMinNegPeaks', 'yAngVelMinNegPeaks', 'zAngVelMinNegPeaks', \
		'xAccZC', 'yAccZC', 'zAccZC', 'xAngVelZC', 'yAngVelZC', 'zAngVelZC', \
		'corAccXY','corAccXZ','corAccYZ','corAngVelXY','corAngVelXZ','corAngVelYZ']


cacheSize = 1000 # Size of the kernel cache for SVC algorithm
SVMLibrary = 'LinearSVC'
# Decision trees
maxDepth = 3000
minSamplesLeaf = 3000