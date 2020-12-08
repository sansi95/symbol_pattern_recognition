# import the necessary packages
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from imutils import paths
import argparse
import cv2
import os
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from symbol import symbol

def binary_function(type_rec):

	l = []
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--training", required=False, default= 'images/training/',
		help="path to the training images")
	ap.add_argument("-e", "--testing", required=False, default= 'images/testing/',
		help="path to the tesitng images")
	args = vars(ap.parse_args())
	# initialize the local binary patterns descriptor along with
	# the data and label lists
	if type_rec == 'symbol':
		desc = LocalBinaryPatterns(24, 8)
	else:
		desc = LocalBinaryPatterns(24, 3)
	data = []
	labels = []

	# loop over the training images
	for imagePath in paths.list_images(args["training"]):
	# for imagePath in os.listdir('images/training/'):
		# load the image, convert it to grayscale, and describe it
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		hist = desc.describe(gray)
		# extract the label from the image path, then update the
		# label and data lists
		labels.append(imagePath.split(os.path.sep)[-2])
		data.append(hist)
	model = DecisionTreeClassifier(random_state=0)
	model.fit(data, labels)

	# loop over the testing images
	for imagePath in paths.list_images(args["testing"]):
	# for imagePath in os.listdir('images/testing/'):
		# load the image, convert it to grayscale, describe it,
		# and classify it
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		hist = desc.describe(gray)
		prediction = model.predict(hist.reshape(1, -1))

		# display the image and the prediction
		# cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		# 	1.0, (0, 0, 255), 3)
		# cv2.imshow("Image", image)
		# cv2.waitKey(0)
		l.append([prediction[0], imagePath])

	# print (l)
	if type_rec == 'symbol':
		result = symbol(l[0][0])
		string =  "The symbol is " + str(l[0][0]) +  " Meaning of symbol: " + result[0] + " Reference: " + result[1]
	else:
		s_p = []
		for i in l:
			s_p.append(i[0])
		patt = ', '.join(s_p)
		string = "The symbols identified are " + patt

	return string
