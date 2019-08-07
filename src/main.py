import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import PIL
from PIL import Image
import numpy as np
import cv2

def sigmoid(x, derivative=False):
	return x*(1-x) if derivative else 1/(1+np.exp(-x))

def scoreIsMaximumInLocalWindow(keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores):
	localMaximum = True
	height = len(scores)
	width = len(scores[0])

	yStart = max(heatmapY-localMaximumRadius, 0)
	yEnd = min(heatmapY+localMaximumRadius+1, height)
	#print("ystart and yend", yStart, yEnd)
	for yCurrent in range(yStart, yEnd):
		xStart = max(heatmapX-localMaximumRadius, 0)
		xEnd = min(heatmapX+localMaximumRadius+1, width)
		for xCurrent in range(xStart, xEnd):
			if sigmoid(scores[yCurrent][xCurrent][keypointId]) > score:
				localMaximum = False
				break
		if not localMaximum:
			break
	
	return localMaximum



def buildPartWithScoreQueue(scores, threshold, localMaximumRadius):
	print("pq build start")
	pq = []
	minScore = 9999.9999
	maxScore = -9999.9999
	print("dimension: ", len(scores), " ", len(scores[0]), " " , len(scores[0][0]))
	for heatmapY in range(len(scores)):
		for heatmapX in range(len(scores[0])):
			for keypointId in range(len(scores[0][0])):
				score = sigmoid(scores[heatmapY][heatmapX][keypointId])
				if score < 0.0:#threshold:
					continue
				if score > maxScore:
					maxScore = score
				if score < minScore:
					minScore = score
				#print(score)
				if scoreIsMaximumInLocalWindow(keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores):
					res = {}
					res["score"]=score
					res["y"]=heatmapY
					res["x"]=heatmapX
					res["partId"]=keypointId
					pq.append(res)

	pq.sort(key=lambda _score: _score["score"], reverse = True)
	print("min score: ", minScore)
	print("max score: ", maxScore)
	print("pq build complete. Length of pq is ",len(pq), "\n\n")
	return pq

def getImageCoords(keypoint, outputStride, numParts, offsets):
	print("## getImageCoords")
	heatmapY = int(keypoint["y"])
	heatmapX = int(keypoint["x"])
	print(heatmapY, heatmapX)
	keypointId = int(keypoint["partId"])
	offsetY = offsets[heatmapY][heatmapX][keypointId]
	offsetX = offsets[heatmapY][heatmapX][keypointId + numParts]
	
	y = heatmapY * outputStride + offsetY
	x = heatmapX * outputStride + offsetX

	re = [y, x]
	return re

def withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, y, x, keypointId):
	for idx in range(len(poses)):
		pose = poses[idx]
		keypoints={}
		keypoints = pose["keypoints"]
		correspondingKeypoint={}
		correspondingKeypoint=keypoints[keypointId]
		_x=float(correspondingKeypoint["x"] * inputSize - x)
		_y=float(correspondingKeypoint["y"] * inputSize - y)
		squaredDistance = _x*_x + _y*_y
		if squaredDistance <= squaredNmsRadius:
			return True

	return False

def traverseToTargetKeypoint(edgeId, sourceKeypoint, targetKeypointId, scores, offsets, outputStride, displacements):
	height = len(scores)
	width = len(scores[0])
	numKeypoints = len(scores[0][0])
	sourceKeypointY = float(sourceKeypoint["y"] * inputSize)
	sourceKeypointX = float(sourceKeypoint["x"] * inputSize)

	sourceKeypointIndices = getStridedIndexNearPoint(sourceKeypointY, sourceKeypointX, outputStride, height, width)
	displacement = getDisplacement(edgeId, sourceKeypointIndices, displacements)
	print("displacement: ", displacement)
	displacedPoint = [sourceKeypointY + displacement[0], sourceKeypointX + displacement[1]]
	
	targetKeypoint = displacedPoint

	offsetRefineStep = 5
	for i in range(offsetRefineStep):
		targetKeypointIndices = getStridedIndexNearPoint(targetKeypoint[0], targetKeypoint[1], outputStride, height, width)
		targetKeypointY = int(targetKeypointIndices[0])
		targetKeypointX = int(targetKeypointIndices[1])

		offsetY = offsets[targetKeypointY][targetKeypointX][targetKeypointId]
		offsetX = offsets[targetKeypointY][targetKeypointX][targetKeypointId + numKeypoints]
		targetKeypoint = [targetKeypointY*outputStride + offsetY, targetKeypointX*outputStride + offsetX]
	
	
	targetKeypointIndices = getStridedIndexNearPoint(targetKeypoint[0], targetKeypoint[1], outputStride, height, width)
	score = sigmoid(scores[int(targetKeypointIndices[0])][int(targetKeypointIndices[1])][targetKeypointId])
	keypoint = {}
	keypoint["score"] = score
	keypoint["part"] = partNames[targetKeypointId]
	keypoint["y"] = targetKeypoint[0] / inputSize
	keypoint["x"] = targetKeypoint[1] / inputSize
	return keypoint

def getStridedIndexNearPoint(_y, _x, outputStride, height, width):
	y_ = np.around(_y / outputStride)
	x_ = np.around(_x / outputStride)
	y = y_
	x = x_
	if y<0:
		y = 0
	elif y>height-1:
		y=height-1
	if x<0:
		x = 0
	elif x>width-1:
		x=width-1
	return [y, x]

def getDisplacement(edgeId, keypoint, displacements):
	numEdges = int(len(displacements[0][0])/2)
	y = int(keypoint[0])
	x = int(keypoint[1])
	return [displacements[y][x][edgeId], displacements[y][x][edgeId+numEdges]]

def getInstanceScore(keypoints, numKeypoints):
	scores = 0
	for keyIdx in keypoints:
		scores = scores + keypoints[keyIdx]["score"]
	
	return scores / numKeypoints




if __name__ =="__main__":

	# DEF. PARAMETERS
	img_row, img_column = 500,500
	num_channel = 3
	num_batch = 1
	inputSize=337
	inputChannels=3
	threshold = 0.5
	localMaximumRadius = 1
	nmsRadius = 10
	imageMean=125.0
	imageStd=125.0
	numResults=1
	outputStride=16
	

	# include the path containing the model (.lite, .tflite)
	path_1 = r'../model/posenet_mv1_075_float_from_checkpoints.tflite'


	# TFLITE INTERPRETER CON.
	interpreter = tf.contrib.lite.Interpreter(path_1)
	interpreter.allocate_tensors()


	# obtaining the input-output shapes and types
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	print("## input_details")
	print(input_details)
	print("## output_details")
	print(output_details, "\n\n")


	# INPUT SELECTION
	file_path = r'../data/input/1.jpg'
	white_file_path = '../data/output/white.jpg'
	input_img = Image.open(file_path)
	img_row, img_column = input_img.size[1], input_img.size[0]

	print("## input_img size")
	print(input_img.size)
	print("row(=y), column(=x) : ", img_row, ", ", img_column, "\n\n")


	# resizing input_img and make two x_matrix. one for calculate, one for draw a pose
	x_matrix_uint8 = np.array([np.asarray(input_img)], dtype=np.uint8)
	input_img = input_img.resize((inputSize, inputSize))
	x_matrix = np.array([np.asarray(input_img)], 'f')

	print("## resized_img size: ")
	print(input_img.size, "\n\n")

	print("## x_matrix")
	print(x_matrix.shape)
	print(x_matrix.dtype)
	print("## x_matrix_uint8")
	print(x_matrix_uint8.shape)
	print(x_matrix_uint8.dtype, "\n\n")


	# normalize with two parameters, imageMean and imageStd
	for idxi in range(inputSize):
		for idxj in range(inputSize):
			for idxk in range(3):
				x_matrix[0][idxi][idxj][idxk] = (x_matrix[0][idxi][idxj][idxk]-imageMean)/imageStd
				
	
	# create white background
	for idxi in range(img_row):
		for idxj in range(img_column):
			for idxk in range(3):
				x_matrix_uint8[0][idxi][idxj][idxk]=255

	whiteImg = Image.fromarray(x_matrix_uint8[0], 'RGB')
	whiteImg.save(white_file_path)


	# RUNNING INTERPRETER
	# setting the input tensor with the selected input
	interpreter.set_tensor(input_details[0]['index'], x_matrix)

	# running inference
	interpreter.invoke()

	scores  = interpreter.get_tensor(output_details[0]['index'])[0]
	offsets = interpreter.get_tensor(output_details[1]['index'])[0]
	displacementsFwd = interpreter.get_tensor(output_details[2]['index'])[0]
	displacementsBwd = interpreter.get_tensor(output_details[3]['index'])[0]


	print("## output_details")
	print(output_details, "\n\n")
	'''
	# check details
	print("## scores")
	print("shape: ", scores.shape)
	#print("value: ", scores)
	for i in range(len(output_details)):
		print("## output_detail[",i,"]['name'], output_details[",i,"]['name' , 'shape']")
		# print(output_details[i]['name'], output_details[i]['shape'], "\n")
	'''


	
	# set the array parentToChildEdges/childToParentEdges. link two keypoints those could be connected in natural
	partNames = ["nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
	    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
	    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"]

	poseChain = [
	    ["nose", "leftEye"], ["leftEye", "leftEar"], ["nose", "rightEye"],
	    ["rightEye", "rightEar"], ["nose", "leftShoulder"],
	    ["leftShoulder", "leftElbow"], ["leftElbow", "leftWrist"],
	    ["leftShoulder", "leftHip"], ["leftHip", "leftKnee"],
	    ["leftKnee", "leftAnkle"], ["nose", "rightShoulder"],
	    ["rightShoulder", "rightElbow"], ["rightElbow", "rightWrist"],
	    ["rightShoulder", "rightHip"], ["rightHip", "rightKnee"],
	    ["rightKnee", "rightAnkle"]]

	drawChain = [
		["leftShoulder", "rightShoulder"], ["rightHip", "leftHip"],
	    ["leftShoulder", "leftElbow"], ["leftElbow", "leftWrist"],
	    ["leftShoulder", "leftHip"], ["leftHip", "leftKnee"],
	    ["leftKnee", "leftAnkle"],
	    ["rightShoulder", "rightElbow"], ["rightElbow", "rightWrist"],
	    ["rightShoulder", "rightHip"], ["rightHip", "rightKnee"],
	    ["rightKnee", "rightAnkle"]]

	partsIds = {}
	parentToChildEdges = []
	childToParentEdges = []
	drawEdges = []
	for i in range(len(partNames)):
		partsIds[partNames[i]]=i
		
	for i in range(len(poseChain)):
		parentToChildEdges.append(partsIds[poseChain[i][1]])
		childToParentEdges.append(partsIds[poseChain[i][0]])

	for i in range(len(drawChain)):
		drawEdges.append( (partsIds[drawChain[i][0]], partsIds[drawChain[i][1]]) )


	# get keypoints with high score
	pq = []
	pq = buildPartWithScoreQueue(scores, threshold, localMaximumRadius)
	for pq_idx in range(len(pq)):
		print(pq[pq_idx])
	print("\n\n")



	# start with root keypoint,(highest keypoints) and then link other keypoints which have high probability to be linked with root keypoint
	print("## start traverseToTargetKeypoint")
	numParts = len(scores[0][0])
	numEdges = len(parentToChildEdges)
	sqaredNmsRadius = nmsRadius * nmsRadius

	results = []
	pq_idx=0
	
	while len(results) < numResults and pq_idx<len(pq):
		keyPoint = {}
		keyPoints = {}
		root = pq[pq_idx]
		pq_idx = pq_idx+1
		rootPoint = getImageCoords(root, outputStride, numParts, offsets)
		if withinNmsRadiusOfCorrespondingPoint(results, sqaredNmsRadius, rootPoint[0], rootPoint[1], int(root["partId"])):
			print("over NMS radius")
			continue
		print("add to keyPoints")
		keyPoint["score"] = root["score"]
		keyPoint["part"] = partNames[int(root["partId"])]
		keyPoint["y"] = rootPoint[0]/inputSize
		keyPoint["x"] = rootPoint[1]/inputSize
		keyPoints[root["partId"]] = keyPoint

		# traverseToTargetKeypoint. link the keypoints with high probability to be linked with sourceKeypoint
		for edge in reversed(range(numEdges)):
			sourceKeypointId = parentToChildEdges[edge]
			targetKeypointId = childToParentEdges[edge]
			if sourceKeypointId in keyPoints and targetKeypointId not in keyPoints:
				keyPoint = traverseToTargetKeypoint(edge, keyPoints[sourceKeypointId], targetKeypointId, scores, offsets, outputStride, displacementsBwd)
				keyPoints[targetKeypointId]=keyPoint
		for edge in range(numEdges):
			sourceKeypointId = childToParentEdges[edge]
			targetKeypointId = parentToChildEdges[edge]
			if sourceKeypointId in keyPoints and targetKeypointId not in keyPoints:
				keyPoint = traverseToTargetKeypoint(edge, keyPoints[sourceKeypointId], targetKeypointId, scores, offsets, outputStride, displacementsFwd)
				keyPoints[targetKeypointId]=keyPoint

		result = {}
		result["keypoints"] = keyPoints
		result["score"] = getInstanceScore(keyPoints, numParts)
		results.append(result)
		pq_idx += 1



	# get result
	print('## results')



	# calculate average accuracy. and modify the image.(add keypoint)
	total_accuracy=0.0
	total_spots = [] 
	for idx in range(len(results[0]["keypoints"])):
		print(results[0]["keypoints"][idx])
		total_accuracy+=results[0]["keypoints"][idx]['score']
		yf = results[0]["keypoints"][idx]['y']*img_row
		yint = int(np.around(yf))
		xf = results[0]["keypoints"][idx]['x']*img_column
		xint = int(np.around(xf))
		
		keySpot={}
		keySpot['x']=xint
		keySpot['y']=yint
		total_spots.append(keySpot)

	total_accuracy/=len(results[0]["keypoints"])
	print("average accuracy : ", total_accuracy)


	# opencv2 works
	cv_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
	cv_white_img = cv2.imread(white_file_path, cv2.IMREAD_UNCHANGED)
	for idx in range(len(total_spots)):
		yint,xint=total_spots[idx]['y'], total_spots[idx]['x']
		cv_img = cv2.line(cv_img, (xint, yint), (xint, yint), (0,0,255), 5)
		cv_white_img = cv2.line(cv_white_img, (xint, yint), (xint, yint), (0,0,255), 5)

	for edge in range(len(drawEdges)):
		sourceKeypointId = drawEdges[edge][0]
		targetKeypointId = drawEdges[edge][1]
		y1int, x1int=total_spots[sourceKeypointId]['y'], total_spots[sourceKeypointId]['x']
		y2int, x2int=total_spots[targetKeypointId]['y'], total_spots[targetKeypointId]['x']

		cv_img = cv2.line(cv_img, (x1int, y1int), (x2int, y2int),(0,0,255), 2)	
		cv_white_img = cv2.line(cv_white_img, (x1int, y1int), (x2int, y2int),(0,0,255), 2)	


	cv2.imwrite('../data/output/bone.jpg', cv_img)
	cv2.imwrite('../data/output/bone_white.jpg', cv_white_img)

