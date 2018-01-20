import matplotlib.pyplot as plt
import numpy as np
import collections
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

np.random.seed(42)


equal_distance_counter = 0
equal_vote_counter = 0
k =  5

#0, 1, 2 or 3
#0 none, 1 only k, 2 only vote, 3 both
OPTIMIZED_MODE = 0


dataset = load_iris()

#information[i] to look at the i'ths data sample
X = dataset.data
#classes[i] to get the class of the i'ts component
y = dataset.target

#split set into train and seperate test set
X_train, X_test, y_train, y_test = train_test_split(X, y ,train_size=0.7)

def subList(posX,k):
    if(posX-k < 0):
        return dataArr[0:posX+k+1]
    elif(posX+k > len(dataArr)):
        return dataArr[posX-k:len(dataArr)]
    else:
        return dataArr[posX-k:posX+k+1]

def closestIndex(x):
    i=0
    while i < len(dataArr) and x > dataArr[i]:
        i += 1
    return i

#expects two numpy array
def calcManhattanDistance(point1, point2):
	len1 = len(point1)
	len2 = len(point2)

	assert len1 == len2, "point1 and point2 have different dimensions: {}, {}"\
	                     .format(len1,len2)

	distances = np.zeros((len1))
	for i in range(len1):
		distances[i] = abs(point1[i] - point2[i])

	return sum(distances)

#dataPoints as numpy arr, distanceMetric is a function to calculate the distance
#between two points
def calcDistances(dataPoints, unkownPoint, distanceMetric):

	distances = np.zeros((len(dataPoints),2))

	for i in range(len(dataPoints)):
		dist = distanceMetric(dataPoints[i], unkownPoint)
		distances[i][0] = dist
		distances[i][1] = i

	sortedDistances = sorted(distances,key=lambda distances: distances[0])
	return sortedDistances

def kNearestPoints(sortedDistances, k):
	global equal_distance_counter
	k_old = k
	#increase the range if the last one you pick is the same length as the following
	while k < len(sortedDistances) and\
	      sortedDistances[k-1][0] == sortedDistances[k][0]:
		equal_distance_counter += 1
		k += 1

	#TODO: dynamic somewhat more elegant
	if(OPTIMIZED_MODE == 1 or OPTIMIZED_MODE == 3):
		return sortedDistances[0:k]
	else:
		return sortedDistances[0:k_old]

#training Point is a array with index and distance
def getClassForTrainingPoint(trainingPoint):
	index = int(trainingPoint[1])
	return y_train[index]

def vote(distanceArr):
	distinctClasses = np.array(list(set(y_train)))
	voteCount = np.zeros(len(distinctClasses))

	for i in range(len(distanceArr)):
		pClass = getClassForTrainingPoint(distanceArr[i])
		for i in range(len(distinctClasses)):
			if(pClass == distinctClasses[i]):
				voteCount[i] += 1
				break

	equalVotesIndices = np.where( voteCount == max(voteCount))[0]
	if  len(equalVotesIndices) > 1:
		global equal_vote_counter
		equal_vote_counter += len(equalVotesIndices) -1

		if(OPTIMIZED_MODE == 2 or OPTIMIZED_MODE == 3):
			reversedDistanceArr = distanceArr[::-1]
			for elem in reversedDistanceArr:
				elemClass = getClassForTrainingPoint(elem)
				for indices in equalVotesIndices:
					if elemClass == distinctClasses[indices]:
						return elemClass
			print("THIS SHOULD NOT HAPPEN")

	return np.argmax(voteCount)

#p(x) = K/NV
def prob(x,k):
    volume = calcVolume(x,k)
    return k/(numberOfData * volume)

def predictAPoint(trainingData, point, k):
	distances = calcDistances(trainingData,point,calcManhattanDistance)
	distances = kNearestPoints(distances, k)
	return vote(distances)

def main():
	totalWrong = 0
	for k in range(1,105):
		#print("Iteration ", k)
		correctGuessed = 0
		wronglyGuessed = 0
		for i in range(len(X_test)):
			pointClass = predictAPoint(X_train, X_test[i], k)
			if(pointClass == y_test[i]):
				correctGuessed += 1
			else:
				wronglyGuessed += 1
		#print("Results: c: {}, w: {}".format(correctGuessed, wronglyGuessed))
		totalWrong += wronglyGuessed
	print("FINAL STATS: equal distances: {}, equal Votes: {}"\
	      .format(equal_distance_counter, equal_vote_counter))
	print("Total wrong guesses: ", totalWrong)

if __name__ == "__main__":
	main()

#FAZIT:
#vorher 4 danach 5 o.O
