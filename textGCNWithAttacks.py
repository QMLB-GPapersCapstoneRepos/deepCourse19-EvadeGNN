import os
import sys
import keras
import torch
import random
import gensim
from os import walk
import numpy as np
import pandas as pd
from sklearn.metrics import *
import torch.nn.functional as F
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# load embeddings
"""You'll need to place embeddings at Embeddings/gloveEmbeddings.txt"""
wordVectorModel = gensim.models.KeyedVectors.load_word2vec_format("Embeddings/gloveEmbeddings.txt")


# hyperparameters
WINDOW_SIZE = 1

#######################################################################################################################################################################
# load data

authors = []
for(_,dirs,_) in walk("AuthorsData"):
	authors.extend(dirs)
print('Authors list', authors)

authors = list(sorted(authors))

AUTHORS_TO_KEEP = 5
authorToId = dict([(authors[i], i) for i in range(0, len(authors))])

authorsData = []
for author in authors[:AUTHORS_TO_KEEP]:
	authorId = authorToId[author]

	# get author data
	authorFiles = []
	for(_,_,files) in walk("AuthorsData/" + author):
		authorFiles.extend(files)

	for file in authorFiles:
		data = open("AuthorsData/" + author + "/" + str(file), "r").readlines()
		data = ''.join(str(line) + " " for line in data)
		authorsData.append([authorId, data])

random.shuffle(authorsData)
rawData = list(authorsData)
print('Length of author data', len(authorsData))
dataSizeToUse = len(rawData)

data = list(authorsData)
wordsDict = {}
for row in data:
	rowWords = keras.preprocessing.text.text_to_word_sequence(row[1], filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
	for word in rowWords:
		wordsDict[word] = True

# compute total words in corpus
allWordsInCorpus = list(sorted(list(wordsDict.keys())))
print('Total words found in corpus are', len(allWordsInCorpus))

# dictionaries for mapping words to ids
word2Id = {}
id2Word = {}

# start from index after counting all documents since we need to assign ids to documents as well
wordIndex = len(data) + 1
startWordIndex = int(wordIndex)
for word in allWordsInCorpus:
	word2Id[word] = wordIndex
	id2Word[wordIndex] = word

	# increase index
	wordIndex = wordIndex + 1

endWordIndex = int(wordIndex)
print('Start and end indices for nodes are', startWordIndex, endWordIndex)

##################################################################### create graph data ###############################################################
## creating node to node graph

# nodes that are connected via edges
edgeIndexNodesOne = []
edgeIndexNodesTwo = []

totalEdgesCreated = 0
edgesDictionary = {}

wordPresenceDict = {}
for i in range(0, len(data)):
	row = data[i]
	label = row[0]

	words = keras.preprocessing.text.text_to_word_sequence(row[1], filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

	# go over a window to create co-occurrence counts
	for j in range(WINDOW_SIZE, len(words) - WINDOW_SIZE):
		wordsInWindow = words[j - WINDOW_SIZE: j + WINDOW_SIZE]

		# iterating over all pairs
		for k in range(0, len(wordsInWindow)):
			for l in range(k, len(wordsInWindow)):

				firstWord = wordsInWindow[k]
				secondWord = wordsInWindow[l]

				if firstWord == secondWord:
					continue

				# skip if edge already present
				edgeKey = firstWord + "::" + secondWord
				if edgeKey in edgesDictionary:
					continue

				# creating an edge
				#print('Creating an edge between %s and %s with indexes %d %d' % (firstWord, secondWord, k, l))
				edgeIndexNodesOne.append(word2Id[firstWord])
				edgeIndexNodesOne.append(word2Id[secondWord])

				edgeIndexNodesTwo.append(word2Id[secondWord])
				edgeIndexNodesTwo.append(word2Id[firstWord])
		
				totalEdgesCreated = totalEdgesCreated + 1

				# store edges for removing redundancy
				edgesDictionary[firstWord + "::" + secondWord] = True

				wordPresenceDict[firstWord] = True
				wordPresenceDict[secondWord] = True

print('Total edges in the graph after node 2 node graph creation', totalEdgesCreated)

## creating document to node graph
doc2NodeEdgeInfo = {}
for i in range(0, len(data)):
	row = data[i]
	label = row[0]

	words = keras.preprocessing.text.text_to_word_sequence(row[1], filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
	for word in words:

		doc2NodeEdgeInfo[len(edgeIndexNodesOne)] = [word, word2Id[word], i, label]

		edgeIndexNodesOne.append(i)
		edgeIndexNodesOne.append(word2Id[word])

		edgeIndexNodesTwo.append(word2Id[word])
		edgeIndexNodesTwo.append(i)

		

		totalEdgesCreated = totalEdgesCreated + 1

#print(doc2NodeEdgeInfo)
print('Total Edges that we can modify', len(doc2NodeEdgeInfo))
print('Total edges in the graph after document 2 node graph creation', totalEdgesCreated)

## node features
nodeFeatures = []
for i in range(0, len(data)):
	array = [0] * endWordIndex
	array[i] = 1
	nodeFeatures.append(array)

print('Length of node features after adding documents', len(nodeFeatures))
for j in range(startWordIndex, endWordIndex + 1):
	array = [0] * endWordIndex
	try:
		array[j] = 1
	except Exception as e:
		print('Error',j)
	nodeFeatures.append(array)

print('Length of node features after adding nodes', len(nodeFeatures))
nodeFeatureLength = endWordIndex

################################################################## create stuff for pytorch #################################################################

######### GCN model ##############
class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = GCNConv(nodeFeatureLength, 64)
		self.conv2 = GCNConv(64, AUTHORS_TO_KEEP)
		
	def forward(self, x, edge_index):
		#x, edge_index = data.x, data.edge_index
		x = F.relu(self.conv1(x, edge_index))
		x = F.dropout(x, 0.4)
		x = self.conv2(x, edge_index)
		return F.log_softmax(x, dim=1)  #log_softmax


trainingPercentage = 80
trainingSamples = int((len(rawData) * trainingPercentage) / 100)
trainMask = torch.tensor([1 if i < trainingSamples else 0 for i in range(0, dataSizeToUse)], dtype=torch.long)
validationMask = trainMask
testMask = torch.tensor([1 if i >= trainingSamples else 0 for i in range(0, dataSizeToUse)], dtype=torch.long)

edgeIndex = torch.tensor([edgeIndexNodesOne,
                           edgeIndexNodesTwo], dtype=torch.long)

x = torch.tensor(nodeFeatures, dtype=torch.float)
y = torch.tensor([row[0] for row in data], dtype=torch.long)
data = Data(x=x, edge_index=edgeIndex, y = y)
data.train_mask = trainMask
data.val_mask = validationMask
data.test_mask = testMask

nodesToUse = torch.tensor([i for i in range(0, trainingSamples)], dtype=torch.long)
testNodesToUse = [i for i in range(trainingSamples, len(rawData))]
testLabelsToUse = [rawData[i][0] for i in testNodesToUse]
#print('Train and test nodes to use', nodesToUse, testNodesToUse)
#exit()

data.train_idx = nodesToUse


print('Shape of edge index, x, and data', edgeIndex.shape, x.shape, data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('Model has been created')

def train():
	#model.train()
	optimizer.zero_grad()
	out = model(data.x, data.edge_index)

	#nn.CrossEntropyLoss(), F.nll_loss
	loss = F.nll_loss(out[data.train_idx], data.y[:trainingSamples])

	lossVal = round(float(loss.item()), 3)

	loss.backward()
	optimizer.step()

	return lossVal


def test():
	model.eval()
	logits, accs = model(data.x, data.edge_index), []

	# test scores
	predTest = logits[trainingSamples:dataSizeToUse].max(1)[1]
	print(predTest)
	labelsTest = [row[0] for row in rawData[trainingSamples:dataSizeToUse]]
	accuracyTest = accuracy_score(labelsTest, predTest.tolist())

	# train scores
	predTrain = logits[:trainingSamples].max(1)[1]
	labelsTrain = [row[0] for row in rawData[:trainingSamples]]
	accuracyTrain = accuracy_score(labelsTrain, predTrain.tolist())

	accuracyTrain = round(accuracyTrain, 3)
	accuracyTest = round(accuracyTest, 3)

	return accuracyTrain, accuracyTest
	
best_val_acc = test_acc = 0
for epoch in range(1, 60):
	lossVal = train()
	accuracyTrain, accuracyTest = test()
	print('Epoch, Loss, Train and Test accuracy', epoch, lossVal, accuracyTrain, accuracyTest)
	
for param in model.parameters():
	param.requires_grad = False

## items to use during attack
edgesList = list(doc2NodeEdgeInfo.keys())
wordIds = list(word2Id.values())

## attacking
attackIterations = 5
modificationsInIterationRange = 10
neighborsCount = 10
neighborsToKeep = 3
epochs = 50
topGenes = 5

successfulAttacks = []
totalNodes = 0


from collections import defaultdict

confidenceDecrease = defaultdict(list)
for j in range(0, len(testNodesToUse)):

	testNode = testNodesToUse[j]
	testNodeLabel = testLabelsToUse[j]

	# initial prediction confidence
	logits = model(data.x, data.edge_index)
	testNodeConfidenceBeforeAttack = logits[testNode:testNode+1][0].tolist()[testNodeLabel]
	testNodeExactPredictions = int(logits[testNode:testNode+1].max(1)[1][0])

	if testNodeExactPredictions != testNodeLabel:
		print('Already wrong prediction, no need to obfuscate@')
		continue

	totalNodes = totalNodes + 1
	print('**********************************')
	print('Test node and its initial prediction confidence', testNode, testNodeConfidenceBeforeAttack, testNodeLabel)
	print('**********************************')

	# genes
	genesList = []
	genesList.append([testNodeConfidenceBeforeAttack, edgeIndexNodesOne, edgeIndexNodesTwo])


	nodeObfuscated = False
	for epoch in range(epochs):

		# keep the top genes
		print('Number of items in gene list before selection', len(genesList))

		genesList = list(sorted(genesList))[:topGenes]
		scores = [item[0] for item in genesList]
		print('Top scores in this epoch', scores)
		confidenceDecrease[testNode].append(scores[0])


		for iteration in range(attackIterations):

			# get one of the top genes
			confidenceScore, edgeIndexNodesOneForAttack, edgeIndexNodesTwoForAttack = random.choice(genesList)

			# modify edges in the graph
			modificationsInIteration = random.randrange(1, modificationsInIterationRange)
			modifications = 0
			while modifications < modificationsInIteration:

				# create an attack
				edgeToModify = random.choice(edgesList)
				word, wordId, node, label = doc2NodeEdgeInfo[edgeToModify]

				while node != testNode:
					edgeToModify = random.choice(edgesList)
					word, wordId, node, label = doc2NodeEdgeInfo[edgeToModify]

				edgeIndexValue = edgeToModify
				try:
					wordNeighbors = list(wordVectorModel.similar_by_word(word, neighborsCount))
					wordNeighbors = [item[0] for item in wordNeighbors]
					wordNeighbors = [neighbor for neighbor in wordNeighbors if neighbor in word2Id][:neighborsToKeep]

					if len(wordNeighbors) == 0:
						continue
				except:
					continue


				randomNeighbor = random.choice(wordNeighbors) # random replacement -> random.choice(list(word2Id.keys()))#
				edgeIndexNodesOneForAttack[edgeIndexValue] = node
				edgeIndexNodesOneForAttack[edgeIndexValue + 1] = word2Id[randomNeighbor]

				edgeIndexNodesTwoForAttack[edgeIndexValue] = word2Id[randomNeighbor]
				edgeIndexNodesTwoForAttack[edgeIndexValue + 1] = node

				modifications = modifications + 1

			# test attack
			edgeIndexNodesOneForAttack = list(edgeIndexNodesOneForAttack)
			edgeIndexNodesTwoForAttack = list(edgeIndexNodesTwoForAttack)

			edgeIndexUpdated = torch.tensor([edgeIndexNodesOneForAttack,
		                           edgeIndexNodesTwoForAttack], dtype=torch.long)

			xNew = torch.tensor(nodeFeatures, dtype=torch.float)
			dataUpdated = Data(x=xNew, edge_index=edgeIndexUpdated)

			# get results
			model.eval()
			logits = model(dataUpdated.x, dataUpdated.edge_index)

			# test node confidence
			testNodeConfidenceAfterAttack = logits[testNode:testNode+1][0].tolist()[testNodeLabel]
			testNodeExactPredictionAfterAttack = int(logits[testNode:testNode+1].max(1)[1][0])
			allPredictionsAfterAttack = logits[testNode:testNode+1]
			

			print('Iteration and test node confidence', iteration, testNodeConfidenceAfterAttack, testNodeLabel, testNodeExactPredictionAfterAttack,
																										allPredictionsAfterAttack)
			if testNodeLabel != testNodeExactPredictionAfterAttack:
				successfulAttacks.append([testNode, iteration, epoch, testNodeConfidenceAfterAttack, testNodeConfidenceBeforeAttack])
				nodeObfuscated = True
				break

			# add to genes
			genesList.append([testNodeConfidenceAfterAttack, edgeIndexNodesOneForAttack, edgeIndexNodesTwoForAttack])

		if nodeObfuscated == True:
			break	


print(successfulAttacks)
print(len(successfulAttacks))
print(totalNodes)

np.save("confidenceScores.npy", confidenceDecrease)