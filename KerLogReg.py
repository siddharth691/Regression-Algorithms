import numpy as np
import math
from random import *
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv
import time

def read_file(filename):

	with open(filename) as f:
		f.readline()
		f.readline()
		f.readline()

		"""
		Creating a list of lists containing the entire dataset
		"""
		content = f.readlines()
		content = ([x.strip() for x in content])
		data = []
		for i in range(0,len(content)):
			content[i] = content[i].split(' ')
			data.append(list(map(float,content[i])))
		data = np.array(data)

	return data

def hypothesis(x_new):

	"""
	Using inner product within hilbert space
	"""
	x_new = np.array(x_new).reshape(-1,1)
	f_sum = 0
	for i, x in enumerate(self.X):
		x = np.array(x).reshape(-1,1)
		f_sum+= self.hil_alpha[i]*self.rbf(x,x_new,self.hyp)

	return f_sum

def rbf(x,x1,hyp):

	"""
	x: (no_features, 1)
	x1: (no_features, 1)

	output : scaler (one value of the Kernel)
	"""
	ell = hyp[0]
	sf = hyp[1]
	P = ell*ell*np.identity(x.shape[0])
	dist = (x-x1).T.dot(inv(P)).dot(x-x1);

	return sf*sf*np.exp(-0.5*dist)

def K_X_X1 (hyp, X, X1):
	
	K_X_X1 = []
	for x in X:
		x = x.reshape(-1,1)
		row = []
		for x1 in X1:
			x1 = x1.reshape(-1,1)
			row.append(rbf(x.reshape(-1,1), x1.reshape(-1,1), hyp)[0,0])
		K_X_X1.append(row)
	return np.array(K_X_X1)

def func(hyp, X, X1, alpha):
	alpha = alpha.reshape(-1,1)
	return alpha.T.dot(K_X_X1(hyp, X, X1))[0,0]


def predict_class(mean):

	mean = np.array(mean).reshape(-1,1)
	predict = np.zeros(np.shape(mean))
	predict[mean>0]=1
	predict[(mean==0)|(mean<0)]=0

	return predict
def main():

	#Formatting content.......

	print("Reading data.............")

	

	data = read_file('./data/oakland_part3_am_rf.node_features')
	data[:,4] = data[:,4].astype(int)
	test = read_file('./data/oakland_part3_an_rf.node_features')
	test[:,4] = test[:,4].astype(int)


	#############################################################################################
	#Taking two classes

	train_c1 = data[data[:,4] == 1004,:]
	train_c2 = data[data[:,4] == 1400,:]    
	train_c1 = train_c1[1:1000,:]
	train_c2 = train_c2[1:1000,:]

	new_data = np.concatenate((train_c1,train_c2), axis = 0)
	np.random.shuffle(new_data)

	new_data[new_data[:,4] == 1004,4] =0
	new_data[new_data[:,4] == 1400,4] =1

	test_c1 = test[test[:,4] == 1004,:]
	test_c2 = test[test[:,4] == 1400,:]
	test_c1 = test_c1[1:1000,:]
	test_c2 = test_c2[1:1000,:]


	new_test = np.concatenate((test_c1, test_c2), axis = 0)
	np.random.shuffle(new_test)
	new_test[new_test[:,4] == 1004,4] =0
	new_test[new_test[:,4] == 1400,4] =1


	#############################################################################################
	#Converting data into X and y, Xtest and ytest

	X = new_data[:,[0,1,2,3,5,6,7,8,9,10,11,12,13,14]]
	y = new_data[:,4].reshape(-1,1)

	print(y)
	Xt = new_test[:,[0,1,2,3,5,6,7,8,9,10,11,12,13,14]]
	yt = new_test[:,4].reshape(-1,1)

	batch_size = 50
	i = 0
	start_indexes = []
	alpha = np.ones((batch_size,1))
	eta = 0.08
	lam = 0.0001
	hyp = [8, 15]

	colors = ['green','yellow']
	labels = [0,1]

	t1 = time.time()
	while(i < len(X)):

		print(i)
		if(i + batch_size > len(X)):
			break

		Xb = X[i:i+batch_size,:]
		yb = y[i:i+batch_size,:]


		gamma = []
		for j in range(batch_size):

			gamma.append(yb[j][0]/(1+np.exp(-1*yb[j][0]*func(hyp, Xb, Xb[j,:], alpha))))

		gamma = np.array(gamma).reshape(-1,1)
		print('before alpha: {}'.format(np.shape(alpha)))

		alpha = alpha - eta*(lam*alpha + gamma)
		print(np.shape(eta*(lam*alpha + gamma)))
		print('after alpha: {}'.format(alpha))
		start_indexes.append(i)
		i+=batch_size

	t2 = time.time()
	print('training time: {}'.format(t2 - t1))
	predict = []

	t3 = time.time()
	for i, new_x in enumerate(Xt):

		cur_idx = random.choice(start_indexes)
		Xb = X[cur_idx:cur_idx+batch_size,:]
		yb = y[cur_idx:cur_idx+batch_size,:]
		predict.append(func(hyp, np.array(Xb).reshape(-1,14), np.array(new_x).reshape(1,-1), alpha))

	t4 = time.time()
	print('Prediction time: {}'.format(t4 - t3))

	predicted_output = np.array(predict_class(predict)).reshape(-1,)

	print('Error: {}'.format(len(predicted_output[predicted_output!=yt.reshape(-1,)])/len(predicted_output)))

	#Plotting the 3d point cloud
	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111, projection = '3d')
	
	#Plotting testing
	for index, label in enumerate(labels):

		test_labels = new_test[(predicted_output.reshape(-1,) == label)]
		ax2.scatter(test_labels[:,0], test_labels[:,1], test_labels[:,2], c = colors[index])

	plt.show()


if __name__ == '__main__':
	main()
