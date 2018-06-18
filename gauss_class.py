#Gaussian Process Regressor
import numpy as np
from random import *
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

def predict_class(mean):

    mean = mean.reshape(-1,1)
    predict = np.zeros(np.shape(mean))
    predict[mean>0]=1
    predict[mean==0]=0

    return predict

#Coding GPR using Gaussian Kernel
class GaussianProcessRegressor:

    def __init__(self, hyp, kernel, sigma):
        self.hyp = hyp
        self.sigma = sigma

        try:
            self.kernel = getattr(self, kernel)

        except AttributeError:
            raise NotImplementedError(" No such kernel")



    def rbf(self, x,x1,hyp):

        """
        x: (no_features, 1)
        x1: (no_features, 1)

        output : scaler (one value of the Kernel)
        """
        ell = hyp[0]
        sf = hyp[1]
        P = ell*ell*np.identity(x.shape[0])
        dist = (x-x1).T.dot(inv(P)).dot(x-x1);

        return sf*sf*np.exp(-0.5*dist);


    def K_X_X (self, hyp, kernel, X, X1):

        K_X_X = []
        for x in X:
            x = x.reshape(-1,1)
            row = []
            for x1 in X1:
                x1 = x1.reshape(-1,1)
                row.append(kernel(x.reshape(-1,1),x1.reshape(-1,1),hyp)[0,0])
            K_X_X.append(row)
        return np.array(K_X_X)

    def K_X1_X1 (self, hyp, kernel, X, X1):

        K_X1_X1 = []
        for x in X:
            x = x.reshape(-1,1)
            row = []
            for x1 in X1:
                x1 = x1.reshape(-1,1)
                row.append(kernel(x.reshape(-1,1),x1.reshape(-1,1),hyp)[0,0])
            K_X1_X1.append(row)
        return np.array(K_X1_X1)


    def K_X_X1 (self, hyp, kernel, X, X1):
        K_X_X1 = []
        for x in X:
            x = x.reshape(-1,1)
            row = []
            for x1 in X1:
                x1 = x1.reshape(-1,1)
                row.append(kernel(x.reshape(-1,1), x1.reshape(-1,1), hyp)[0,0])
            K_X_X1.append(row)
        return np.array(K_X_X1)

    def K_X1_X (self, hyp, kernel, X1, X):
        K_X1_X = []
        for x in X1:
            x = x.reshape(-1,1)
            row= []
            for x1 in X:
                x1 = x1.reshape(-1,1)
                row.append(kernel(x.reshape(-1,1),x1.reshape(-1,1),hyp)[0,0])
            K_X1_X.append(row)
        return np.array(K_X1_X)

    def fit(self,X,y):

        """
        X(number of samples, number of features)
        y(number of samples, 1)

        """
        self.X = X
        self.y = y

        self.alpha = np.linalg.inv(self.K_X_X(self.hyp, self.kernel, self.X, self.X) + self.sigma*self.sigma*np.identity(np.shape(self.X)[0]))
        self.alpha_y = self.alpha.dot(self.y)


    def predict(self, Xt):

        """
        Xt (no. of samples, no. of features)

        """

        mean = self.K_X1_X(self.hyp , self.kernel, Xt, self.X).dot(self.alpha_y).reshape(-1,)
        cov = self.K_X1_X1(self.hyp, self.kernel, Xt, Xt) - self.K_X1_X(self.hyp, self.kernel, Xt, self.X).dot(self.alpha).dot(self.K_X_X1(self.hyp, self.kernel, self.X, Xt))

        return mean, cov



    # def plot_mean_variance(self, Xt, sample_size):

    #     mean, cov = self.predict(Xt)

    #     #sampling from the gaussian process
    #     samples = np.random.multivariate_normal(mean.reshape(-1,), cov, size = sample_size)

    #     sample_mean = np.mean(samples, axis = 0)
    #     sample_std = np.std(samples, axis = 0)

    #     fig, ax = plt.subplots()
    #     ax.plot(Xt.reshape(-1,), mean, color = 'red')
    #     ax.scatter(self.X,self.y)
    #     ax.fill_between(Xt.reshape(-1,), mean + 1*sample_std, mean - 1*sample_std, color = 'cyan', alpha = 0.4)
    #     plt.axis('tight')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.legend(['mean estimate of testing data', 'training data'])
    #     plt.title(title_string)
    #     plt.show()


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
    train_c1 = train_c1[1:500,:]
    train_c2 = train_c2[1:500,:]

    new_data = np.concatenate((train_c1,train_c2), axis = 0)

    new_data[new_data[:,4] == 1004,4] =0
    new_data[new_data[:,4] == 1400,4] =1

    test_c1 = test[test[:,4] == 1004,:]
    test_c2 = test[test[:,4] == 1400,:]
    test_c1 = test_c1[1:500,:]
    test_c2 = test_c2[1:500,:]


    new_test = np.concatenate((test_c1, test_c2), axis = 0)
    new_test[new_test[:,4] == 1004,4] =0
    new_test[new_test[:,4] == 1400,4] =1


    #############################################################################################
    #Converting data into X and y, Xtest and ytest

    X = new_data[:,[0,1,2,3,5,6,7,8,9,10,11,12,13,14]]
    y = new_data[:,4]


    Xt = new_test[:,[0,1,2,3,5,6,7,8,9,10,11,12,13,14]]
    yt = new_test[:,4]

    colors = ['green','yellow']
    labels = [0,1]

    #Plotting testing original
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection = '3d')
    

    for index, label in enumerate(labels):

        test_labels = new_test[(yt.reshape(-1,) == label)]
        ax1.scatter(test_labels[:,0], test_labels[:,1], test_labels[:,2], c = colors[index])

    plt.show()

    #Hyperparameters of Gaussian Process Regression
    hyp = [8, 15]
    kernel = 'rbf'
    sigma = 1.3

    print('here')
    gpr = GaussianProcessRegressor(hyp, kernel, sigma)

    t1 = time.time()
    gpr.fit(X,y)
    t2 = time.time()

    print('training time: {}'.format(t2 - t1))
    t3 = time.time()
    mean, cov = gpr.predict(Xt)
    t4 = time.time()

    print('prediction time: {}'.format(t3 - t4))

    predict = predict_class(mean).reshape(-1,)
    
    print('Error: {}'.format(len(predict[predict!=yt])/len(predict)))

    #Plotting the 3d point cloud
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection = '3d')
    
    #Plotting testing
    for index, label in enumerate(labels):

        test_labels = new_test[(predict.reshape(-1,) == label)]
        ax2.scatter(test_labels[:,0], test_labels[:,1], test_labels[:,2], c = colors[index])

    plt.show()
if __name__ == '__main__':
    main()


    



# def plot_mean_variance(kernel = rbf, l = 1, title_string = 'Kernel = rbf, l = 1'):
    
#     mean_1_rbf = (K_X1_X(hyp , kernel).dot()).reshape(-1,)
#     cov_1_rbf = K_X1_X1(hyp, kernel) - K_X1_X(hyp,kernel).dot(np.linalg.inv(K_X_X(hyp,kernel) + sigma*sigma*np.identity(3))).dot(K_X_X1(hyp, kernel))

#     #sampling from the gaussian process
#     samples = np.random.multivariate_normal(mean_1_rbf.reshape(-1,), cov_1_rbf, size = 10000)

#     sample_mean = np.mean(samples, axis = 0)
#     sample_std = np.std(samples, axis = 0)

#     fig, ax = plt.subplots()
#     ax.plot(X1.reshape(-1,), mean_1_rbf, color = 'red')
#     ax.scatter(X,y)
#     ax.fill_between(X1.reshape(-1,), mean_1_rbf + 1*sample_std, mean_1_rbf - 1*sample_std, color = 'cyan', alpha = 0.4)
#     plt.axis('tight')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend(['mean estimate of testing data', 'training data'])
#     plt.title(title_string)
#     plt.show()


# print(np.shape(data[data[:,4] == 1004,:]))
# print(np.shape(data[data[:,4] == 1100,:]))
# print(np.shape(data[data[:,4] == 1103,:]))
# print(np.shape(data[data[:,4] == 1200,:]))
# print(np.shape(data[data[:,4] == 1400,:]))

# data[data[:,4] == 1200,4] = 1
# data[(data[:,4] == 1004)|(data[:,4] == 1100)|(data[:,4] == 1103)|(data[:,4] == 1400),4] = 0

# print(np.shape(data[data[:,4]==1]))
# print(np.shape(data[data[:,4]==0]))
# #Combining and shuffling
# data_combined = np.concatenate((data1, data2), axis=0)
# np.random.shuffle(data_combined)

# data = data_combined[0:int(70/100*len(data_combined)),:]

# test = data_combined[int(70/100*len(data_combined) + 1):,:]


# ###########################################################################################
# print("Training.............")

# labels = [1004, 1100, 1103, 1200, 1400]

# def calc_label(y_o, i):
#     y = y_o.copy()
#     y[y == labels[i]] = 1
#     y[y!= 1] = 0
#     return y


# #Create a vector of classifiers.. one for each class 
# w_t = []
# for i in range(5):
# 	w = []
# 	for j in range(10):
# 		w.append(uniform(-1,1))
# 	w_t.append(w)

# w_t = np.array(w_t)

# # print('Initial Weights{}'.format(w_t))
# #Initialize classifiers, labels and step_size
# print("Hyper parameters.............")
# alpha = 0.001 #Step_size
# num_class = len(labels)
# num_pass = 10
# p = 0
# lamb = 0.0032

# print("alpha: {}".format(alpha))
# print("num of passes: {}".format(num_pass))
# print("lambda: {}".format(lamb))

# label_conv = {0:1004, 1:1100, 2:1103, 3:1200, 4:1400}
# #Dictionary to store the weights
# w = {} 
# w[0] = np.zeros((len(data)*num_pass,10))
# w[1] = np.zeros((len(data)*num_pass,10))
# w[2] = np.zeros((len(data)*num_pass,10))
# w[3] = np.zeros((len(data)*num_pass,10))
# w[4] = np.zeros((len(data)*num_pass,10))

# w[0][0,:] = w_t[0,:]
# w[1][0,:] = w_t[1,:]
# w[2][0,:] = w_t[2,:]
# w[3][0,:] = w_t[3,:]
# w[4][0,:] = w_t[4,:]

# loss = []

# cross_entropy_train = []
# while(p < num_pass): # Infinite passes over the dataset --> to mimic online learning with continuous stream of data
#     """
#     Shuffle data..        
#     """
#     np.random.shuffle(data)
#     #Extract X and Y from the shuffled data
#     y_original = data[:,4].copy()
#     x = data[:,5:]
#     Y = np.zeros((len(data), num_class))

#     for c in range(num_class):
#         Y[:,c] = calc_label(y_original, c)

#     predict_train = []
#     for j in range(0,len(data)): #Passing each data one at at time.. like online learning with sequential data streaming
        
#         l = []
#         hypothesis = []
#         for i in range(0, num_class): #Testing Each classifier
#             """
#             --> one v/s all approach
#             For each classifier, make label corresponding to it as 1 and rest as 0 
#             Predict based on maximum score
#             Score = (w.T)*x
#             """
            
#             #Calculating the best loss by averaging over the weights till now
#             l.append(0.5*(np.sum(w_t[i,:].dot(x[j,:]) - Y[j,i])**2) + 0.5*lamb*np.sum(np.square(w_t[i,:])))

#             #Updating weights
#             w_t[i,:] = w_t[i,:] - alpha*((w_t[i,:].reshape(1,-1).dot(x[j,:].reshape(-1,1)) - Y[j,i])*x[j,:] + 2*lamb*w_t[i,:])

#             #Storing the weights
#             w[i][j + len(data)*p,:] = w_t[i,:]

#             hypothesis.append(w_t[i,:].dot(x[j,:]))

#         predict_train.append(label_conv[np.argmax(hypothesis)])

#         loss.append(l)

#     predict_train =  np.array(predict_train)
#     cross_entropy_train.append((len(predict_train[predict_train != data[:,4]]))/len(predict_train))
    
#     print('current pass: {}'.format(p))
#     p = p + 1

# ####################################################################################################################################
# #Plotting training cross entropy pass wise
# print('Plotting training cross entropy pass wise')
# f,ax = plt.subplots()
# ax.plot(range(num_pass), cross_entropy_train)
# plt.ylabel('Cross Entropy training error pass wise')
# plt.xlabel('Iterations')
# plt.title('Error')
# #Weights after training
# # for c in range(0, num_class):
# # 	print('Average Weights after training for classifier {}: {}'.format(c, np.mean(w[c], axis = 0)))

# #####################################################################################################################################
# #TESTING
# print("Testing Calculations.............")
# #Try on testing data....... (Calculating test loss)
# #Testing data
# y_test = test[:,4]
# x = test[:,5:]

# #Encoding
# y_encoded_test = np.zeros((len(test), num_class))
# for c in range(num_class):
#     y_encoded_test[:,c] = calc_label(y_test, c)

# #Variables
# test_loss = []
# avg_weights = {}
# avg_weights[0] = np.mean(w[0], axis =0)
# avg_weights[1] = np.mean(w[1], axis =0)
# avg_weights[2] = np.mean(w[2], axis= 0)
# avg_weights[3] = np.mean(w[3], axis= 0)
# avg_weights[4] = np.mean(w[4], axis= 0)
# predict = []

# #Calculating predicted values, cross entropy error and loss (batch wise)
# batch = 100
# index =0
# while(index+batch < len(test)):
# 	l = np.zeros(num_class)
# 	hypothesis = np.zeros((batch,num_class))
# 	for c in range(num_class):

# 		l[c] = (1/batch)*0.5*np.sum((avg_weights[c].dot(x[index:index+batch,:].T).T - y_encoded_test[index:index+batch,c])**2)
# 		hypothesis[:,c] = avg_weights[c].dot(x[index:index+batch,:].T).T
# 	predict.extend([label_conv[x] for x in np.argmax(hypothesis, axis = 1)])
# 	test_loss.append(l)

# 	index+=batch

# if(index < len(test)):
# 	batch = len(test)-index
# 	l = np.zeros(num_class)
# 	hypothesis = np.zeros((batch, num_class))
# 	for c in range(num_class):
# 		l[c] = (1/batch)*0.5*np.sum((avg_weights[c].dot(x[index:index+batch,:].T).T - y_encoded_test[index:index+batch,c])**2)
# 		hypothesis[:,c] = avg_weights[c].dot(x[index:index+batch,:].T).T
# 	predict.extend([label_conv[x] for x in np.argmax(hypothesis, axis = 1)])
# 	test_loss.append(l)	


# predict = np.array(predict)
# cross_entropy_error = (len(predict[predict != y_test]))/len(predict)
# print("Cross entropy testing error : {}".format(cross_entropy_error))	
# test_loss = np.array(test_loss)


# #Plotting testing loss for each classifier
# label_ = ['classifier1', 'classifier2', 'classifier3', 'classifier4', 'classifier5']
# f,ax = plt.subplots()
# for c in range(0,5):
# 	ax.plot(range(len(test_loss[:,c])), test_loss[:,c], label = label_[c])
# plt.legend()
# plt.title('Testing loss (with average weights) vs iterations')
# plt.ylabel('Testing loss')
# plt.xlabel('Iterations')
# plt.show()

# #Testing 3D cloud
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# colors = ['green','black','cyan', 'brown','yellow']

# for index, label in enumerate(labels):

#     test_label = test[(predict.reshape(-1,) == label) ]
#     ax.scatter(test_label[:,0], test_label[:,1], test_label[:,2], c = colors[index])

# plt.show()



# ################################################################################################################
# #TRAINING
# print("Training loss calculations.............")
# #Training loss and point cloud
# y_train = data[:,4]
# x_train = data[:,5:]

# #Encoding
# y_encoded_train = np.zeros((len(data), num_class))
# for c in range(num_class):
#     y_encoded_train[:,c] = calc_label(y_train, c)

# #Variables
# train_loss = []
# train_predict = []

# #Calculating predicted values, cross entropy error and loss for training data
# batch = 100
# index =0
# while(index+batch < len(data)):
# 	l = np.zeros(num_class)
# 	hypothesis = np.zeros((batch,num_class))
# 	for c in range(num_class):

# 		l[c] = (1/batch)*0.5*np.sum((avg_weights[c].dot(x_train[index:index+batch,:].T).T - y_encoded_train[index:index+batch,c])**2)
# 		hypothesis[:,c] = avg_weights[c].dot(x_train[index:index+batch,:].T).T
# 	train_predict.extend([label_conv[x] for x in np.argmax(hypothesis, axis = 1)])
# 	train_loss.append(l)

# 	index+=batch

# if(index < len(data)):
# 	batch = len(data)-index
# 	l = np.zeros(num_class)
# 	hypothesis = np.zeros((batch, num_class))
# 	for c in range(num_class):
# 		l[c] = (1/batch)*0.5*np.sum((avg_weights[c].dot(x_train[index:index+batch,:].T).T - y_encoded_train[index:index+batch,c])**2)
# 		hypothesis[:,c] = avg_weights[c].dot(x_train[index:index+batch,:].T).T
# 	train_predict.extend([label_conv[x] for x in np.argmax(hypothesis, axis = 1)])
# 	train_loss.append(l)	


# train_predict = np.array(train_predict)
# cross_entropy_error = (len(train_predict[train_predict != y_train]))/len(train_predict)
# print("Cross entropy training error : {}".format(cross_entropy_error))	
# train_loss = np.array(train_loss)


# #Plotting training loss for each classifier
# label_ = ['classifier1', 'classifier2', 'classifier3', 'classifier4', 'classifier5']
# f,ax = plt.subplots()
# for c in range(0,5):
# 	ax.plot(range(len(train_loss[:,c])), train_loss[:,c], label = label_[c])
# plt.legend()
# plt.title('Training loss (with average weights) vs iterations')
# plt.ylabel('Training loss')
# plt.xlabel('Iterations')
# plt.show()

# #Training 3D cloud
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# colors = ['green','black','cyan', 'brown','yellow']

# for index, label in enumerate(labels):

#     train_label = data[(train_predict.reshape(-1,) == label) ]
#     ax.scatter(train_label[:,0], train_label[:,1], train_label[:,2], c = colors[index])

# plt.show()

