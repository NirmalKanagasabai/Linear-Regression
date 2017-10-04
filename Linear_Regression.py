### The core logic of Linear Regression is implemented through 3 classes:
### Class Runner(): This class takes in the input, initializes, Dataset_Splits the test and train datsets, computes mean square error, validates the data and predicts the result
### Class Learn():

### Pandas are imported to perform Read_CSV and DataFrame operations!
import pandas as pd
### Numpy is used to perform the random permutation operations in the Split_Data method in Run class!
### Numpy is predominantly used for carrying out matrix operations (implementing the LR logic!)
import numpy as np
### ArgParse, as the name states, is the argument parser. It is used to parse the input provided by the user at run_time!
import argparse

### Global Name Definitions (I/O files)

### Input file (after data shaping and cleaning)
Regression_Input = 'Regression_Input.CSV'
### Output file (which contains the predictions of the final time that would be taken up by the participants!)
Regression_Output = 'Regression_Output.csv'


class linearRegression():

	def __init__(self, X = [0], y = 0, alpha = 0.07, models = 1, NoOfIterations = 5000, W = 0, regular = 0.02):

		""" This method performs the initialization for linear regression through Gradient Descent...
		'NoOfIterations' - Number of Iterations
		'regular' - Regularization (Lambda Value) """

		self.alpha = alpha
		self.NoOfIterations = NoOfIterations
		self.regular = regular

	def loss_factor(self):

		""" This method computes the loss factor (to compute cost) in Gradient Descent"""

		wTx = np.dot(self.X,np.transpose(self.W))
		difference = wTx - self.y
		loss_factor = (1. / (2 * self.m)) * (difference ** 2)
		return np.sum(loss_factor)

	def gradient_descent(self):

		""" This method implements the core logic for Gradient Descent"""

		### For each iteration, compute 'w', 'w-transpose-x', differenceerence, weight and cost(loss_factor)

		for i in range(0, self.NoOfIterations):

			w = self.W
			wTx = np.dot(self.X, np.transpose(self.W))

			difference = wTx - self.y

			self.W = w - self.alpha * (1. / self.m) * difference.dot(self.X)
			cost = self.loss_factor()
			print "Cost: ", cost

	def Normalization(self,X):

		""" This method performs feature normalization"""

		Xcopy = X.copy()
		Xcopy = np.asarray(Xcopy)

		mean = np.mean(Xcopy, axis=0)

		std = np.std(Xcopy, axis=0)
		std[std == 0.0] = 1.0

		Xr = np.rollaxis(Xcopy, 0)
		Xr = Xr - mean
		Xr /= std

		return Xr

	def fit_oprn(self,X,y):

		""" This method is used to perform the 'fitting' operation """

		self.m, self.n = X.shape
		Xn = X.copy()
		Xn = self.Normalization(Xn)
		ones = np.array([1] * self.m).reshape(self.m, 1)
		self.X = np.append(ones,Xn,axis=1)
		self.y = y
		self.W = np.array([0.0] * (self.n + 1))
		self.gradient_descent()

	def predict(self, X):

		""" This method implements the prediction logic"""

		X = np.array(X)
		m = X.shape[0]

		Xn = X.copy()
		Xn = self.Normalization(Xn)

		ones = np.array([1] * m).reshape(m, 1)

		Xr = np.append(ones,Xn,axis=1)
		p = np.dot(Xr,np.transpose(self.W))

		return p



### This class, as already stated, implements the logic of K-fold Cross Validation

class Cross_Validate:

	def cross_validate(self, cross , X, y, cv = 1):

		""" This method implements the logic of K-fold cross validation"""

		scores = []
		preds = []

		splitX = np.split(np.array(X), cv, axis=0)
		splitY = np.split(np.array(y), cv, axis=0)

		for i in range(cv):

			XSplit = splitX[:]
			YSplit = splitY[:]

			### Testing and Training values of X
			TestX = XSplit.pop(i)
			TrainX = np.concatenate(XSplit, axis=0)

			### Testing and Training Values of Y
			TestY = YSplit.pop(i)
			TrainY = np.concatenate(YSplit, axis=0)

			### Performing the Cross Fit on the Training Dataset
			cross.fit_oprn(TrainX,TrainY)

			predicted_val_y = cross.predict(TestX)
			preds.extend(predicted_val_y.tolist())

			if isinstance(cross,linearRegression):
				score = ((predicted_val_y - TestY) ** 2).mean()
				print "Iteration ", i, "Mean_Squared_Error :",score

			else:
				score =  np.mean(TestY == predicted_val_y)
				print "Iteration ", i, "Accuracy :",score

			scores.append(score)
		return scores,preds


class Runner():

	"""The Runner Class for Linear Regression!"""

	def __init__(self, Regression_Input,output_file):

		""" Initialization method for the 'Run' class which takes in input, validates, predicts and provides the output!"""

		self.RegInput = pd.read_csv(Regression_Input)
		self.output = output_file
		print "***** Linear Regression *****"
		self.validation()
		self.prediction()

	def Split_Data(self, X, y, per = 0.7):

		""" This method splits the data set for testing and training purposes!
		It returns the Testing and Training datasets of X and Y"""

		### Can be changed however we want! As there was a huge dataset, we set it to 23000!
		Dataset_Split = 23000

		index = np.random.permutation(X.shape[0])

		idxTrain, idxTest = index[:Dataset_Split], index[Dataset_Split:]

		TrainX,TestX = X[idxTrain], X[idxTest]
		TrainY,TestY = y[idxTrain], y[idxTest]

		return TrainX,TestX,TrainY,TestY

	def Mean_Squared_Error(self, predicted_val_y, actual_val_y):

		""" This method computes the Mean Squared Error for Regression """

		return ((predicted_val_y - actual_val_y) ** 2).mean()

	def Regression_Data(self, stype='train'):

		""" Pass X,y needed for regression """
		X = []
		y = []

		reg = self.RegInput

		if stype == 'train':

			### Enlisting the columns that will be taken up!
			### Column name and this argument must be the same!
			X = reg[['Participant_ID','Age','Gender','AvgTime2003','AvgTime2004','AvgTime2005','AvgTime2006','AvgTime2007','AvgTime2008','AvgTime2009','AvgTime2010','AvgTime2011','AvgTime2012','AvgTime2013','AvgTime2014','AvgTime2015','AvgTimeforAllMarathons','TotalNoofRaces']]
			y = reg.AvgTime2015

		else:
			X = reg[['Participant_ID','Age','Gender','AvgTime2003','AvgTime2004','AvgTime2005','AvgTime2006','AvgTime2007','AvgTime2008','AvgTime2009','AvgTime2010','AvgTime2011','AvgTime2012','AvgTime2013','AvgTime2014','AvgTime2015','AvgTimeforAllMarathons','TotalNoofRaces']]

		return X,y

	def validation(self):

		""" This method validates the algorithms for both test and train datasets """

		X,y = self.Regression_Data()
		X = X.as_matrix()
		y = y.as_matrix()

		TrainX,TestX,TrainY,TestY = self.Split_Data(X,y)

		### The number of iterations has been set to 25000
		lr = linearRegression(NoOfIterations= 25000)

		lr.fit_oprn(TrainX,TrainY)
		y_p2 = lr.predict(TestX)

		print 'Performing Cross Validation: '
		### The '5' is set as default!

		scores,pred = Cross_Validate().cross_validate(lr, TrainX, TrainY, 5)

		print '\n Average Mean_Squared_Error (across iterations):', self.Mean_Squared_Error(pred, TrainY)
		print '\n Mean Squared Error:', self.Mean_Squared_Error(y_p2, TestY)

		self.linear = lr

	def prediction(self):

		""" This method performs the final prediction for Miami Marathon """

		### RegData is the data frame which captures the final predictions for this Marathon
		RegData = pd.DataFrame(columns = ['Marathon_Time'])
		X,y = self.Regression_Data(stype = 'predict')

		print "\n Predicting 'Finishing_Time' using Linear Regression \n"

		### Linear Regression (Hypothesis) 'y' given 'x'
		lry = self.linear.predict(X)
		lry = lry
		finish_time = []

		### Conversion of the seconds to 'hh:mm:ss' format
		for seconds in lry:
			mins, secs = divmod(seconds, 60)
			hours, mins = divmod(mins, 60)
			time = "%d:%02d:%02d" % (hours, mins, secs)
			finish_time.append(time)

		### Final Predictions
		print 'No. of Rows Predicted:',len(finish_time)
		RegData.Marathon_Time = finish_time
		### Writing the data to the output file
		RegData.to_csv(self.output)


if __name__ == '__main__':
	""" This is the main method of the file
	It initializes the parser!
	The parser uses ArgParse method to parse the input provided by the user!"""

	parse = argparse.ArgumentParser()
	### Parsing the Input File
	parse.add_argument('-i','--Input',type=str, help='Input File',required=True)
	### Parsing the Output File
	parse.add_argument('-o','--Output',type=str, help='Output File',required=True)
	args = parse.parse_args()

### Calling the run method with the input and output file as parameters!
Runner(args.Input,args.Output)
