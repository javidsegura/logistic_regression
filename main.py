import numpy as np
import pandas as pd

from sklearn.metrics import (roc_curve, roc_auc_score)



class LogisticRegression:


      """
      
      Note: initially data has to be given transformed. No splitting and no normalization
      
      """


      def __init__(self, X, y,df) -> None:
            self.y = y
            self.df = df
            self.X, self.theta = self.set_up(X)

      def set_up(self,X):
            self.m, self.n = X.shape

            theta = np.zeros(self.n +1) # Adding intercept
            intercept = np.ones((self.m, 1))
            X = np.hstack((intercept, X))

            if len(X[0]) != len(theta):
                  print(f"Dimensions between features and theta do not match!")

            return X,theta
      
      def compute_hypothesis(self):

            z = np.dot(self.X,self.theta)

            hypothesis = 1 / (1 + np.exp(-z))

            return hypothesis


      def gradient_descent(self, alpha, n_iterations):

            X,y,theta = self.X, self.y, self.theta

            for i in range(n_iterations):

                  z = np.dot(np.array(X), np.array(theta))

                  hypothesis = 1 / (1 + np.exp(-z))

                  error = hypothesis -y

                  gradient = 1 / self.m * np.dot(X.T, error)

                  theta -= alpha * gradient
                  

            self.theta = theta
            
            return theta
      
      def cost_function(self):

            if sum(self.theta) == 0:
                  return "Parameters have not been fitted"
            else:

                  hypothesis = self.compute_hypothesis()

                  y = self.y

                  cost = -np.mean(y * np.log(hypothesis) + (1-y) * np.log(1-hypothesis))
            
                  return cost
            
      def make_predictions(self, treshold = 0.5):
            
            hypothesis = self.compute_hypothesis()

            predictions = list()

            for i in hypothesis:
                  if i >= treshold: 
                        predictions.append(1)
                  else:
                        predictions.append(0)

            df = self.df

            df["predictions"] = predictions

            true_positive = ((df["diagnosis"] == 1) & (df["predictions"] == 1)).sum()
            false_positive = ((df["diagnosis"] == 0) & (df["predictions"] == 1)).sum()
            true_negative = ((df["diagnosis"] == 0) & (df["predictions"] == 0)).sum()
            false_negative = ((df["diagnosis"] == 1) & (df["predictions"] == 0)).sum()

            sensitivity = true_positive / (true_positive + false_negative)
            specificity = true_negative / (true_negative + false_positive) 
            accuracy = (true_negative + true_positive) / len(predictions)

            metrics = f"Sensitivity: {sensitivity}, Specificity: {specificity}, Accuracy: {accuracy}"


            confusion_matrix_manual = pd.DataFrame({
            'Predicted 0': [true_negative, false_positive],
            'Predicted 1': [false_negative, true_positive]
            }, index=['Actual 0', 'Actual 1'])

            youdend_stat = true_positive + true_negative - 1


            return predictions, confusion_matrix_manual, metrics, youdend_stat
      
      def roc(self):
            true = self.df["diagnosis"].values
            predicted = self.df["predictions"].values

            roc = roc_curve(true,predicted)
            auc = roc_auc_score(true,predicted)

            return roc, auc
      
      def optimal_treshold(self):

            stats = []

            for i in range(1,101):
                  results = self.make_predictions(treshold= i / 100)
                  stats.append(results[-1])

            index = stats.index(max(stats))

            treshold = (index + 1 ) /100
            

            return f"Optimal treshold is {treshold}"



            


    
url = "/Users/javierdominguezsegura/Programming/Python/Algorithms/machine_learning/logistic_regression/data/breast-cancer.csv"

df = pd.read_csv(url)

df.drop(columns=["id"], inplace=True)

df["diagnosis"] = np.where(df["diagnosis"] == "M", 1,0)

names = [i for i in df.columns]

names.remove("diagnosis")

X = df[names].values
y = df["diagnosis"].values



model = LogisticRegression(X,y,df)

print(model.gradient_descent(.001,1000))


prediction = model.make_predictions(.84)

print(*prediction, sep = "\n\n")

print(model.roc())

print(model.optimal_treshold())
