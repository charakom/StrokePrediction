#Making the imports of the libraries I am going to need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

#Import and read the data
dataset = pd.read_csv('C:/Users/xarak/Documents/StrokePrediction/healthcare-dataset-stroke-data.csv')

#Drop the first column which is the id and I don't need. Replace missing values with NaN
dataset = dataset.drop('id', axis=1)
dataset['smoking_status'].replace('Unknown', np.nan, inplace=True)
dataset['bmi'].replace('N/A', np.nan, inplace=True)


#Replacing the nan values with the mean of each column
dataset['bmi'].fillna(dataset['bmi'].mean(), inplace= True)
dataset['smoking_status'].fillna(dataset['smoking_status'].mode()[0], inplace=True)
print(dataset.isna().sum())


#Label encoder to NORMALISE the data ( remove decimal points, make values binary from T-F .etc)
from sklearn.preprocessing import LabelEncoder
def label_encoded(feat):
    le = LabelEncoder()
    le.fit(feat)
    print(feat.name,le.classes_)
    return le.transform(feat)


for col in dataset.columns:
    dataset[str(col)] = label_encoded(dataset[str(col)])

print(dataset)


#Create two datasets Stroke False and True for people who hadn't and had respectively strokes

stroke_False = dataset[dataset['stroke'] == 0]
stroke_True = dataset[(dataset['stroke'] == 1)]

print("People who have not had a stroke in percentage", len(stroke_False)/len(dataset)* 100, '%')
print("People who have had a stroke in percentage", len(stroke_True)/len(dataset)* 100, '%')

#Correlation figure
plt.figure(figsize=(10,10))
sns.heatmap(dataset.corr(), vmin=-1, cmap='coolwarm', annot=True)
#plt.show()

#VISUALIZE
#sns.pairplot(dataset, hue = 'stroke', vars=['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke'])
#plt.show()



#Create the input dataset and the target class with the output data
X = dataset.drop(['stroke'], axis =1)
y = dataset['stroke']

#Split Training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

print("----------------------------------------------")

#Gives the number of rows and columns
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)