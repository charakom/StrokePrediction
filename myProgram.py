#           IMPORTS
#Making the imports of the libraries I am going to need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import plotly.express as px
import imblearn

#           INITIAL DATA ANALYSIS
#IMPORT the data as dataframe from the csv file

dataset = pd.read_csv('C:/Users/xarak/Documents/StrokePrediction/healthcare-dataset-stroke-data.csv')

#GET MORE INFO FOR THE DATASET
print("This is how the dataset looks like :", dataset.head())

#Get info on the type of data AND description of basic statistsics
print(dataset.info())
print(dataset.describe())

#Get the size of the dataset
print("The dataset size is : ", dataset.shape)

# After having the size of the data, see how many distinct ids I have
print(dataset['id'].nunique())

#Check for duplicate  rows:
print("There are ", dataset.duplicated().sum(), " duplicate rows.")


#          SEPARATE CATEGORICAL FROM NUMERICAL DATA FOR FURTHER ANALYSIS
#Create a Categorical values dataframe
categorical = dataset.select_dtypes(include=['object']).columns.tolist()

#Create a Numerical values dataframe
numerical = dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical.remove('stroke')


print('categorical variables:', categorical)
print('numerical variables:', numerical)

# Run through the categorical data and get the distinct values
for i in categorical:
    print(dataset[i].value_counts().to_frame(), '\n')

#From the above analysis we can see that only one person identified as Other
#in the gender column and thus we can count it as an outlier.
#I might want to remove this entry from the dataset as its not adding any value. Remember
#that this analysis contains missing values and possibly empty cells.

# Display the statistical overview of the categorical data
print(dataset.describe(include=[np.object]))

#Print a describe to see some statistics for the numerical data !!
print(dataset.describe())




#                         HANDLING MISSING VALUES

#DROPPING UNWANTED COLUMNS
# Drop the ID column as its not needed
dataset = dataset.drop('id', axis=1)

#Sum up all the MISSING values from each column
print("Table with the missing values of each feature: ", '\n', dataset.isna().sum())

#Replace UNKNOWN with nan which I located from the previous table
#The Unknown values of 'smoking_status' feature I could retrieve from the categorical data distinct values result.
dataset['smoking_status'].replace('Unknown', np.nan, inplace=True)

#The 'bmi' missing values where located from the  comand above "dataset.isna().sum()"
dataset['bmi'].replace('N/A', np.nan, inplace=True)

#Count unique values of the feature gender VS Count and display the unique values of feature gender
# print("Number of unique gender values are: ", dataset['gender'].nunique())

#Drop the gender row with the outlier
dataset = dataset[dataset['gender'] != 'Other']

#Replace the nan values with the mean of each column
dataset['bmi'].fillna(dataset['bmi'].mean(), inplace=True)


# The nan values for 'smoking_status' is huge 1544 I don't know if it's good practice to replace the nan with the dominant value.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dataset['smoking_status'].fillna(dataset['smoking_status'].mode()[0], inplace=True)

# This is to show that after the handling I have NO missing/Unknown data
print(dataset.isna().sum())


#           EXPLORATORY DATA ANALYSIS
dataset.info()

#Create two datasets Stroke False and True for people who hadn't and had strokes respectively

stroke_False = dataset[dataset['stroke'] == 0]
stroke_True = dataset[(dataset['stroke'] == 1)]

print("People who have not had a stroke in percentage", format(len(stroke_False)/len(dataset)* 100, '.2f'), '%')
print("People who have had a stroke in percentage", format(len(stroke_True)/len(dataset)* 100, '.2f'), '%')

# We can see that from the dataset 95.13 % haven't had a stroke and 4.87% have had stroke. The dataset is not balanced and we should consider
#ways of sampling to improve the skewness of the dataset.




#                  VISUALIZIONS

#Correlation heatmap
plt.figure(figsize=(16, 6))
mask = np.triu(np.ones_like(dataset.corr(), dtype=np.bool))
heatmap = sns.heatmap(dataset.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16)
plt.show()

#From the correlation heatmap we can see if there is high or low correlation between the features.
#A strange result is that the correlation between bmi and heart_disease is very low on 3%.
#This is considered a strange result which we should further investigate since usually obesse people have heart
#as their age progress.

# We can further deduce that age is highly correlated to all the features.
# The highest correlation is 33% and it's between bmi and age.
#Age is a critical feature and one that has direct connection with every other feature on our dataset
# Apart from that the highest correlation is between bmi and hypertension. Which is at 16%.


#Create a list of continuous numerical data
continuous_numerical = ['age','avg_glucose_level','bmi']
# Plot the distribution of clinical patient continuous numerical data: age, average glucose level and BMI.

# Set up the matplotlib figure
f, axes = plt.subplots(ncols=3, figsize=(17, 6))

#Numerical Features Distribution
count = 0
color = ['m','y','g']
for feature in continuous_numerical:
    sns.histplot(dataset[feature], kde=True, color=color[count], ax=axes[count]).set_title(feature + ' distribution')
    axes[count].set_ylabel('Patient Count')
    count = count + 1
plt.show()


# Create plots to display the relation of categorical data having a stroke
for feature in categorical:
    sns.countplot(x=feature, hue='stroke', data=dataset,  palette=["#7fcdbb", "#edf8b1"])
    plt.title(feature + " - stroke frequency")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

# From these 3  charts we can see the count of the binary features in relation to stroke.
binary = ['hypertension', 'ever_married','gender','heart_disease', ]
for i in binary:
    sns.countplot(x=i , hue= 'stroke', data= dataset )
    plt.title("Stroke - " + i + " Count")
    plt.xlabel(i)
    plt.ylabel('Count')
    plt.show()

#Create count plots only for the proportion of people that had a stroke. Note that the imbalance of the dataset its very
#obvious since in  many cases we would have expected completely different results on the plots.
for i in binary + ['Residence_type'] :
    sns.countplot(x=i , data= stroke_True , color='r' )
    plt.title(i + " on people who suffered stroke ")
    plt.xlabel(i)
    plt.ylabel('Count')
    plt.show()


# A kernel density estimate (KDE) plot is a method for visualizing the distribution of observations in a dataset,
# analagous to a histogram. KDE represents the data using a continuous probability density curve in one or more dimensions.

# Set up the figure
f, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")

# Draw a contour plot to represent each bivariate density
sns.kdeplot(
    data=dataset, x="avg_glucose_level",
    y="age",
    hue="stroke")
plt.show()

# On this plot we can visually infer that there seems to be that there are 2 clusters
# for people who have had a stroke. One is bigger than the other and seems to include people
# of age more than 60 and of within normal glucose levels. And the second group shows
# that people of age with very high glucose levels have had a stroke.


# # This plot does not provide any value
# # Set up the figure
# f, ax = plt.subplots(figsize=(8, 8))
# ax.set_aspect("equal")
#
# # Draw a contour plot to represent each bivariate density
# sns.kdeplot(
#     data=dataset, x="age",
#     y="bmi",
#     hue="stroke")
# plt.show()


########################SCALING THE DATA##############################

## #Label encoder to NORMALISE the data ( remove decimal points, make values binary from T-F .etc)
from sklearn.preprocessing import LabelEncoder
def label_encoded(feat):
    le = LabelEncoder()
    le.fit(feat)
    print(feat.name,le.classes_)
    return le.transform(feat)


for col in dataset.columns:
    dataset[str(col)] = label_encoded(dataset[str(col)])

print(dataset)


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

############SMOTE########
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

############### DECISION TREE CLASSIFIER ################
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()   #Instantiate an object out of our class
decision_tree.fit(X_res,y_res)

from sklearn.metrics import classification_report , confusion_matrix
from sklearn.metrics import accuracy_score
y_predict_test = decision_tree.predict(X_test)
print(y_predict_test)
print(y_test)
cm = confusion_matrix(y_test, y_predict_test)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('True', fontsize=20)
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticklabels(['Did not suffer stroke','suffered stroke'], fontsize = 10)
ax.xaxis.tick_top()

ax.set_ylabel('Predicted', fontsize=20)
ax.yaxis.set_ticklabels(['Did not suffer stroke', 'suffered stroke'], fontsize = 10)
plt.show()
print(classification_report(y_test, y_predict_test))
print('Accuracy score is: ' , accuracy_score(y_test,y_predict_test))



#####################################################
print("___________________RandomForestClassifier_____________________")
from sklearn.ensemble import RandomForestClassifier
RandomForest = RandomForestClassifier(n_estimators = 150)
RandomForest.fit(X_res, y_res)
y_predict_test = RandomForest.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('True', fontsize=20)
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticklabels(['NO stroke','Had stroke'], fontsize = 10)
ax.xaxis.tick_top()

ax.set_ylabel('Predicted', fontsize=20)
ax.yaxis.set_ticklabels(['NO stroke', 'Had stroke'], fontsize = 10)
plt.show()

print(classification_report(y_test, y_predict_test))
print('Accuracy score is: ' , accuracy_score(y_test,y_predict_test))


print("___________________LogisticRegression_________________________")
from sklearn.linear_model import LogisticRegression
LogisticRegressionclf = LogisticRegression(random_state=0, max_iter = 400)
LogisticRegressionclf.fit(X_res,y_res)
y_predict_test = LogisticRegressionclf.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)

print(classification_report(y_test, y_predict_test))
print('Accuracy score is: ' , accuracy_score(y_test,y_predict_test))




import scipy.stats
scipy.stats.pearsonr(dataset['smoking_status'],dataset['stroke'],)    # Pearson's r




















