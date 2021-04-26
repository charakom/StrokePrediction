#           IMPORTS
#Making the imports of the libraries I am going to need
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import dexplot as dxp

#           INITIAL DATA ANALYSIS
#IMPORT the data as dataframe from the csv file
dataset = pd.read_csv('C:/Users/xarak/Documents/StrokePrediction/healthcare-dataset-stroke-data.csv')

#GET MORE INFO FOR THE DATASET
print("This is how the dataset looks like :", dataset.head())

#Get info on the type of data AND description of basic statistsics
print(dataset.info())
print(dataset.describe())

#Get the size of the dataset.
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
#Print a describe to see some statistics for the numerical data !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! EXPLAIN MORE!
print(dataset.describe())




#                         HANDLING MISSING VALUES

#DROPPING UNWANTED COLUMNS
#Drop the ID column as its not needed
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
dataset= dataset[dataset['gender'] != 'Other']

#Replace the nan values with the mean of each column
dataset['bmi'].fillna(dataset['bmi'].mean(), inplace=True)

# The nan values for 'smoking_status' is huge 1544 I don't know if it's good practice to replace the nan with the dominant value?????????????????????????????
#????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
dataset['smoking_status'].fillna(dataset['smoking_status'].mode()[0], inplace=True)

# This is to show that after the handling I have NO missing/Unknown data
print(dataset.isna().sum())


#           EXPLORATORY DATA ANALYSIS



#Relation of Gender and Stroke
dataset.groupby('gender', 'stroke_status').size().plot(kind ='bar')







# #Create two datasets Stroke False and True for people who hadn't and had respectively strokes
#
# stroke_False = dataset[dataset['stroke'] == 0]
# stroke_True = dataset[(dataset['stroke'] == 1)]
#
# print("People who have not had a stroke in percentage", len(stroke_False)/len(dataset)* 100, '%')
# print("People who have had a stroke in percentage", len(stroke_True)/len(dataset)* 100, '%')
#
# #Correlation figure
# plt.figure(figsize=(10,10))
# sns.heatmap(dataset.corr(), vmin=-1, cmap='coolwarm', annot=True)
# #plt.show()
#
# #VISUALIZE
# #sns.pairplot(dataset, hue = 'stroke', vars=['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke'])
# #plt.show()
#



## #Label encoder to NORMALISE the data ( remove decimal points, make values binary from T-F .etc)
# from sklearn.preprocessing import LabelEncoder
# def label_encoded(feat):
#     le = LabelEncoder()
#     le.fit(feat)
#     print(feat.name,le.classes_)
#     return le.transform(feat)
#
#
# for col in dataset.columns:
#     dataset[str(col)] = label_encoded(dataset[str(col)])
#
# print(dataset)
#



# #Create the input dataset and the target class with the output data
# X = dataset.drop(['stroke'], axis =1)
# y = dataset['stroke']
#
# #Split Training and testing data
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
#
# print("----------------------------------------------")
#
# #Gives the number of rows and columns
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)