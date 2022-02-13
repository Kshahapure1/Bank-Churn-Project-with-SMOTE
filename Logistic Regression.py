#Import Libraries
import pandas as pd
import numpy as np

#scikit library for macjhine learning
import statsmodels.api as smapi
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

#read the data file
data = pd.read_csv("bankchurn1.csv")

print(data)
print(data.head())
print(data.tail())
data.shape
data.columns
#CHECK THE DATA TYPES
data.dtypes

#Remove unwanted features

data.drop(columns = ["custid","surname"],inplace = True)
data.columns

#Split the columns into numeric and factors 
data.dtypes

nc = data.select_dtypes(exclude = "object").columns.values
fc = data.select_dtypes(include = "object").columns.values
nc
fc

#Check for Distribution of the y - Variable
sns.countplot(x="churn",data=data) 
plt.title("Distribution of the classes")
data.churn.value_counts()


#EDA

#Null

data.info()

#EDA on nc

#ZEROS (0)
data[nc][data[nc]==0].count()

#Check for distribution and outliers and multicolinearity 

# function to plot the histogram, correlation matrix, boxplot based on the chart-type
def plotdata(data,nc,ctype):
    if ctype not in ['h','c','b']:
        msg='Invalid Chart Type specified'
        return(msg)
    
    if ctype=='c':
        cor = data[nc].corr()
        cor = np.tril(cor)
        sns.heatmap(cor,vmin=-1,vmax=1,xticklabels=nc,
                    yticklabels=nc,square=False,annot=True,linewidths=1)
    else:
        COLS = 2
        ROWS = np.ceil(len(nc)/COLS)
        POS = 1
        
        fig = plt.figure() # outer plot
        for c in nc:
            fig.add_subplot(ROWS,COLS,POS)
            if ctype=='b':
                sns.boxplot(data[c],color='yellow')
            else:
                sns.distplot(data[c],bins=20,color='green')
            
            POS+=1
    return(1)

#Check the plotas to identify the outliers and multicolinearity

plotdata(data,nc,"b")
plotdata(data,nc,"c")


#Converting the factor into dummy variables
data_tr = data.copy()

for c in fc:
    dummy = pd.get_dummies(data[c],drop_first =True,prefix=c)
    data_tr = data_tr.join(dummy)


#Check the new column count
data_tr.columns

#Remove the old factor columns
data_tr.drop(columns = fc, inplace = True)
data_tr.columns

data_tr.columns

#Write generic function to splid data, build model, prediction and converting prob to class

#Function: split data
#Input = data,y,ratio
#returns = trainx,trainy,testx,testy
def splitdata(data,y,ratio = 0.3):
    trainx,testx,trainy,testy = train_test_split(data.drop(y,1),data[y],test_size=ratio)
    
    return(trainx,trainy,testx,testy)

trainx1,trainy1,testx1,testy1 = splitdata(data_tr,"churn")

#Function Build model 
#Build the logistic regression model 
#Input = trainx,trainy
#returns = Logit model
def buildModel(trainx,trainy):
    model = smapi.Logit(trainy,trainx).fit()
    return(model)



#Function predictthe class
#Predict the churn and convert the probabbilities into classes 
#input : predicted Probabilities,cut off value
#p1 < 0.5   --- 1
#p1> 0.5 --------0 
def predictClass(probs,cutoff):
    if (0 <= cutoff <=1):
        P = probs.copy()
        P[ P < cutoff ]  = 0
        P[ P > cutoff ]  = 1
        
        return(P.astype(int))
    
#print the  Confusion Matrix
#Input actual y, Predicted y
#eturns - 
def cm(actual,predicted):
    #Methid 1
    print(confusion_matrix(actual,predicted))
    df = pd.DataFrame({"actual":actual,"predicted":predicted })
    print(pd.crosstab(df.actual,df.predicted,margins = True))
    
    #print the classification report
    print(classification_report(actual,predicted))

#-----------------------------------------------------------

#Build model M1
m1 = buildModel(trainx1,trainy1)

##Summarise the model
m1.summary()
p1 = m1.predict(testx1)
print(p1)
cutoff = 0.5
pred_y1 = predictClass(p1,cutoff)
#Confusion matrix
cm(testy1,pred_y1)

#Cross veryfy the classes 0 and 1
len(testy1[testy1 == 0])
len(testy1[testy1 == 1])

data.dtypes
#----------------------------------------------------------------
from sklearn.feature_selection import f_classif

#Build the function for feature selection
#Returns the scores of all the features of train data set
#Input(train(x&y))
def bestFeatures(trainx,trainy):
    features = trainx.columns
    
    fscore,pval = f_classif(trainx,trainy)
    
    df = pd.DataFrame({"feature":features,"fscore":fscore,"pval":pval})
    df = df.sort_values("fscore",ascending = False)
    return(df)

#-------------------------------------------------------------------

bestFeatures(trainx1,trainy1)

m1.summary2()


#AS salary and tenurre is not significant drop those columns one by one and build model eafter each one of drop

#Drop Salary 

data_tr1_1 = data_tr.drop(["salary"],1)
data_tr.columns

#Write generic function to splid data, build model, prediction and converting prob to class

#Function: split data
#Input = data,y,ratio
#returns = trainx,trainy,testx,testy
def splitdata(data,y,ratio = 0.3):
    trainx1_1,testx1_1,trainy1_1,testy1_1 = train_test_split(data.drop(y,1),data[y],test_size=ratio)
    
    return(trainx1_1,trainy1_1,testx1_1,testy1_1)

trainx1_1,trainy1_1,testx1_1,testy1_1 = splitdata(data_tr1_1,"churn")

#Function Build model 
#Build the logistic regression model 
#Input = trainx,trainy
#returns = Logit model
def buildModel(trainx,trainy):
    model = smapi.Logit(trainy,trainx).fit()
    return(model)



#Function predictthe class
#Predict the churn and convert the probabbilities into classes 
#input : predicted Probabilities,cut off value
#p1 < 0.5   --- 1
#p1> 0.5 --------0 
def predictClass(probs,cutoff):
    if (0 <= cutoff <=1):
        P = probs.copy()
        P[ P < cutoff ]  = 0
        P[ P > cutoff ]  = 1
        
        return(P.astype(int))
    
#print the  Confusion Matrix
#Input actual y, Predicted y
#eturns - 
def cm(actual,predicted):
    #Methid 1
    print(confusion_matrix(actual,predicted))
    df = pd.DataFrame({"actual":actual,"predicted":predicted })
    print(pd.crosstab(df.actual,df.predicted,margins = True))
    
    
    
    #print the classification report
    print(classification_report(actual,predicted))
#Build model M1_1(droped salary)
m1_1 = buildModel(trainx1_1,trainy1_1)

##Summarise the model
m1_1.summary()

p1_1 = m1.predict(testx1_1)
print(p1_1)

cutoff = 0.5
pred_y1_1 = predictClass(p1_1,cutoff)

#Confusion matrix
cm(testy1_1,pred_y1_1)

#---------------------------------------
#Drop Tenure

data_tr1_2 = data_tr.drop(["tenure","salary"],1)

data_tr1_2.columns

#Write generic function to splid data, build model, prediction and converting prob to class

#Function: split data
#Input = data,y,ratio
#returns = trainx,trainy,testx,testy
def splitdata(data,y,ratio = 0.3):
    trainx1_2,testx1_2,trainy1_2,testy1_2 = train_test_split(data.drop(y,1),data[y],test_size=ratio)
    
    return(trainx1_2,trainy1_2,testx1_2,testy1_2)


trainx1_2,trainy1_2,testx1_2,testy1_2 = splitdata(data_tr1_2,"churn")

#Function Build model 
#Build the logistic regression model 
#Input = trainx,trainy
#returns = Logit model
def buildModel(trainx,trainy):
    model = smapi.Logit(trainy,trainx).fit()
    return(model)



#Function predictthe class
#Predict the churn and convert the probabbilities into classes 
#input : predicted Probabilities,cut off value
#p1 < 0.5   --- 1
#p1> 0.5 --------0 
def predictClass(probs,cutoff):
    if (0 <= cutoff <=1):
        P = probs.copy()
        P[ P < cutoff ]  = 0
        P[ P > cutoff ]  = 1
        
        return(P.astype(int))
    
#print the  Confusion Matrix
#Input actual y, Predicted y
#eturns - 
def cm(actual,predicted):
    #Methid 1
    print(confusion_matrix(actual,predicted))
    df = pd.DataFrame({"actual":actual,"predicted":predicted })
    print(pd.crosstab(df.actual,df.predicted,margins = True))
    
    
    
    #print the classification report
    print(classification_report(actual,predicted))
#Build model M1_1(droped salary)
m1_2 = buildModel(trainx1_2,trainy1_2)


##Summarise the model
m1_2.summary()

p1 = m1.predict(testx1_2)
print(p1_2)

cutoff = 0.5
pred_y1_2 = predictClass(p1_2,cutoff)

#Confusion matrix
cm(testy1_1,pred_y1_1)



























































#-----------------------------------------------------------
#Run the following command to install the lyb in Anaconda prompt
# pip install imballanced-learn
#-----------------------------------------------------
###OVERSAMPLING#########
#-----------------------------------------------------
#SMOTE ()

from imblearn.over_sampling import SMOTE

sm = SMOTE()
smX,smY  = sm.fit_resample(data_tr.drop("churn",1),data_tr.churn)

#CREATE NEW DATASET

data_tr2 = smX.join(smY)


#Compare te 2 dataset
len(data_tr),len(data_tr2)

#Compare Distributio of the classes(original/oversampled)

data_tr.churn.value_counts(),data_tr2.churn.value_counts()


#Function: split data
#Input = data,y,ratio
#returns = trainx,trainy,testx,testy
def splitdata(data,y,ratio = 0.3):
    trainx2,testx2,trainy2,testy2 = train_test_split(data.drop(y,1),data[y],test_size=ratio)
    
    return(trainx,trainy2,testx2,testy2)


trainx2,trainy2,testx2,testy2 = splitdata(data_tr2,"churn")


#Builed the model on oversampled data
#Function Build model 
#Build the logistic regression model 
#Input = trainx,trainy
#returns = Logit model
def buildModel(trainx,trainy):
    model = smapi.Logit(trainy2,trainx2).fit()
    return(model)

#Build model M2
m2 = buildModel(trainx2,trainy2)

##Summarise the model
m2.summary()


p2 = m2.predict(testx2)
print(p1)

cutoff = 0.50
pred_y2 = predictClass(p2,cutoff)

#Confusion matrix
cm(testy2,pred_y2)

#Cross veryfy the classes 0 and 1
len(testy2[testy2 == 0])
len(testy2[testy2 == 1])



#_----------------------------------------------

####UNDERSAMPLING###########

#-----------------------------------------------

from imblearn.under_sampling import NearMiss
nm = NearMiss()
nmX,nmY = nm.fit_resample(data_tr.drop("churn",1),data_tr.churn)


#Create the new dataset for under sampling

data_tr3 = nmX.join(nmY)

#Compare the len and distribution of the data
#Compare Distributio of the classes(original/undersampled)
data_tr.churn.value_counts(),data_tr3.churn.value_counts()



#Function: split data
#Input = data,y,ratio
#returns = trainx,trainy,testx,testy
def splitdata(data,y,ratio = 0.3):
    trainx3,testx3,trainy3,testy3 = train_test_split(data.drop(y,1),data[y],test_size=ratio)
    
    return(trainx3,trainy3,testx3,testy3)


trainx3,trainy3,testx3,testy3 = splitdata(data_tr3,"churn")


#Builed the model on oversampled data
#Function Build model 
#Build the logistic regression model 
#Input = trainx,trainy
#returns = Logit model
def buildModel(trainx,trainy):
    model = smapi.Logit(trainy3,trainx3).fit()
    return(model)

#Build model M2
m3 = buildModel(trainx3,trainy3)

##Summarise the model
m3.summary()


p3 = m3.predict(testx3)
print(p3)

cutoff = 0.50
pred_y3 = predictClass(p3,cutoff)

#Confusion matrix
cm(testy3,pred_y3)

#Cross veryfy the classes 0 and 1
len(testy3[testy3 == 0])
len(testy3[testy3 == 1])



#--------------------------------------------------
#Balanced sampling
#-------------------------------------------------

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

perc = 0.75
oversample = SMOTE(sampling_strategy = perc)
undersample = RandomUnderSampler(sampling_strategy = perc)


steps = [("o",oversample),("u",undersample)]

bsX,bsY = Pipeline(steps=steps).fit_resample(data_tr.drop("churn",1),data_tr.churn)


data_tr4 = bsX.join(bsY)

#Function: split data
#Input = data,y,ratio
#returns = trainx,trainy,testx,testy
def splitdata(data,y,ratio = 0.3):
    trainx4,testx4,trainy4,testy4= train_test_split(data.drop(y,1),data[y],test_size=ratio)
    
    return(trainx4,trainy4,testx4,testy4)


trainx4,trainy4,testx4,testy4 = splitdata(data_tr4,"churn")


#Builed the model on oversampled data
#Function Build model 
#Build the logistic regression model 
#Input = trainx,trainy
#returns = Logit model
def buildModel(trainx,trainy):
    model = smapi.Logit(trainy4,trainx4).fit()
    return(model)

#Build model M2
m4 = buildModel(trainx4,trainy4)

##Summarise the model
m4.summary()


p4 = m4.predict(testx4)
print(p4)

cutoff = 0.50
pred_y4 = predictClass(p4,cutoff)

#Confusion matrix
cm(testy4,pred_y4)




#---------------------------------------------------------

#Feature selection techiques

#--------------------------------------------------------

from sklearn.feature_selection import f_classif

#Build the function for feature selection
#Returns the scores of all the features of train data set
#Input(train(x&y))
def bestFeatures(trainx,trainy):
    features = trainx.columns
    
    fscore,pval = f_classif(trainx,trainy)
    
    df = pd.DataFrame({"feature":features,"fscore":fscore,"pval":pval})
    df = df.sort_values("fscore",ascending = False)
    return(df)



bestFeatures(trainx1,trainy1)

m1.summary2()

































































