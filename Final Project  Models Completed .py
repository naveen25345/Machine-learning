#!/usr/bin/env python
# coding: utf-8





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle





dataset= pd.read_csv(r'C:\Users\Krishna\ML internship\autos.csv',encoding='ISO-8859–1')
dataset



type(dataset)

dataset[['vehicleType']]





dataset[['vehicleType']].isnull().any()





dataset[['vehicleType']].isnull()





dataset[["vehicleType"]].values





dataset.isnull().any()




dataset.nunique()

# =============================================================================
# Description of each column
# 
# dateCrawled         : when advert was first crawled, all field-values are taken from this date
# name                : headline, which the owner of the car gave to the advert
# seller              : 'privat'(ger)/'private'(en) or 'gewerblich'(ger)/'dealer'(en)
# offerType           : 'Angebot'(ger)/'offer'(en) or 'Gesuch'(ger)/'request'(en)
# price               : the price on the advert to sell the car
# 
# abtest              : ebay-intern variable 
# vehicleType         : one of eight vehicle-categories 
# yearOfRegistration  : at which year the car was first registered
# gearbox             : 'manuell'(ger)/'manual'(en) or 'automatik'(ger)/'automatic'(en)
# powerPS             : the power of the car in PS
# 
# model               : the cars model
# kilometer           : how many kilometres the car has driven
# monthOfRegistration : at which month the car was first registered
# fuelType            : one of seven fuel-categories
# brand               : the cars brand
# 
# notRepairedDamage   : if the car has a damage which is not repaired yet
# dateCreated         : the date for which the advert at 'ebay Kleinanzeigen' was created
# nrOfPictures        : number of pictures in the advert
# postalCode          : where in germany the car is located
# lastSeenOnline      : when the crawler saw this advert last online
# =============================================================================
# # Drop unnecessary columns




dataset.groupby('offerType').size()





#Gesuch means request and no need to consider 12 requests so drop offer type column





dataset['nrOfPictures'].sum()





# no pictures so remove that column





#DateCrawled , Date Created and last seen are for all practical purposes the same for all the cars so remove those columns 





dataset.groupby('seller').size()





# Basically all cars have same sellers so remove that column too





#abtest isn't related to the cars so its removed





dataset.drop(['dateCrawled','nrOfPictures','lastSeen','postalCode','dateCreated','abtest','seller','offerType'],axis=1,inplace=True)





# Removing month by adding 1 to year if month >=6 and keeping year as it is if month is less than 6





dataset['yearOfRegistration']





dataset.columns





a=dataset.iloc[:,3].values
a





b=dataset.iloc[:,8].values
b





n=len(a)
i=0
while i<= n-1:
    if b[i]<6:
        pass
    else:
        a[i]=a[i]+1
    i+=1
        





a





type(a)





new_series = pd.Series(a)





dataset['yearOfRegistration']=new_series
dataset['yearOfRegistration']





dataset.drop([ 'monthOfRegistration'],axis=1,inplace=True)





dataset


# # Cleaning up the Data 




#Take a small samle of our data





sample_data=dataset.sample(n=10000, random_state=1)





#Seeing and analysing price





plt.subplot(3,1,1)
sample_data['price'].hist(bins=40)
plt.title('Original-Histogram price')
plt.show()





# Cars at  100000 or more are too expensive for used cars as they might be some vintage collecters cars and they might skew the rest of the data





dataset = dataset[dataset.price < 100000]
sample_data=dataset.sample(n=10000, random_state=1)





plt.subplot(3,2,1)
sample_data['price'].hist(bins=50)
plt.title('Intermediate-Histogram price')
plt.show()





# Price Distribution looks a little better but still work to do





# We gotta remove cars with extremely low price from skewing the data because they are like donations or even scams





# So remove cars with price less than 10





dataset=dataset[dataset.price>10]





sample_data=dataset.sample(n=10000, random_state=1)





plt.subplot(3,1,3)
sample_data['price'].hist(bins=50)
plt.title('Final-Histogram price')
plt.show()





#Final Histogram looks good with less freebies and very expensive cars





# Now onto the years of registration





dataset['yearOfRegistration'].describe()





# 1000 AD and 9999 AD can't be true . Cars were first made in 1863 and present day in 2020. Removing data that is just not feasible 





dataset=dataset[(dataset.yearOfRegistration>1863)&(dataset.yearOfRegistration<=2020)]





# Power in ps of 0 and over 1000 is unrealistic so remove those rows





dataset = dataset[(dataset.powerPS > 0) & (dataset.powerPS < 1000)]





# Now we have clean data but it has some Nan values





clean_data=dataset


# # Dealing with null values




clean_data.isnull().any()





clean_data['gearbox'].value_counts()





clean_data['model'].value_counts()





clean_data['fuelType'].value_counts()





clean_data['notRepairedDamage'].value_counts()




clean_data['vehicleType'].value_counts()

# =============================================================================
# All the data which has null values has a good enough distribution of all the categories of the column. 
# We are forced to delete the rows that have null values for crucial data,because we can't predict the null values reliably
# =============================================================================



# Making clean data super clean





superclean_data = clean_data.dropna()
print(superclean_data.describe())


# #  Encoding




superclean_data.columns





superclean_data.isnull().any()





superclean_data['brand'].value_counts()

# =============================================================================
#  We use label encoding for vehicleType,gearbox,model ,fuelType,brand,notRepairedDamage.
#  We use One Hot encoding for name .
#  We will replace yearOfRegistration with "ageOfCar" which is basically '2020-yearOfRegistration'
# =============================================================================



# first dealing with year of Registration





h=superclean_data.iloc[:,3].values
h





len(h)





g=list(h)
g




d=0





f=[]
for item in g:
    d=2020-item
    f.append(d)





f





#Add age of car column





superclean_data['ageOfCar']=f





superclean_data





#Drop year of Registration





superclean_data.drop(['yearOfRegistration'],axis=1,inplace=True)





superclean_data


# # Label Encoding




from sklearn.preprocessing import LabelEncoder





superclean_data.loc[:,"vehicleType"] =LabelEncoder().fit_transform(superclean_data.loc[:,"vehicleType"])
superclean_data.loc[:,"fuelType"] =LabelEncoder().fit_transform(superclean_data.loc[:,"fuelType"])
superclean_data.loc[:,"gearbox"] =LabelEncoder().fit_transform(superclean_data.loc[:,"gearbox"])
superclean_data.loc[:,"notRepairedDamage"] =LabelEncoder().fit_transform(superclean_data.loc[:,"notRepairedDamage"])
superclean_data.loc[:,"brand"] =LabelEncoder().fit_transform(superclean_data.loc[:,"brand"])
superclean_data.loc[:,"model"] =LabelEncoder().fit_transform(superclean_data.loc[:,"model"])
superclean_data.loc[:,"name"] =LabelEncoder().fit_transform(superclean_data.loc[:,"name"])


# Reason for using label encoding on name given further below




superclean_data


# # Splitting  Data into x ,y




y=superclean_data.loc[:,"price"].values
x=superclean_data.loc[:,['name','gearbox','vehicleType','powerPS','model','kilometer','fuelType','brand','notRepairedDamage','ageOfCar']].values











type(x)





x





x.shape




y.shape

# =============================================================================
# ohe=OneHotEncoder(categories=[0])ct =ColumnTransformer([("on",OneHotEncoder(sparse=False),[0])],remainder='passthrough')
# x=ct.fit_transform(x)lb=LabelBinarizer()x= lb.fit_transform(x[:,0])
# =============================================================================



superclean_data.nunique()


# We quite simply can't use One hot Encoding or even Label Binarizer for name because both of them ask for ridiculus amounts of memory which results in memory error. They ask around 29GB and 10.9  respectively. So instead I am using Label Encoding for name which isn't ideal but has been forced by memory limitations

# # Train Test Split




x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)





x_train


# # Random Forest Regressor




from sklearn.ensemble import RandomForestRegressor





rf=RandomForestRegressor(n_estimators=100,criterion="mse",random_state=0)





x_test





y_test





rf.fit(x_train,y_train)





y_pred=rf.predict(x_test)
y_pred





y_test





acuuracy= r2_score(y_test,y_pred)



pickle.dump(rf, open('my_model.pkl','wb'))





my_model = pickle.load(open('my_model.pkl','rb'))
print(my_model.predict(x_test[[2]]))


