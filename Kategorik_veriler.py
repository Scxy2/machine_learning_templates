import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer
from sklearn import preprocessing


veriler = pd.read_csv("eksikveriler.csv")
print(veriler)
imputer = SimpleImputer(missing_values=np.nan ,strategy='mean')
yas = veriler.iloc[:,1:4].values
print(yas)
print("*********************************************")
imputer= imputer.fit(yas[:,1:4]) #ögrenme 
yas[:,1:4] = imputer.transform(yas[:,1:4])#ogrendigini uygulama 
print(yas)
print("*********************************************")
ulke=veriler.iloc[:,0:1].values    #veriden kategorik verileri ayırma 
print(ulke)
print("*********************************************")
le=preprocessing.LabelEncoder()    #encoder cagırma 
ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) #fit ve transform fonksiyonlarını cagırma
print(ulke)                                     
print("*********************************************")
ohe = preprocessing.OneHotEncoder()  #onehotencoder cagırma 
ulke=ohe.fit_transform(ulke).toarray()  #kategorik verinin programın anlayacagı sekle getirme orn[1,0,0] [0,1,0] [0,0,1]
print(ulke) 
print("*********************************************")

