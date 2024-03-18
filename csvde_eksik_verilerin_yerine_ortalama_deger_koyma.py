#import'lar 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer
#kodlar 
#veri yukleme 
#veriler = pd.read_csv("veriler.csv")  #kolaylık olsun diye aynı dizin 
#print(veriler)
#veri on isleme 
#boy = veriler[['boy']]
#print(boy)
#boykilo = veriler[['boy','kilo']]
#print(boykilo)
#sinif olusturma 
class insan: 
    boy = 180 
    def kosmak(self,b):
        return b+10
#eksik veriler
veriler = pd.read_csv("eksikveriler.csv")
print(veriler)
#eksik verileri giterme 
imputer = SimpleImputer(missing_values=np.nan ,strategy='mean')
yas = veriler.iloc[:,1:4].values
print(yas)
imputer= imputer.fit(yas[:,1:4]) #ögrenme 
yas[:,1:4] = imputer.transform(yas[:,1:4])#ogrendigini uygulama 
print(yas)
