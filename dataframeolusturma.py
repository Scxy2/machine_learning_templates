import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer
from sklearn import preprocessing


veriler = pd.read_csv("eksikveriler.csv")
#print(veriler)
imputer = SimpleImputer(missing_values=np.nan ,strategy='mean')
yas = veriler.iloc[:,1:4].values
#print(yas)
#print("*********************************************")
imputer= imputer.fit(yas[:,1:4]) #ögrenme 
yas[:,1:4] = imputer.transform(yas[:,1:4])#ogrendigini uygulama 
#print(yas)
#print("*********************************************")
ulke=veriler.iloc[:,0:1].values    #veriden kategorik verileri ayırma 
#print(ulke)
#print("*********************************************")
le=preprocessing.LabelEncoder()    #encoder cagırma 
ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) #fit ve transform fonksiyonlarını cagırma
#print(ulke)                                     
#print("*********************************************")
ohe = preprocessing.OneHotEncoder()  #onehotencoder cagırma 
ulke=ohe.fit_transform(ulke).toarray()  #kategorik verinin programın anlayacagı sekle getirme orn[1,0,0] [0,1,0] [0,0,1]
#print(ulke) 
#print("*********************************************")

sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us']) #kategorik verilerin dataframe 
print(sonuc)
print("*********************************************")
sonuc2 = pd.DataFrame(data=yas, index = range(22), columns = ['boy','kilo','yas'])  #eksik verilerin dataframe
print(sonuc2)
print("*********************************************")
cinsiyet = veriler.iloc[:,-1].values      #cinsiyeti normal datasetten cekme 
print(cinsiyet)
print("*********************************************")
sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ['cinsiyet'])   #cinsyeti dataframe haline getirme 
print(sonuc3)
print("*********************************************")
#default concat kolonlar üzerinden olmayan kolonlar nan yazılır 
#axis satırlara gore birlestirme 
s=pd.concat([sonuc,sonuc2], axis=1)    #dataframleri birlestirme 
print(s)
print("*********************************************")
s2=pd.concat([s,sonuc3], axis=1)       #dataframleri birlestirme 
print(s2)
print("*********************************************")