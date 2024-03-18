import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#veri yukleme
veriler = pd.read_csv("eksikveriler.csv")

#veri onisleme

#eksik verileri ortalama deger atama 
imputer = SimpleImputer(missing_values=np.nan ,strategy='mean')
yas = veriler.iloc[:,1:4].values
imputer= imputer.fit(yas[:,1:4]) 
yas[:,1:4] = imputer.transform(yas[:,1:4])

#veriden kategorik verileri ayırma 
ulke=veriler.iloc[:,0:1].values
le=preprocessing.LabelEncoder()    #encoder cagırma 
ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) #fit ve transform fonksiyonlarını cagırma
ohe = preprocessing.OneHotEncoder()  #onehotencoder cagırma 
ulke=ohe.fit_transform(ulke).toarray()  #kategorik verinin programın anlayacagı sekle getirme orn[1,0,0] [0,1,0] [0,0,1]

#dataframe olusturma
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

#dataframleri birlestirme 
#default concat kolonlar üzerinden olmayan kolonlar nan yazılır 
#axis satırlara gore birlestirme 
s=pd.concat([sonuc,sonuc2], axis=1)    
print(s)
print("*********************************************")
s2=pd.concat([s,sonuc3], axis=1)       
print(s2)
print("*********************************************")

#datasetin egitim ve test icin split edilmesi
x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size=0.33 , random_state=0)

#verilerin olceklenmesi
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
