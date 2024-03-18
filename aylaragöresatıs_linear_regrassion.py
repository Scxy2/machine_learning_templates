import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
#veri yukleme
veriler = pd.read_csv("satıs.csv")
ay=veriler[['Aylar']]
satıslar=veriler[['Satislar']]
#satıslar=veriler.iloc[:,:1].values

#datasetin egitim ve test icin split edilmesi  (bağımsız değişken , bağımlı değişken) 
x_train,x_test,y_train,y_test=train_test_split(ay,satıslar,test_size=0.33 , random_state=0)
'''
#verilerin olceklenmesi
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''
#linear regression (model insasi)
lr=LinearRegression()   
lr.fit(x_train,y_train)

#tahmin yaptırma
tahmin = lr.predict(x_test)

#görsellestirme
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title("Aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")