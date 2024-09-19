import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris=load_iris()

# veri setindeki anahtarlara bakalım
print(iris.keys())

print("\n**********\n")

# veri setinin özelliklerime bakalım
print(iris["DESCR"])

print("\n**********\n")

# tartget_names tahmin etmek istediğimiz çiçeğin türlerini gösterir
print(iris["target_names"])

print("\n**********\n")

# feature_names niteliklerin isimlerini gösterir
print(iris["feature_names"])

print("\n**********\n")

# verinin tipini görelim
print(type(iris["data"]))

print("\n**********\n")

# varinin yapısını görelim
print(iris["data"].shape) # 1. değer satır sayısını 2. değer sütun sayısını gösterir

print("\n**********\n")

# ilk beş örneklemn nitelik değrlerini görelim
print(iris["data"][:5])

print("\n**********\n")

# target verisi herbir çiçeğin türünü gösteriyor (3 tür var)
print(iris["target"])

print("\n**********\n")

# modeli değerlendirmek veri setini 2 parçaya bölelim
# ilk parça eğitim ikimci parça test verisidşr

# veriyi karıştırmak ve ayırmak için train_test_split() kullanılır
# bu fonksiyon önce satırları karıştırır sonra 75% eğitim verisi ve 25% test verisi olarak eyırır
# ama eğer istersen farklı oranda da parçalayabilirdin

# veri(data) X ile , etiket(target) y ile göterilir   --> f(x)=y

# şimdi eğitim test değişkenlerini oluşturalım
# random state ile veriyi karıştıralım
from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris["data"],iris["target"],
                                               random_state=0)
print(X_train.shape) # iki boyutludur
print(y_train.shape) # tek boyutludur
print()
print(X_test.shape) # iki boyutludur
print(y_test.shape) # tek boyutludur

# dikkat edersem eğitim verisi yüze 75 oldu kalanı test oldu

print("\n**********\n")
# VERİ ÖN İNCELEME
# görselleştirme ile inceleyelim
# eğitim verisini dataframe çevirip görselleştiricez
import pandas as pd
iris_df=pd.DataFrame(X_train,columns=iris.feature_names)
from pandas.plotting import scatter_matrix
# c verinin türlere göre renklenmesini sağşar çünkü y_train türlere eşittir
# histogramların dikdörtgen genişlikleri için bins:20 oldı
# noktaların büyüklüğü için s=80
# noktaların görünrlüğü 0.8 olsun
scatter_matrix(iris_df,c=y_train,figsize=(15,15),
               marker="o",hist_kwds={"bins":20},
               s=80,alpha=0.8)
# plt.show()

print("\n**********\n")

# grafiğe göre parçalar ayrı ayru güzel gruplanmış demekki makine öğrenmesi modelimiz güzel çalışacak
# k en yakın komşu sınıflandırmasını yapalım:
# model kurma yalnız eğitim verisinden yapılır.
# Yani bir veri noktası için algoritma bu kontaya en yakın eğitim verisindeki noktayı buluur
# bu eğitin verisi noktasının etiketi yeni veri noktasına atanır
from sklearn.neighbors import  KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)

# modeli kurmak için fit()
knn.fit(X_train,y_train)

# şimdi bu modeli kulllanarak yeni veriler için tahmin yapalım veriler şöyle olsun
# sepal_length:5  , sepal_width:2.9 , petal_length:1 , petal_width:0.2
import  numpy as np
#
X_yeni=np.array([[5,2.9,1,0.2]])

# tahmin yapalım predict  (yeni irisin türünü tahmin et)
tahmin=knn.predict(X_yeni)
print("Tahmin sınıfı:",tahmin)
print("Tahmin türü:",iris["target_names"][tahmin])

print("\n**********\n")

# modelin performansına bakalım
# şişmdi her bir iris çiçeği için tahmin yapıp, çıkan sonuçları gerçek türler ile karşılaştırmalıyız
# böylece modelin ne kadar iyi çalıştuğına bakalım

# önce tahminlere bakalım
y_predict=knn.predict(X_test)
print(y_predict)

print("\n**********\n")

# şimdi gerçek verilerlr tahminleri karşılaştıralim
# burdan çıkan sonuç modelin iris verisini yüzde kaç tahmin ettiğini göwsterir (%97)
print(np.mean(y_predict==y_test))
print()
# diğer yöntem de şudur:
print(knn.score(X_test,y_test))

