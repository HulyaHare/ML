# bağımlı ve bağımlar arasındaki ilişkiyi inceleme tekniğidir
# mesela bir kişinin kilosu pnun bpyuna bağlıdır
# bağımlı ve bağımsız değişken arasında lineer yani doğrusal ilişki aranır
import matplotlib.pyplot as plt

# girdiler(attribute) bağımsız değişkendir ama çıktılar bağımlı(target) değişkendir
# girdilker matristir, çıktılar vektördür

# analizde 1 tane öznitelik varsa basit lineer regresyondur
# 1'den fazla öznitelik varsa çoklu lineer regresyon denir

# girdi verliler kullanılarak çıktı tahmin edilir

# bir öğrencinin ders çalışma süresi ve bir önceki sımav notuna göre sınav notunu tahmin etmek istersek:
# ders çalışma süresi ve bir önceki notu özniteliktir, sonvazdan aldığı notu da hedef değişkendir

# lineer regresyonda sayısal değişkenler arasındaki ikişkiyi bulmak için çizgi kulalanılır
# en küçük kareletr yöntemiyle bulunur yani:
# tahmin ve hedef değer arasondaki farkların karesel toplamı hatayı verir

# y = w[0]*x[0] + ..... + w[p]*x[p] + b
# X[0] dan X[p] ye kadar olan değişkenler özniteliklerdir yani örneğimizde bunlar öğrencinin çalışma saati ve önceki not oralamasıdır
# p öznitetlik sayısıdır
# w ve b değerleri modelin parametreleridir. en küçük kareler ile bunlar tahmin edilir
# y model tahminidir yani öğrendinin sımav notudur

# basit bir lineer regresyon modeli şudur
# y = w[0]*x[0] + b
# bu model doğruyu gösterşrş
# w[0] eğimi gösterir
# b sabit terimi gösterir

# önce lineer regresyonu mglesrn ile öğrenelim view veri setibi kullanarak öğrenicez
import mglearn
import pandas as pd

# eğim ve sabiti bulalım
mglearn.plots.plot_linear_regression_wave()
# plt.show()

print()

# veris etinde iki öznitelik varsa düzlem, 2'den fazla olanlar için çok boyutlu düzlem çizilir

# eğim ve sabit terim en küçük kareler ile bulunur

from sklearn.linear_model import LinearRegression
X,y=mglearn.datasets.make_wave(n_samples=60)

# veri setini parçalayalım
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
# modeli kuralım
lr=LinearRegression().fit(X_train,y_train)

#modeldeki eğimi yani katesayısı görelim
print(lr.coef_)
print()
# modeldeki sabiti görelim
print(lr.intercept_)

print()

# şimdi eğitim ve test verilerine göre doğruluk skorlarını ekrana yazdıralım
print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))
# birbirlerine yakınlar ama 1'e yakın değiller
# demekki underfitting var
# yani olmadı yani sonuç değişkenini açıklamak için başka değişkenlere de ihtiyazdımız var demektir



print("\n**********\n")



#gerçek dünya veri seti kullanalum
# amaç öğrencinin final notunu tahmin etmek
import pandas as pd
veri=pd.read_csv("student-mat.csv",sep=";")

# veri setindeki özniteliklerden final notunu ve bu nota etki eden öznitelikleri seçelim
# G1:1.not , G2:2.not , G3:finalnotu
veri=veri[["G1","G2","G3","studytime","failures","absences","age"]]

print(veri.head())
print()

# özniteliklerin ismini değiştirelim
veri.rename(columns={"G1":"Not1",
                     "G2":"Not1",
                     "G3":"Final",
                     "studytime":"Calisma_Suresi",
                     "failures":"Sinif_Tekrari",
                     "absences":"Devamsizlik",
                     "age":"yas"},
            inplace=True)

print(veri.head())
print()

# veri seitndeki değişkenlerin veri yapısına bakalım
print(veri.dtypes)
print()

# analiz için girdi çıktı değpişkenleri oluşturalım
# tahmin edilecek değişkeni yani final notunu y değişkenine atayalım
# scikitlearnda model oluşturulurken veri numpy şeklinde olmalıdır
import numpy as np
y=np.array(veri["Final"])
# girdi verisi için hedef değişkeni drop ile veriden düşürüp kullanalım
X=np.array(veri.drop("Final",axis=1))

# modeli eğitim test olarak parçalayalım
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

# şimdi modeli lineer regresyonla kuralım
from sklearn.linear_model import LinearRegression
# bu sınıftan bir örnek alalım
linear=LinearRegression()

# veriyi eğitelim
linear.fit(X_train,y_train)

#test verilerini kullanarak modelin performansını görelim
print(linear.score(X_test,y_test)) # demekki model %83 oranında doğru tahmin yapıyot
print()
print(linear.score(X_train,y_train))

# test ve epitim doğruluk skorları birbirine yakın yani güzel

# model ağırlıklarını ekrana yazdıralım
print("Katsayılar: \n",linear.coef_)
print("Sabit: ",linear.intercept_)

print()

# baştaki veri setini tekrar uazdıralım
# ilk katsayı Not1'in, ikinci katsayı Not2'nin, Final değişkeni tahmin değişkeni, 3. atsayı çalışma süresinin şeklimde eşleştirilir

# kurduğumuz modele göre şimdi kendi oluşturduğumuz verinin final notunu tahmin edelim
yeni_veri=np.array([[10,14,3,0,4,16]])
print(linear.predict(yeni_veri)) # yukardaki notlara göre tahmini final notunu bulduk

print("\n**********\n")


print("\n**********\n")


print("\n**********\n")


print("\n**********\n")


print("\n**********\n")


print("\n**********\n")


print("\n**********\n")


print("\n**********\n")


print("\n**********\n")


print("\n**********\n")


print("\n**********\n")

