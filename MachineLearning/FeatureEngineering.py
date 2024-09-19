import pandas as pd

data=[{"not":85,"kardes":4,"ders":"mat"},
      {"not":70,"kardes":3,"ders":"ing"},
      {"not":65,"kardes":3,"ders":"mat"},
      {"not":60,"kardes":2,"ders":"fiz"},
      ]

# veriyi sayısal kodlama ile haritalayabiliriz yani kodlayabiliriz
# {"mat":1,"ing":2,"fiz":3}
# ama direk bunu yaparsak python bunu anlayamaz o yüzden OneHotCoding lullanarak otomatik yapmalıyız
# ekstra sütun eklenerek kategoriin varlığı ve yokluğı durumları için 0 ve 1 kulllanılır
# DictVectorizer sözlü formatındaki veriyi sayısal veriye dönüştürür
from sklearn.feature_extraction import  DictVectorizer

# eğer matrisinizde çok fazla 0 yoksa sparse=false yap
vek=DictVectorizer(sparse=False,dtype=int)

# fit ile verilen data öğrenilir ve transform ile sayısal forma çevrilir
print(vek.fit_transform(data))

print()

# şimdi jherbir sütunun etiketine bakalım
print(vek.get_feature_names_out())

print("\n**********\n")

# şimdi kelime sayısına göre sayısal kodlara çevirelim
veri=["hava iyi","iyi insan","hava bozuk"]

from sklearn.feature_extraction.text import CountVectorizer
vek=CountVectorizer()

X=vek.fit_transform(veri)

# dönüştürdüğümüz veriyiy görmek için pandas kullanalım
print(pd.DataFrame(X.toarray(),columns=vek.get_feature_names_out()))

print("\n**********\n")

# öznitelik türetme
# öznitelikler girdi verilerinden polinomsal öznitelik oluşturular
import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,4,5])
y=np.array([5,3,1,2,7])

# saçılım grafiğime balaşom
plt.scatter(x,y)
plt.show()

# şimdi limeer regresyon kullanarak veriye doğru uyduralım
from sklearn.linear_model import LinearRegression
X=x[:,np.newaxis] # veriyi iki boyutlu yaptık
print(X)

print()

# modeli kuralım
model=LinearRegression().fit(X,y)

# veriyi tahmin edelim
y_fit=model.predict(X)
plt.scatter(X,y)
plt.plot(X,y,y_fit)
plt.show()

print("\n**********\n")

# daha gelişmiş modelle ilişki tanımlanmalıdır
# veriyi- dönğştğrerek modeli iyi hale getireniliriz
# böylece ekstra öznitelik sütunu ekleyerek modele esneklik kazandırabiliriz
from sklearn.preprocessing import PolynomialFeatures
# veriye PolynomialFeatures ile verinin özelliğinin karesini küpünü falan da ekleyip doğrusal ol mayan modelle daha iyi öprenmesinş sağlıcaz
# degree=3 diyerek 3. derece polinom elde edicez
# include_bias ile sabit terimi eklemeyi engelledik çünkü false
pol=PolynomialFeatures(degree=3,include_bias=False)

X2=pol.fit_transform(X)
print(X2)

# şimdi modeli kuralım
model=LinearRegression().fit(X2,y)
y_fit=model.predict(X2)
plt.scatter(X,y)
plt.plot(X,y,y_fit)
plt.show()

print("\n**********\n")

# eksik verileri ele alalım
from numpy import nan
X=np.array([[1,nan,3],
            [5,6,9],
            [4,5,2],
            [4,6,nan],
            [9,8,1]])
y=np.array([10,13,-2,7,-6])

# bu verilere makina öğrenmöesi modelini uygulamak için kayıp verileri dolduralım
# kayıp veirleri sütun ortalamasıyla dolsuralım
from sklearn.impute import SimpleImputer
imp=SimpleImputer(strategy="mean")
X2=imp.fit_transform(X)
print(X2)

print()

# makina öğrenmesi yapalım
model=LinearRegression().fit(X2,y)
print(model.predict(X2))

print("\n**********\n")

# pipeline tekniği
# az önce yaptıklarımızı aynı anda yapabiliriz
"""
Eksik Verileri Doldurma (SimpleImputer): Modelin eksik değerlerle düzgün çalışabilmesi için eksik verileri sütun ortalamalarıyla doldururuz. Böylece, eksik veri içeren gözlemler de eğitim sürecine dahil edilir.
Polinom Özellikler Ekleme (PolynomialFeatures): Doğrusal modelin doğrusal olmayan ilişkileri de öğrenebilmesi için giriş verilerine polinom (2. derece) özellikler ekleriz. Bu, modelin karmaşık ilişkileri daha iyi öğrenmesini sağlar.
Doğrusal Regresyon Modeli (LinearRegression): Elde edilen yeni verilerle doğrusal regresyon modelini eğitiriz ve tahminler yaparız.
"""
from sklearn.pipeline import make_pipeline
model=make_pipeline(SimpleImputer(strategy="mean"),
                    PolynomialFeatures(degree=2),
                    LinearRegression())
model.fit(X,y)
print(y)
print()
print(model.predict(X))
