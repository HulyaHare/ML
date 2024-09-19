# bağımsız değişkenler kullanılarak bağımlı değişken tahmin edilir
# burada yine lineer regresyon yapıcaz
# veri setinde öznitelik sayısı fazöa olduğunda ağırlıkları yani katsayıları tahmin eden en küçük kareler yönteminden dolayı model komplexliği artar
# Bu kötü bişey çünkü özniyelik ekleyip modelii-n komplexliği artında model ezberlemeye başlar. Ezberleme overfitting problemini ortaya çıkartır
# modelin ezberlememesi gerek
# ezberleme probleminin üstesşnden gelmek için ridge ve lasso kullanılr
import numpy as np
# bu derste Ridge regression, regülerleştirme, Lasso Regression ve ElasticNet konularına bakacağız

# RİDGE REGRESSİON
# bu da bir lineer regresyon modelidir
# katsayılar yine eğitim verisinden eğitilir ama bir kısıtlama ile fit edilir
# katsayıların değeri mümkün olduğunda küçük olmalıdır çünkü hem iyi bir tahmin yapmak istiyoruz hem de mümkün olduğunca az özniteliğin sonuca etkietmesini istiyoruz

# buna regülerleştrime denir
# regülerleştşrme ile overfittingden kaçınmak için model kısırtlanır

# en küçük kareler formülüne model katsayısılarının karelerş toplamı eklendiği için ridge regresyona aynı zamanda r2  regülerleştirme de denir
# regülerleitirmede komplexlik parametresi olan alpha büzülmeyi kontrol eder

# yani alpha büyük oldukça grafikteli her öznitelik birbirine yaklaşur ama lüçük pşdıukça uzaklaiır büyülmez

import pandas as pd
from sklearn.model_selection import train_test_split

boston = pd.read_csv("boston.csv",sep=";")
print(boston.head())

# eksil verileri dolduralım
boston=boston.fillna(boston.mean())

print("\n**********\n")

# şimdi X ve y değişkenlerini şçyle bulucaz
X = boston.iloc[:, :-1]  # Tüm sütunları al fakat son sütunu çıkar
y = boston.iloc[:, -1]   # Sadece son sütunu al

# şimdi X değişkenini genişletmemiz gerek
# X değişkenini bulurken genişletme yapıcaz yani matrisi X,X^2,X^3 diye giderek büyütücez
# veri setini genişleterek doğrusal olmayan ilişklilei yakalıcaz modl performamsomo artırıcaz ve genel ilişkileri öğrenicez
from sklearn.preprocessing import PolynomialFeatures
# özellikleri genişletelim:
poly=PolynomialFeatures(degree=2,include_bias=False) # include_bias deyip genişletilmiş matrise sabit terim eklemeyi öğrendik. Böylece, yalnızca orijinal ve polinom türevli (polynomial) terimler kalır. Yani gereksiz işlem yapılmasını önledik modelde zaten sabit terim var
X=poly.fit_transform(X)

print(X.shape) # görülüpü gibi X'in bpyutu (506,13) iken genişledi ve (506,104) oldu

# önce eğitim ve testleri bulalım
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

# r,dge regresyon kullanalım
from sklearn.linear_model import Ridge
ridge=Ridge().fit(X_train,y_train)

# şimdi eğitim ve test verilerine göre modelin doğruluk skorlarını göelim
print(ridge.score(X_train,y_train))
print(ridge.score(X_test,y_test))
"""ÖNEMLİ"""
# yani ridge modelin eğitim verisindeki performansı azalırken genelleştirilemsi daha iyi oldu lineere göre
# eğer genelleştirme ile ilgilenirsen lineer regresyon yerine ridge kullan

print("\n**********\n")

# ridge modeli daha çok kısıtladığından model daha az komplex olur
# daha az kompexin eğitim verisindekli performansı daha kötüdür ama modelin genelleştirilmesinde iyidir bunu yukarda da görsük
# aslında ridge modelin basitliği ile eğitim veririsindeki modelin perfornansı arasındaki trade-off denen dengelemeyi sağlar

# modelin eğitim verisindeki performansına karşı basitliğin ne kadar olacağını alpha belirler
# bu alpha öntanımlı olarak 1 gelir
# alphanın değeri arttıkça katsayılar sıfıra doğru daha da yaklaşır
# bu durum modelin eğitim verisindeki performansını azaltırken genelleştirmeyi arttırır

# mesela alpha 10 olsun (eğitim performansı azalır genelleştirme artar)
ridge10=Ridge(alpha=10).fit(X_train,y_train)
print(ridge10.score(X_train,y_train))
print(ridge10.score(X_test,y_test))

print()

# şimdi alpha 0.1 olsun (eğitim performansı artar genelleştirme azalır)
# model lineer regresyona benzer
ridge01=Ridge(alpha=0.1).fit(X_train,y_train)
print(ridge01.score(X_train,y_train))
print(ridge01.score(X_test,y_test))

print("\n**********\n")


print("\n**********\n")


# lasso da bir limeer regresyondur
# lineer regresyona mutlak değer kullanarak eklenen regülrelrştırmeden dolayı lassoya R1 regülerleştirmesi denir
# lassonun mandtığı ridgeye benzer
# lasso öznitelik sayısını azaltır
# ridge gibi lasso da katsayıları 0'a yaklaştırmak için kısıtlamalr yağılır
# hatta bazı katsayılar 0 alınır yani bazı öznitelikler modele dahil edilmez
# böylece modelde daha önemli öznitelikler yer alır

# şimdi boston veri setine lassso uygulayalım
from sklearn.linear_model import Lasso
lasso=Lasso().fit(X_train,y_train)

print(lasso.score(X_train,y_train))
print(lasso.score(X_test,y_test))

print()

# skorlar kötü çünkğ modelde underfittinglik var yani değişken sayısı az
# modelde kullanılan değişken sayısına bakalım
print(np.sum(lasso.coef_!=0))

# modelde 105 özitelikten yalnızcsa 35 tanesi kullanılır
# underfitting azaltmak için alpayı azaltalım
# böylece düşük alpha değeri daha komplex öpdel oluşturmayı sağlar
# max_iter=100000: Algoritmanın maksimum yineleme (iterasyon) sayısını belirtir. Bu, modelin eğitim sırasında belirli bir doğruluğa ulaşamadığında çalışmaya devam etmesini sağlar.
lasso001=Lasso(alpha=0.01,max_iter=100000).fit(X_train,y_train)
print(lasso001.score(X_train,y_train))
print(lasso001.score(X_test,y_test))

print()



print("\n**********\n")

