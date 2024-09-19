# UNSUPERVİSED LEARNING

# denetimli öğrenmede sonuçlar ya da etiketler biliniyordu ve bu sonuçlara danışılıyordu
# denetimsiz öğrenmde sonuçlar bilinmiyor
# iki çeşit denetimsiz öğrenme var : veri setini dönüştürme veya kümeleme

# veri setini dönüştürmenşn en genel uygulaması boyut indirgemedir
# boyut indirgeme ile birçok öznitelikten oluşan yüksek boyutlu veri, temel karakterlerş gösteren özniteliklere indirgenir
# 2 boyuta indirgenir
# denetimsiz öğrenme için ilk şey veriyi oluşturan bileşenleri bulmaktır

# veri setini dönüştürmenşn diğer bir yolu kümelemedir
# verilerin aynı grupda yer alacak şekilde ayrılması sağlanır.
# aynı gruptaki nesneler oabildiğince birbirine benzer olmalıdır

# denetimsiz öğrenmede etiketler olmadığından sonuçların ne kadar doğru olduğunu tam bilemeyiz

# denetimsiz öğrenmenin diğer çeşidi şudur:
# denetimli öğrenme algoritmaları için preproccessing yapar
# çünkü verileri önişlemek performansı arttırır ve bellekte az yer kaplamasını sağlar
# ölçekleme ve önişleme metotlaro genelde denetimli öğrenme algoritmalarından önce kullanılır

# ŞİMDİ ÖNİŞLEME VE VERİ ÖLÇEKLEME YAPMAYI ÖĞRENİCEZ

# mesela kanser veri setine destek vektör uygulamak istiyoruz
# veri önişleme için minmax scaler kullanalım

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

kanser=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(kanser.data,kanser.target,random_state=1)

# veriyi ölçeklemek için sınıfı impırt edelim
# MinMaxScaler, veriyi ölçeklendirmek için kullanılan bir sınıftır.
# Verilerin, belirli bir aralık (genellikle 0 ile 1) içinde normalize edilmesini sağlar.
from sklearn.preprocessing import MinMaxScaler
# şimdi snıftan nesne alıp onu fit edelim
# bu fit eğitimi sadece X_train nesnesine uygulanır
scaler=MinMaxScaler().fit(X_train)
# "Fit" işlemi, her bir özelliğin minimum ve maksimum değerlerini öğrenir ve bu değerlere göre dönüşüm kurallarını belirler.
# Bu işlemin ardından scaler nesnesi, veriyi 0 ile 1 arasında ölçeklendirmek için gerekli bilgiyi (min ve max değerleri) içerir.

# şimdi bu ölçeklemeye göre veriyi dönğştürmek için transfırm() kullan
# sklearnda her zaman veriyi yeni şekilde ifade etmek için transform() kullanılır
X_train_olcekli=scaler.transform(X_train)

# şimdi ölçeklenmeden önceki ve ölçeklendirkten sonraki veri setindeki özniteliklerin min değerlerin görelim
print(X_train.min(axis=0))
print(X_train.min(axis=0))
print()
print(X_train_olcekli.min(axis=0)) # ölçeklenmiş daha iyi
print(X_train_olcekli.min(axis=0)) # ölçeklenmiş daha iyi
# yani veri ölçeklendi
# dçnüştürlen veri eski veriyle aynı yapıda aynu kastsayılara fakan sahip ama 0-1 arasındadır

print()
print()

# test verisini de dönüştürmemiz gerekir
# bunlar ama sadece 0 ve 1 olmaxlar çünkü
# X_train fit edilmişti yani test verisi farklı değerler olabşkşr ölçelense bile
X_test_olcekli=scaler.transform(X_test)
print(X_test_olcekli.min(axis=0)) # ölçeklenmiş daha iyi
print(X_test_olcekli.min(axis=0)) # ölçeklenmiş daha iyi



print("\n*************\n")




# şimdi destek vektör makineleri üzwirnde minmax scaleer ölçeklenmesinde bakalım
from sklearn.svm import SVC
# veri setini eğitim test olarak parçalaualım
X_train,X_test,y_train,y_test=train_test_split(kanser.data,kanser.target,random_state=0)
# C: Modelin hatalara ne kadar tolerans göstereceğini kontrol eder.
# Düşük C: Daha genel model, Yüksek C: Daha hassas model (overfitting).
# Gamma: Her bir veri noktasının karar sınırına ne kadar etki edeceğini belirler.
# Düşük gamma: Daha genel etkiler, Yüksek gamma: Daha yerel ve hassas etkiler Overfitting.
svm=SVC(C=100,gamma="auto")
svm.fit(X_train,y_train)
print(svm.score(X_test,y_test))

print()
# şimdi bir de modeli kurmadan önce veriyi ölçekleyip kuralım
scaler=MinMaxScaler().fit(X_train)
X_train_olcekli=scaler.transform(X_train)
X_test_olcekli=scaler.transform(X_test)
# şimdi ölçeklenömiş veriyle tekrar model kuralım
svm.fit(X_train_olcekli,y_train)
print(svm.score(X_test_olcekli,y_test)) # skor arttu

