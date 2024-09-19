# İKİLİ SINIFLANDIRMA, LOJİSTİK REGRESYON, ÇOKLU SINIFLANDIRMA, LİNEER DESTEK VEKTÖR MAKİNELERİ
import matplotlib.pyplot as plt
# ikili sınıflansırma yani classificationda sınfılar bir doğru ile ikiye ayrılır
# logistig regressiyon da bir sınıflandırmadır regresyon değildir
# yani kategorikaldır

# mesela yaşa göre sağlık sigortasının karşılanıp karşılanmayacağını bulur (evet,hauır)
# en yakın noktaların ordan ortalama bir eğti çizilir

from sklearn.datasets import load_breast_cancer
kanser=load_breast_cancer()

# verileri eğitim ve test biçiminde iki parçaya ayıralım
from sklearn.model_selection import train_test_split

# stratify: veriyi bölerken her sınıftan (örneğin kanserli ve kansersiz) yeterli miktarda veri olmasını garanti eder. Böylece eğitim ve test setlerindeki sınıf oranları aynı kalır.
X_train,X_test,y_train,y_test=train_test_split(kanser.data,kanser.target,
                                               stratify=kanser.target,random_state=42)

#şimdi logistic regreyonu import edelim
from sklearn.linear_model import LogisticRegression

# modelden bir örnek alıp eğitim verisine göre eğitelim
# C öntanımlı olarak 1
# ayrıca FutureWarningg uyarısı almamak için istersen şunu yap solver="libliner"
logreg=LogisticRegression(solver="liblinear").fit(X_train,y_train)

# şimdi modelin performanslarına bakalım
print(logreg.score(X_train,y_train))
print(logreg.score(X_test,y_test))
# eğitim ve test verileri birbirine çok yakın olduğına göre modelde underfitting var
# bu problemin üstesünden gelmek için C argümanını 1'den 100'e arttıralım

print()

logreg100=LogisticRegression(C=100,solver="liblinear").fit(X_train,y_train)
print(logreg100.score(X_train,y_train))
print(logreg100.score(X_test,y_test))
# şimdi eğitim ve test veririndeli performans arttı yani model daha iyi çalıştı
# modelde regülerlrştirmeyi artttırdık böylece epitim test verisindeki doğruluk arttu

print()

# şimdi C değerimi düşürelşm
logreg001=LogisticRegression(C=0.01,solver="liblinear").fit(X_train,y_train)
print(logreg001.score(X_train,y_train))
print(logreg001.score(X_test,y_test))
# regülerleştirmeyi azalttık böylece eğitim ve test doğruluk değerleri düştü ki bu olmamaoı bu kötü bişi

print("YANİ C DEĞERİ ARTTIKÇA REGÜLERLEŞTİRME ARTAR VE EĞİTİM TEST VERİSİ PERFORMANSI ARTAR Kİ BU ASIL İSTEDİĞİMİZ ŞEYDİR")

print("\n*************\n")

# şimdi C'nin aldığı değerler göre modellerin öğrendiği katsayıları grafikte görelim
# modelde x eklseninde modeldeki öznitekil katsayıları vardır. y ekseninde ise bu katsayıların değerleri vardur
# max_iter=1000 parametresi, makine öğrenmesi modellerinde maksimum yineleme (iteration) sayısını belirler. Özellikle Lojistik Regresyon, Destek Vektör Makineleri (SVM) ve Gradient Descent tabanlı optimizasyon algoritmaları kullanan diğer modellerde yaygın olarak kullanılır.
# max_iter=1000 parametresi, modelin eğitimi sırasında en fazla 1000 defa deneme yapacağını belirtir. Eğer model bu 1000 deneme içinde çözüm bulamazsa, durur ve "çözüm bulunamadı" hatası verebilir.
# penalty="l1" parametresi, modeli daha basit hale getirmek ve aşırı uyumu (overfitting) önlemek için bazı özelliklerin (feature) etkisini sıfıra indirmeye çalışır. Bu sayede, model daha az sayıda özellik kullanarak tahmin yapar ve daha anlaşılır bir hale gelir.
# şimdi for döngüsüyle c'nin bu 3 farklı değerleri için modekin eğitim test verisi için doğruluk değerlerşnş bulalım
for C,market in zip([0.001,1,100],["o","^","v"]):
    lr_l1=LogisticRegression(penalty="l1",max_iter=1000,solver="liblinear",C=C).fit(X_train,y_train)
    print("C={:.3f} için eğitim doğruluk {:.2f}".format(C,lr_l1.score(X_train,y_train)))
    print("C={:.3f} için test doğruluk {:.2f}".format(C,lr_l1.score(X_test,y_test)))

# EĞER ÇOK ÖZNİTELİK VARSA VE MODELİN KOLAY YORUMLANMASINI İSTERSEK LASSO KULLANILIR

print("\n*************\n")

# ŞİMDİ İKİLİ SINIFLANDIRMA ALGORİTMAMIZI ÇOKLI SINIFLANDIRMAYA DÖNÜŞTÜRELŞM
from sklearn.datasets import make_blobs
import mglearn

# veri setindeki girdi ve çıkları görelim
X,y=make_blobs(random_state=42)
# X'in herbir değeri x ve y koordinatlarını içeren veri noktalarıdır
#y hedef değişkenleri tutar
print(X[:5])
print(y[:5])
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.xlabel("Öznitelik 1")
plt.ylabel("Öznitelik 2")
plt.legend(["Sınıf-0", "Sınıf-1", "Sınıf-2"])
plt.show()

print("\n*************\n")

# destek vektör makinelerine bakalım
from sklearn.svm import LinearSVC
linear_svm=LinearSVC().fit(X,y)

# modelin katasayılarına bakalım
print(linear_svm.coef_)

# şimdi sınfıları ikiye parçalayan 3 doğrunun grafiğini çizelim
# öncelikle veri setindeki ilk iki özellğin saçılım grafiğini çizelim
import numpy as np
mglearn.discrete_scatter(X[:,0],X[:,0],y)
line=np.linspace(-15,15) # -15 ve 15 arasında değerler üretelim
for coef,intercept,color in zip(linear_svm.coef_,linear_svm.intercept_,["b","r","g"]):
    plt.plot(line,-(line*coef[0]+intercept)/coef[1],c=color)
plt.ylim(-10,15)
plt.ylim(-10,8)
plt.xlabel("Öznitelik 1")
plt.ylabel("Öznitelik 2")
plt.legend(["Sınıf-0","Sınıf-1","Sınıf-2","Doğru-Sınıf-0","Doğru-Sınıf-1","Doğru-Sınıf-2"],
           loc=(1.02,0.4))
plt.show()

