# geçemn derslerde sınıflama performansını değrlendirmek için accuricy, regresyon performansını değrlendirmek için r kare kullanmıştık
# ama modelş depğerlendirmek için bal-şka metotlar da varıdr
# iyi değerlendirmömek için bu metriği iyi seçmelisin
import matplotlib.pyplot as plt
# tahmin performansını değrelendirmek için 3 farklı metrik vaardır:
# birincisinde tahminciler bir score metoduna sahiptir: bu metot öntanımlıdır
# ikincisi cross validation kullanarak model değelendirmedir
# üçüncüsü de metrik modülü ölçüm fonksiyonudur (tahmin hatalarını dpesifik olarak değelendirir)

# dummy tahmincileri basit model oluşturmamza izin verir
# bu basit model ile kurduğumuz modeli karşılaştırabiliriz
# aslında dummy tahmincileri araştırmadaki doğallığı gösterir
# mesela bir müşteri mağazaya girince öznitelikleirmizden bağımsız olarak 100 lira harcıyor
# bu varsayımı bir baseline olarak belirtirsek kurduğumuz makina öğrenmesi yaklaşımının ne kadar iyi çalıştığını anlauabiliriz

# şimdi dengesiz veri seti için yani imbalanced detasets için ikili sınıflandırma düşünelim
# imbalanced veri setinde iki sınıflamadan biri diğerinden daha fazla tekrar eder
# mesela iris veri setini imbalanced yapalım
# irisin bir türüne karşı diğerierini sınıflayalım
from sklearn.datasets import load_iris
iris=load_iris()

X,y=iris.data,iris.target

# çıktı değişkeninde 1 ile kodlanmayan sınıflara -1 değerini atayalım
y[y!=1]=-1

# veri setini parçalayalım
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

# şimdi destek vejgör makineleri modeli kurup bu model ile dummy tahmincisityle oluşturacağımız modeli karşılaştıralım
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
# fonksiyon doğrusal yani linear olcak
# C hataları ceza ile kontrol eder. Küçük C değerleri daha esnek bir model oluşturur ve yüksek bias (sapma) ile düşük varyansa sahip bir modele yol açabilir. Büyük C değerleri ise daha az esnek bir model oluşturur ve düşük bias ile yüksek varyansa neden olabilir. Bu durumda, C=1 orta düzey bir düzenleme sağlar.
svc=SVC(kernel="linear",C=1).fit(X_train,y_train)
print(svc.score(X_test,y_test))

print()

# şimdi dummy ile modeli kuralım
clf=DummyClassifier(strategy="most_frequent",random_state=0).fit(X_train,y_train)
print(clf.score(X_test,y_test))
print()

#şimdi destek vektör makinelerindeki oranları ayarlayalım
svc=SVC(kernel="rbf",gamma="scale",C=1).fit(X_train,y_train)
print(svc.score(X_test,y_test)) # performans arttı
print()



print("\n*******\n")



# ikili sınıflama değerlendirme sonuçlarını görmek için confusion matrix kullanalım
# bunu göstermek için digits veri setini örnek alalım
from sklearn.datasets import load_digits
digits=load_digits()

# imbalanced veri setini oluşturmak için 9 rakamını bir sınıf ve diğer rakamları siğer sınıf olarak düşünelim
# yani 9 rakamı 1 oldu diğerşeri 0 oldu yani 2 sınıf var
y = digits.target==9
X_train,X_test,y_train,y_test=train_test_split(digits.data,y,random_state=0)

# bu veri seti için logistig regresyon kullanalım
from sklearn.linear_model import LogisticRegression
# solver="liblinear": Küçük veri setleri ve ikili sınıflandırma problemleri için uygun olan optimizasyon algoritmasını belirtir.
# C=0.1: Düzenleme parametresi; modelin karmaşıklığını kontrol eder. Küçük C değerleri, modelin daha basit olmasına ve aşırı uyumdan (overfitting) kaçınmasına yardımcı olur.
logreg=LogisticRegression(solver="liblinear",C=0.1).fit(X_train,y_train)

# bu model ile test verisini tahmin edelim
# yani eğitimli model kullanılarak test verisindeki örnekler için sınıf tahminleri yapılır.
pred_logreg=logreg.predict(X_test)

# şimdi bu tahminleri görelim
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred_logreg)
print(cm)
"""
cm şuna eşit:

Bu çıktı, karmaşıklık matrisi (confusion matrix)'dir ve lojistik regresyon modelinin test verisi üzerindeki performansını özetler. Karmaşıklık matrisi, modelin doğru ve yanlış sınıflandırmalarını tablo halinde gösterir ve dört temel bileşeni vardır:
True Negative (TN): Doğru negatif tahminler (gerçek sınıf negatif, modelin tahmini de negatif).
False Positive (FP): Yanlış pozitif tahminler (gerçek sınıf negatif, modelin tahmini pozitif).
False Negative (FN): Yanlış negatif tahminler (gerçek sınıf pozitif, modelin tahmini negatif).
True Positive (TP): Doğru pozitif tahminler (gerçek sınıf pozitif, modelin tahmini de pozitif).

[[TN  FP]
 [FN  TP]]

cm şu şekildedir:
[[401   2]
 [  8  39]]
 
Bu matrisin her bir elemanı aşağıdaki anlamlara gelir:
[0, 0] (401): True Negatives (TN)
Gerçek sınıfı "9 değil" olan 401 örnek doğru bir şekilde "9 değil" olarak tahmin edilmiştir.
[0, 1] (2): False Positives (FP)
Gerçek sınıfı "9 değil" olan 2 örnek yanlış bir şekilde "9" olarak tahmin edilmiştir.
[1, 0] (8): False Negatives (FN)
Gerçek sınıfı "9" olan 8 örnek yanlış bir şekilde "9 değil" olarak tahmin edilmiştir.
[1, 1] (39): True Positives (TP)
Gerçek sınıfı "9" olan 39 örnek doğru bir şekilde "9" olarak tahmin edilmiştir.
"""

# bu fadeleri daha iyi görmek için mglearn kullanalım
import mglearn
mglearn.plots.plot_confusion_matrix_illustration()
plt.show()



print("\n*******\n")



# confusion matris özetlemenin başka yolları da vardur
from sklearn.metrics import classification_report
print(classification_report(y_test,pred_logreg,target_names=["not nine","nine"]))



print("\n*******\n")



# rog eğrisi analizi (rog curve)
# rog eğrisinin hedefi yanlış pozitif ve yanlış negatifleri minimize edecek eşik değerini bulur
from sklearn.metrics import roc_curve
# şimdi false pozitive, true pozitive ve tresholds(eşik değişkenler) belirleyelim
fpr,tpr,thresholds=roc_curve(y_test,logreg.decision_function(X_test))

# şimdi log eğrisini çizdirelim
plt.plot(fpr,tpr,label="ROC CURVE")
plt.xlabel("FPR")
plt.ylabel("TPR(recall)")
plt.show()

#ideal bir eğri sol üst köşeye yakındır
# analizde sınıflayısının düşük yanlış pozitif orana ve yüksek recall yani hassasiyete sahip olması beklenir



print("\n*******\n")



# sınıflama sayısı 2'den fazla da olabilir
# önce eğitim ve test olarak ayıralın
X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,random_state=0)

#şimdi logistig regresyın ile modeli kuralım
lr=LogisticRegression(solver="liblinear",multi_class="auto").fit(X_train,y_train)

# test verisini tahmin edelim
pred=lr.predict(X_test)
# çoklu sınıfların sonuçları doğruluk(accuricy) ve confusion matris ile değerlendirilir

# öncelikle doğruluk skorunnu elde edelim
from sklearn.metrics import accuracy_score

# şimdi doğruluk ve confusion matris görelim
print(accuracy_score(y_test,pred)) # %95 deoğruluğu var modelin
print()
cm=confusion_matrix(y_test,pred)
print(cm)

print()

# sınıflama performansını görselleştirelim
import seaborn as sns
import pandas as pd

# numpy dizi yapısındad olan veriyi dataframe çevirelim
df=pd.DataFrame(cm,index=digits.target_names,columns=digits.target_names)

# sonuçları görselleştirelim
sns.heatmap(df,annot=True,cbar=True,cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()