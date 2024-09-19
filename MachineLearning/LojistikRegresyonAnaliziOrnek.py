"""
ŞU İŞLEMLER SIRAYLA YAPILACAK:
VERİ SETİ TANINACAK (veri önişleme, temizleme, görselleştirme...)
MODEL EĞİTME
MODEL DEĞERLERNDİRME
YENİ VERİ TAHMİN ETME
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
veri=pd.read_csv("titanic.csv",sep=";")
print(veri.head())

print("\n*************\n")

# veri setinin satrır(örneklem) ve sütun(öznitelik) sayısına bakalım
print(veri.shape)

print("\n*************\n")

# değişkenlerin tiplerine bakalım
print(veri.dtypes)

print("\n*************\n")

# veri setini görsetlleştirelim
import seaborn as sns
# hayatta kalan ve kalmayan kişilerin sayısına bakalım
sns.countplot(x="survived",data=veri)
#plt.show()

print("\n*************\n")

# sınıflara göre hayatta kalan kişilere bakalım
sns.countplot(x="survived",hue="pclass",data=veri)
#plt.show()

print("\n*************\n")

# hayatta kalan ve kalmayan kişilerin cinsiyetlerini de görelim
# yani cinsiyetlere göre hayatta kalan kişilere bakalım
sns.countplot(x="survived",hue="sex",data=veri)
#plt.show()

print("\n*************\n")

# yaş değişkeni floattı
# şimdi yaş değişkeninin histogramını çizdirelim
veri["age"].plot.hist()
#plt.show()

print("\n*************\n")

# bilet üzcreti değişkemmim hiastogramını çizdirelim
# histogramların kaç tane olacağını söyler mesela 20 bar olacak
# figaize grafipin genişliğinin 10 inç olacağını ve yüksekliğinin 5 inç olacağını söyler
veri["fare"].plot.hist(bins=20,figsize=(10,5))
#plt.show()

print("\n*************\n")

# yolcuların kardeş sayılarını görelim
sns.countplot(x="sibsp",data=veri)
# plt.show()

print("\n*************\n")

# ŞİMDİ VERİ ÖNİŞLEME YAPALIM

# eksik verileri düzenlyeleım

# eksik veri var mı bakalım
print(veri.isnull().sum())

print("\n*************\n")

# şimdi bu eksik verileri görselleştirelim heaemap ile yaplır
# eksik veriler sarı renkte görüldü
# veri.isnull() ile eksik veriler tespit edilir
# yticklabels: y eksenindeki etikelerin yani satır isimlerinin görünümünü kapatır
# viridis de renk paletinin nasıl olacağını belirler
sns.heatmap(veri.isnull(),yticklabels=False,cmap="viridis")
#plt.show()

print("\n*************\n")

# eksik veri fazla olan değişkenleri veri setinden kaldıralım
# kaldırılacak şey sütun olduğundan axis=1 olur
veri.drop(["cabin","boat","body","home.dest"],inplace=True,axis=1)
print(veri.isnull().sum()) # eksik verinin çok pşdığu sütılar kaldırışdı

print("\n*************\n")

# hala eksik veri içeren satırlar var
# bu eksik veri içeren satırları da kaldıralım
veri.dropna(inplace=True)
print(veri.isnull().sum()) # hiç eksik veri kalmadı

print("\n*************\n")

# değişkemlerim tipine bakalım
print(veri.dtypes)
# OBJECT TİPİNDEKİ DEĞİŞKENŞER VERİ SETİNE DAHİL EDİLEMEZ ONŞARI DAHİL ETMEK İÇİN DUMMY TİPİNE DÖNÜŞTÜRMELİYİZ

print("\n*************\n")

# object olan değişkenleri dummy değişkenine dönüştürelim bunu get_dmmies ile yapalım
# drop_first ile kategorik değişkenlerin birden fazla sütun olarak ifade edilmesi durumunda, ilk sütunu kaldırırz
# drop_first=True, gereksiz bir sütunu kaldırarak veri setinin daha küçük ve daha kullanışlı olmasını sağlar. Örneğin, cinsiyet ("sex") değişkeni "male" ve "female" kategorilerinden oluşuyorsa, yalnızca birini ("male" veya "female") tutar. Diğer kategori bilgisi bu sütunun 0 veya 1 olmasına bağlı olarak zaten anlaşılabilir.
# yani artık hem female hem male olmucak mesela sadece male olucak onlar 0 1 olucak eğer 0 ise female anlamına gelecek ama 1 ise male anşamına gelecek çünkü female sütunu düşürüldü ve male sütununa 0 1 yazıldı
sex=pd.get_dummies(veri["sex"],drop_first=True)
print(sex.head()) # 0 değeri kadınları, 1 değeri erkekşer, gösterir

print("\n*************\n")

# embarked değişkeninin altgruplarını ekrana yazdıralım
print(veri.embarked.value_counts())

print("\n*************\n")

# şimdi embarked ve pclass değişkenlerini de dummy yapalım
embarked=pd.get_dummies(veri["embarked"],drop_first=True)
pclass=pd.get_dummies(veri["pclass"],drop_first=True)
# böyşece katefpril değişkenler dummy oldu

# şimdi bu değişkenleri veri setine  ekleuelim
# ama önce eski değişkenleri veri setinden çıkaralım
veri.drop(["sex","embarked","pclass"],axis=1,inplace=True)
veri=pd.concat([veri,sex,embarked,pclass],axis=1)

# şimdi tekrar veir setideli değişkenlerin tipine bakalım
print(veri.head())
# artık sadece name ve ticket obje tipinde

# bu değişkemler analize dahil edilmeyecek
# bu yüzden onları veri setindeen kaldırabiliriz
veri.drop(["name","ticket"],axis=1,inplace=True)

print("\n*************\n")

# veri setinin son halime bakalımÇ
print(veri.head())

print("\n*************\n")

# ŞİMDİ LOJİSTİK REGRESYON ANALİZİYLE VERİLERİNİ GİRDĞİMİZ KİŞİNİN HAYATTA KALIP KALAMAYACAĞINI GÖRÜCEZ
# yani hedef değişken survived oldu

# veri setini girdi çıktı olarak ayıralım
X=veri.drop("survived",axis=1) # survivedi veri setinden atıp kalanları girdi olarak aldık
y=veri["survived"]

# veri setini eğitim ve test şeklimde parçalayalım
# eğitim verisiyle model kurulur ve test verisiyle model değerlendirilir
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=100)
"""
regresyon analizi şunlardan oluşur: lineer,polinomal,lojistik regresyon
    hedef değişken sayısal ve bağımsız değişkenler ile arasında doğusal bir ilişki varsa lineer regredyon analizi kullanılır
    hedef değişken sayısal ama bağımsız değişkenler ile arasında doğusal(lineer) bir ilişki yoksa polinomsal regredyon analizi kullanılır
    hedef değişken kategorik ise lojistik regredyon analizi kullanılır
"""
# hedef değişken doğru yanlış evet hayır gibi ikili kategorilerden oluşuyorsa ikili lojistik regresyon kullanılır
# hedef değişken ikiden fazla kategoriden oluşuyorsa çoklı lojistik regresyon kullanılır
# hedef değişken bebek çoçok genç yaşlı gibi sıralı kategorilerden oluşuyorsa sıralı lojistik regresyon kulanılır

# mesela yarın hava nasıl olabilir sorusu için lineer regresynın kullanılır ama yarın hava yağmurlu mu değil mi sorusu için lojistik regresyon kullanılır
# burada da hayatta kaldı mı kalmadı mı onu araştırdığımız için ikili lojistik regresyon kullancaz

# X_train ve X_test sütun isimlerini string türüne dönüştürelim
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

from sklearn.linear_model import LogisticRegression
lg_model=LogisticRegression(max_iter=1000)

# modeli eğitim veriyile eğit
lg_model.fit(X_train,y_train)

# şimdi modelim doğruluk skoruna bakalım
print(lg_model.score(X_test,y_test)) # yani model yeni bir veriyi %77 oranında doğru tahmin eder

print()

# şimdi eğitim verisindeki skora bakalım
print(lg_model.score(X_train,y_train)) # %80
print()

print("\n*************\n")

# ama biz modelin eğitim ve test verisindeli skorlarının yakın olmasını istiyoruz
# bizim modelimiz eğitim verisinde daha iyi demekki overfitting(aşırı uydurma) var
#overfittingi düzeltmek için C argumanını küçültmeliyiz.
# çünkü C argümanı küçülürse test verisindeli skor azalır
lg_model=LogisticRegression(C=0.1,max_iter=1000)
lg_model.fit(X_train,y_train)

print(lg_model.score(X_test,y_test))
print()
print(lg_model.score(X_train,y_train))
# böylece test ve eğitim verileri birbirine yaklaştı

print("\n*************\n")

# MODELİ DEĞERLENDİRELİM
# modelin sınfılarının ne kadar doğrı tahmin ettiğini bulmak için matrisi kullanalım
from sklearn.metrics import confusion_matrix

# şimdi test verisindeki değerlerin tahminlerini bulalım
tahmin=lg_model.predict(X_test)

# şimdi matrix ile tahminleri ekrana yazdıralım
print(confusion_matrix(y_test,tahmin))

print("\n*************\n")

# şimdi bir veriye göre o kişinin hayatta kalıp kalmayacağını bulucaz

# önce kontrol etmek için ilk kişinin verilerini alıp kurduğumuz moele göre doğru tahmin yapıp yapmadığına bakalım
print(veri.head())
print()
# numpy 2 boyutlu bir dizi bekler ama bizim yeni veri_dizimit tek boyulu olduğu için
# reshape() ile onu iki boyutlu hale getirmemiz gerek
yeni_veri=np.array([29,0,0,211.3375,0,0,1,0,0]).reshape(1,-1) # 1 yazarak dizinin 1 satıra dahip olmasını sağladık, -1 yazarak da numpy'a bu boyutun otomatik olarak hesaplanmasını ve belirlenmesini söyler. Yani, dizinin geri kalan elemanları sütunlara dağıtılır.
print(lg_model.predict(yeni_veri)) # sonuç 1 oldu zaten 1 olmalıydı yani model dığru çalışıypr

print()

# şimdi modeli yeni veride deneyelim yeni veridekş kişinin hayatta kalığ kalmama olasıloığına bakalım
yeni_veri2=np.array([30,1,1,150,0,0,1,0,0]).reshape(1,-1)
print(lg_model.predict(yeni_veri2)) # ekrana 1 yazıldıüıma göre bu kişi hayatta kalabilir

