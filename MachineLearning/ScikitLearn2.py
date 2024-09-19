# denetimsiz öğtrnmryi kullanıp resimlerdeki rakamları fark eden algoritma kuralım
import matplotlib.pyplot as plt
import numpy as np
# iris veri setinin boyutunu azaltmalıyız

import pandas as pd

iris=pd.read_csv("iris.txt")
print(iris.head())

print("\n**********\n")

# itis veri setinin 4 tane özniteliği yani boyutu vardır
# boyut azaltma yapmalıyız
# boyut azaltmanın amacı veri setinin daha düşük boyutla temsil edilip edilemeyeceğini görmektir
# boyutu azaltılan veri seti daha kolay görselleştirir amaç 2 boyuta indirmektir

# şimdi öznitelik matirislerini oluşturuo bunları kullanarak boyutu azaktalım

# PCA, verideki en önemli bileşenleri seçer ve gürültüden kaynaklanan önemsiz değişkenleri azaltabilir.

from sklearn.decomposition import  PCA
# n_components=2 diyerek modeli 2 ana bileşene ayırdık yani veri setinin özelliklerinin sayısını 2'ye düşürdük
# yani 4 özelliktem 2'sini önemli bileşen olarak seçip bu bileşenlere indirdik
model=PCA(n_components=2)

#şimdi fit() medotuyka modeli kuralım:

# iris veri setinden öznitelik matrisi ve hedef dizisini oluşturaLIM
# özniteik matrisi:
# species'i görmek istiyoruz ve bu bir sütun olduğundan axis=1 oldu (iki boyutlu)
X_iris=iris.drop("species",axis=1)

# modeli X_iris nesnesiyle kuralım
# bu sayede model verinin önemlş bşleşenlerini bulmayı öğrenir
model.fit(X_iris)

#  modeli dörtten iki boyutlu hale getirelim
X_2D=model.transform(X_iris)
print(X_2D)
print()

# şimdi sonuçların graffiğğini çizdirelim
# bu değişkenleri iris veri setine ekleyerek grafiği daha kolay çizdirirz
iris["PCA1"]=X_2D[:,0]
iris["PCA2"]=X_2D[:,1]

print(iris.head())

print()

# şimdi grafik çizelim
import seaborn as sns
# hue=species diyerek speciese göre grafik yaptık yanı speciesler renklenir
# fit_reg=False diyerek grafikte regresyon çizgisi çizmeyi devre sışı bıraktık
#  Regresyon çizgisi, bir veri kümesindeki noktaların genel eğilimini gösteren düz bir çizgidir
sns.lmplot(x="PCA1",y="PCA2",hue="species",data=iris,fit_reg=False)
#plt.show()
# fit_reg=False diyerek grafikte regresyon çizgisi çizdik
sns.lmplot(x="PCA1",y="PCA2",hue="species",data=iris,fit_reg=True)
#plt.show()

print("\n**********\n")

# KÜMELEME ANALİZİ (denetimsiz öğrenme algoritmasıdır)
# GaussianMixture veriyi gruplara ayırır
from sklearn.mixture import GaussianMixture
# n_components=3 diyerek veriyi 3 farklı gruba ayırır
# covariance_type herbir grubum şeklini ve boyutunu belirler.
# full dediğimiz için elips olur ve her yöne genişleyebilir. Bu esnek ve geneldir yani herbir grubun farklı boyut ve şekillerde olmasına izin verir
model=GaussianMixture(n_components=3,covariance_type="full")

# Gaussian Mixture Model'i X_iris veri setiyle eğitir. Model, bu veriyi kullanarak 3 farklı gruba (cluster) ayırmaya çalışır ve her bir gruba ait olasılıkları belirler.
model.fit(X_iris)

# şimdi kümeleme etiketlerini belirleyelim
# yani herbir veri noktasının hangi gruba(cluster) ait olduğunu tahmin edelim
y_gmm=model.predict(X_iris)
iris["kumeleme"]=y_gmm
print(iris.head())

print("\n**********\n")

# şimdi sonuçların grafiğini çizdirelim
sns.lmplot(x="PCA1",y="PCA2",hue="species",data=iris,col="kumeleme",fit_reg=True)
# plt.show()

print("\n**********\n")

# şimdi denetimsiz öğrenmeyle uygulama yapalım
# el yasısıyla yazılan rakamları tanımayı öğrenelim
# veri seti sklearn içinde vardır
from sklearn.datasets import load_digits
digits=load_digits()

# digits.images: Bu, veri setindeki her bir rakamın 8x8 piksel boyutunda resim olarak saklandığı bir veri yapısıdır.
print(digits.images.shape)
# yani 1797 tane örneklem ve bu örneklemlerin herbiri 8x8 pixelden oluşuyor

print("\n**********\n")

# bu veri setinin ilk 100'ünü görselleştiricez
#Bu satır, bir grafikte birden fazla küçük resim (subplot) oluşturur ve ilk 100 örneği görselleştirmek için kullanılır.
fig,axes=plt.subplots(10,10, # 10 satır ve 10 sütundan oluşan 100 küçük grafik (subplot) oluşturur. Bu, ilk 100 rakamı göstermek için yapılır
                     figsize=(8,8), # Oluşturulan grafik alanının boyutunu inç cinsinden belirler. Burada grafik alanı 8 inç x 8 inç olarak ayarlanmıştır.
                     subplot_kw={"xticks":[],"yticks":[]}, # Her bir küçük grafikte (subplot) x ve y eksenlerinde işaretlerin (tick marks) gösterilmeyeceğini belirtir. Bu, grafikleri daha temiz ve net yapar.
                     gridspec_kw=dict(hspace=0.1,wspace=0.1)) # hspace=0.1: Alt grafikler (subplots) arasında dikeyde (yani yukarıdan aşağıya) küçük bir boşluk bırakır. wspace=0.1: Alt grafikler arasında yatayda (yani soldan sağa) küçük bir boşluk bırakır.

# görselleştirme için for döngüsü kullanılır
# i: Döngü sayacıdır, her döngüde 0'dan 99'a kadar birer birer artar (çünkü 100 rakam var).
# ax: Her döngüde axes.flat içindeki bir alt grafiği (subplot) temsil eder.
# enumerate(axes.flat): axes.flat, 10x10'luk grafikler dizisini düz bir liste haline getirir.
# enumerate ise bu listeye bir sayaç (index) ekler. Yani, i her bir alt grafiğin indeksini (sırasını) ve ax da o grafiği temsil eder.
for i,ax in enumerate(axes.flat):
    # ax.imshow(): Alt grafik (subplot) ax içinde bir resim (image) gösterir.
    # digits.images[i]: digits veri setinden i numaralı resim alınır. Örneğin, i=0 için ilk rakamın (örneğin, "0" rakamının) 8x8 piksellik görüntüsüdür.
    # cmap="binary": Görüntü için kullanılan renk haritası (colormap). binary, siyah-beyaz bir renk haritası kullanır. Siyah ve beyaz tonları ile resmi gösterir.
    # interpolation="nearest": Görüntüyü gösterirken herhangi bir yumuşatma yapılmaz, en yakın piksel değerleri gösterilir. Bu, piksellerin net ve keskin olmasını sağlar.
    ax.imshow(digits.images[i],cmap="binary",interpolation="nearest")
    # ax.text(): Alt grafikte belirtilen konuma bir metin (text) ekler.
    # 0.05, 0.05: Metnin grafikteki x ve y eksenlerinde nerede görüneceğini belirler. (0.05, 0.05), grafiğin sol alt köşesine yakın bir noktadır.
    # transform=ax.transAxes ifadesi, yazının yerini grafik alanının boyutuna göre ayarlamamızı sağlar.
    # transform=ax.transAxes, metni eklerken kullandığımız koordinatların, o alt grafik (subplot) alanı içinde olduğunu ve bu alanın boyutuna göre ayarlandığını belirtir. Böylece, (0.05, 0.05) gibi koordinatlarla yazıyı grafiğin istediğimiz yerine yerleştirebiliriz
    ax.text(0.05,0.05,str(digits.target[i]),transform=ax.transAxes,color="green")
#plt.show()

print("\n**********\n")

#scikit learn içinde bu veri setiyle çalışmak için veriyi örneklem ve öznitelik sayısı olacak şekilde iki boyutlu diziye çevirmeliyiz
# yani resimdeli her pixel bir öznitelik olacaktır
# her rakam 8x8 pixelden oluşuyordu
# her örneklemin pixellerini düz düşümürsek 8x8=64 değerlerinde pixel değerlerinin dizisine sahip oluruz

print("\n**********\n")

# şidmi öznitelik matrisini ve hedef dizisini oluşturalım
X=digits.data
y=digits.target
print(digits)
print()
print(X)
print()
print(y)

print("\n**********\n")

# şimdi öznitelik matrisindeki 64 boyut yani özn,teliği görselleştirelim
# ama bunun için önce veriyi 2 boyuta indirgemeliyiz
from sklearn.manifold import Isomap
# Isomap sınıfından iso isminde bir değişken alıp bu Isomap sınıfındaki bir örneği bu değişkene atayalım
# 2 boyutlı olması için n_components=2 olsun
iso=Isomap(n_components=2)
# fit medounun içine X özniteilik matrisini yazıp öğrenmesini sağşa
# newaxis ile dizi iki boyutlu hale gelir yani:
# Örneğin, x tek boyutlu bir diziyse ([1, 2, 3, ...] gibi), np.newaxis ile bu dizi iki boyutlu hale getirilir ([[1], [2], [3], ...] gibi). Bu, bir sütun vektörü oluşturur.
iso.fit(X)

# şimdi data2 isminde değişken alıp transform metodunu kullanarak dönüşüm yap
data2=iso.transform(X)
print(data2.shape)

print("\n**********\n")

# şimdi grafik çizelim
plt.scatter(data2[:,0],data2[:,1],
            c=digits.target,
            alpha=0.5,
            cmap=plt.cm.get_cmap("tab10",10))
plt.colorbar(label="digit etiket",ticks=range(10))
#plt.show()

print("\n**********\n")

# sınıflandırma algoritmasını veriye uygulayacağız
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)
# modeli tahmin edelim
y_model=model.predict(X_test)

# şimdi modelin doğruluk oranını bulalım
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_model)

# şimdi nerelerede yanlış yaptuğımızı grafik ile görelim
# confusiın matrix yanlış yaptupımız yerleri gösterir
from sklearn.metrics import confusion_matrix
mat=confusion_matrix(y_test,y_model)

# square=True ile hücreler kare oldu
#annot=True ile her hücrenin içine sayısal değeri ekşendi
sns.heatmap(mat,square=True,annot=True,cbar=False)
plt.xlabel("Tahmin Değer")
plt.ylabel("Gerçek Değer")
# plt.show()
