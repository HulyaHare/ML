# TEMEL BİLEŞENLER ANALİZİ
# veriyi görselleştirmek ve az sayıda faktör ile özniteliklerin büyük bşr kısmını temsil etmektir
# bunları yapmak için PCA temel bileşenekere ayırma analizidir
# bu veri setini döndüren bir metottur
# döndürülen öznitelikler istatistiksel olarak ilişkilidir
# böylece veri setindeki temel bileşenler elde edilir

# bu derste şunları göreceğiz:
# PCA(temel bileşenler analizi) nedir
# nasıl boyut indirgenir
# çok boyutlıu veriler nasıl görselleştirilir
# parazitler nasıl filtrelenir
# yğksek boyutrlu veri setlerinden önemli öznitelikler nasıl çekilip çıkartırılır

# bazı makine öğrenmesi algoritmaları bimlerce hatta milyonlarca öznitelikten olulur
# bu kadar çok özmitelikle çalışmak algoritmanın eğitimini yavaşlator ve sonuç bulmaı zorlaitrırı
# temel bileşenlere ayırma ile öznitelik sayısını azaltmak mümkündür
# bu analiz boyut indirgemek için kullanılır aöa diğer yazdıklarım için de olur

# PCA yani temel bileşenler analizi, gözlenen değişkenlerdeki varyansı maximum düzeyde temsil edebilecek bileşenlerin bulunmasıdır

# bazı öznitelikler birbiriyle aşırı ilişkili olabilir
# büyün eğitim örnekler yüksek boyutlu uzaydan daha düşük boyutlu uzay içimde konumlananilir

import numpy as np

# rastgele sayı üreticisi oluşturalım ve içine 1 yazalım
# bu 1 yazmamızın sebebi rastgele sayı üretiminin tekrar edilebilir olmasını sağlar:
# yani kodu her çalıştırdığınızda aynı rastgele sayılar üretilir.
rng=np.random.RandomState(1)

# iki dizi üretelim ve sonra dot() kullanarak birbiti ile çarpalım
# X=np.dot(a.b)  -->  X=a[0]×b[0]+a[1]×b[1]
# rng.rand(2, 2) komutu, 2x2 boyutunda bir matris oluşturur. Bu matrisin elemanları, 0 ile 1 arasında rastgele sayılardır.
X=np.dot(rng.rand(2,2),rng.rand(2,200)).T

print(X.shape)
print()

# şimdi grafiğini çizelim
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1])
plt.show()
# grafiğe göre iki değişkeöznitelikn birbiriyle ilişkilidir

# lineer regresyon ile X değerşerinden y değerlerini tahmin edebiliriz
# ama buradaki dentimsiz öğrenme problemi X ile y arasındaki ilişkiyi öğrenmeye çalışmaktır
# bu ilişki verideki temel bileşenler eksemleri ile bulunur
# bu eksenler veri setini tanımlamak için kullanılır
# pca kullanarak bunu hesaplayalım

from sklearn.decomposition import PCA
# n_components=2 veriyi 2 boyuta(bileşene) indirgeyeceğimizi belirtir AMA:
# veri zaten 2 boyutlu(bileşenli) olduğu için trandform ile bir dönüştürme yapmamıza gerek yoktur
pca=PCA(n_components=2).fit(X)

# böylcece fit veriden bileşenler ve açıklanan varyans gibi şeyleri öğrendi
print(pca.components_) # bileşenler
print()
print(pca.explained_variance_) # açıklanan varyans

print()
# grafikte girdi verilerin üzerinde vektörler orarak bu değerleri görelim
# vektörün yönünü tanımlamak için bileşenler kullanılır
# vektörün uzunluğunu tanımlamak için açıklanan varyans kullanılır

# boyut indirgemek için pca(temel bileşemşer analizi) kullanılarak bir veya daha fazla en küçük temek bileşenler sıfırlanabilir
# veriyi düşük boyuta indirgeyerek maksimum düzeyde veri varyansı sağlanır
# boyut indirgemek için pca'dan örnek alalım:
# n_components=2 demekki bu sefer 2 boyutlu olan verimizş 1 boyuta indirgeyeceğiz
pca=PCA(n_components=1).fit(X)
# şimdi X verisini dönüştürelim (her veri noktası 2 boyuttan 1 boyuta düşürülür)
X_pca=pca.transform(X)

print(X.shape)
print(X_pca.shape)

print()

# şimdi boyut azaltmanın etkisine grafikte bakalım
# ömce azaltılan verinin ters dönüşümünü bulalım
X_yeni=pca.inverse_transform(X_pca)
# şimdi orjinal verinin grafiğini çizelim
plt.scatter(X[:,0],X[:,1],alpha=0.2)
# şimdi orjinal veri boyunca, boyut azaltılmış verinin de grafipini görelim
plt.scatter(X_yeni[:,0],X_yeni[:,1],alpha=0.8)
plt.show()
"""
Ters Dönüşüm (Inverse Transform): 
Ters dönüşüm, indirgenmiş veriyi orijinal özellik uzayına geri döndürür. 
Ancak, bu dönüşüm orijinal veri kaybını geri getirmez. 
Geri projeksiyon sırasında veri, yalnızca PCA tarafından öğrenilen ana bileşene dayalı olarak orijinal uzayda yeniden yapılandırılır. 
Bu nedenle, geri projekte edilen veri (2 boyutlu olan X_yeni), orijinal verinin bir tahminidir ve genellikle daha düz bir yapıdadır.

Eğer boyut indirgemiş veriyi orijinal boyutlarla (yani iki boyutlu haliyle) aynı grafik üzerinde görmek istiyorsanız, 
bu durumda geri dönüşüm yapmalısınız
"""


print("\n************\n")


# yüksek boyutlu bir veride boyut indirgeme yapalım
from sklearn.datasets import load_digits
digits=load_digits()
print(digits.data)
print()
# veri setinin yapısına bakalım
print(digits.data.shape) # 64 boyut var
print()

# veri setini anlamak için pca kullanarka boyut sayısını 2 yapalım
pca=PCA(2)
#hem fit hem transformu, fit_transform metodu ile aynı anda yapalım
data_pca=pca.fit_transform(digits.data)
print(data_pca.shape) # 2 boyuta düştüler


print("\n************\n")



# şimdi veriyi tanımlamak için kaç tane bileşenin gerektipine bakalım
# bu, birikimli açıklanan varyans oranına bakılarak belirlenebilir
pca=PCA().fit(digits.data)

# açıklanan varyans oranının grafiğini çizdirelim
# bunu kümülativ açıklanan varyans grafiği olarak görcez
# mesela grafiğe bakarsan:
# ilk 10 bileşen varyansın aklaşık %75 açıklıyor
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()
# grafikte gördüğünüz gibi veri setini iki boyuta indirgmek çok bilgi kaydbına yol açıyor
# mesela varyansın %90 açıklamak için yaklaşık 290 bileşene ihtiyaç vardır
# yani bunun gibi yüksek boyuttan düşük boyuta indirgenmiş veri setleri için bu grafik ihtiyaç fazlası boyutları anlamaua yardımcı olur


print("\n************\n")



# noise filtering
# temek bileşenler analizi aynı zamanda noisy yani parazit veriyi filtrekemek için de kullanlır
# bunun anlamı parazitin etkisinden çok daha büyük varyanslı herhangi bir bileşen kısmen parazidtten etkilenmez
# temek bileşenlerin en büyük alt kümesini kullanarak veri tekrardan oluşturulursa parazit filtrelenebilir

# şimdi yine digits veri setini düşünelim
# önce parazitli veri oluşturmak için verii setine biraz parazit ekleyelim
# yani sayıkar daha bulanıl ve sayıların etrafında daha çok anlamsız siyahlıklar olsun

# şimdi bu diziden varyansı en çok açıklayan 12 bileşeni alalım ve grafipini çizdirelim
# yani pca yapalım
# böylece veriyi tekrar parazitlerden arındırıp düzgün vir veri elde ettik

