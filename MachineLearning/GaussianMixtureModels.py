# kmeans yöntemi basit ilişkili veri setleri içim nodel kurabilir
# ama gerçek dünya veri setleri için bu model yetersiz kalır
# bu probemi aşmak için gaussian mixture kullanılır
import matplotlib.pyplot as plt
# gaussian mixture modeller verinin kovaryansını kullanarak kmeans kümelemenin genişletilmesidir

# önce kmeans kümelemenin zayıf yanlarını gösterelim:
# bunun için önce örnek bir veri seti oluşturalım
from sklearn.datasets._samples_generator import make_blobs
# cluster_std=0.60: Her bir kümenin standart sapmasını (yayılımını) belirtir. Bu değer, kümelerin ne kadar yayılacağını kontrol eder. Daha düşük bir değer, kümelerin birbirine daha yakın ve sıkı olmasını sağlar; daha yüksek bir değer, daha geniş ve dağınık kümeler oluşturur.
X,y_true=make_blobs(n_samples=400,centers=4,cluster_std=0.60,random_state=0)

print(X[:5,:]) # ilk beş satırı görelim
print()
# girdi verisini daha iyi görselleştirmek için eksenleri yer değiştirelim
X=X[:,::-1] # her satırı sondan başa aldık
print(X[:5,:]) # ilk beş satırı görelim

print()

# modeli kuralum
from sklearn.cluster import KMeans
kmeans=KMeans(4,random_state=0) # 4 küme var 0,1,2,3 etiketleriyle
labels=kmeans.fit(X).predict(X)

print(labels[:30]) # labesl ilk 30 satırına bakalım

# şimdi grafiğini çizdirelim
plt.scatter(X[:,0],X[:,1],c=labels,s=40,cmap="viridis")
plt.show()

# bu verideki sınıflar birbirdnden iyi ayrıldığı için etiketleri bilinmemesine rağmen kmeans kümeleme ile iyi bir gruplama yapıldı
# ama iyi ayrılamasaydı iyi olmazdı

# kmeans kümelemenin iki dezavantajı vardır:
# birincisi kümeleme yapısı esnek değildir
# ikincisi ise olasılıksal bir model değildir

# bu dezavantajlardan dolayı bunun yerşne gaussian mixture kullanılmalıdır




print("\n**************\n")




# GAUSSİAN MİXTURE MODEL
# gaussian mixture, girdi veri seti için en iyi modeli bulmak için çok boyutlu olasılıksal dağılım kullanır
# aynı zamanda bu model basit kümeleme için de kullanılabilir

# az önceki veri seti için bir model kuralım
from sklearn.mixture import GaussianMixture
gm=GaussianMixture(n_components=4).fit(X)

# sınmıfları tahmin edelim
labels=gm.predict(X)

# grafiğini çizdirelim
plt.scatter(X[:,0],X[:,1],c=labels,s=40,cmap="viridis")
plt.show()

# kümelemelerin artama olasılıklarını da bulalım
probs=gm.predict_proba(X)
print(probs[:5].round(3)) # ilk beşine bakalım ve virgülden sonra 3 basamak olsun

# gaussian mixture model, kmeans'e benzer
# bu model expectation maximization yaklaşımını kullanır
# bu yaklaşımda önce lokasyon ve biçime göre bir tahmin yapılır vve bu tahmin sürekli tekrar eder

# verileri sınıflarken genelde ful kovaryans opsiyonu daha iyi performans gösterir




print("\n**************\n")




# GAUSSİAN MİXTURE AS DENSİTY ESTİMATİON:
# GMM, Expectation-Maximization (EM) algoritmasını kullanarak verinin olasılık dağılımını iteratif olarak öğrenir.
# İlk olarak, her bir veri noktasının belirli bir bileşene ait olma olasılıklarını hesaplar (Expectation Adımı).
# Daha sonra, bu olasılıkları kullanarak her bir bileşenin parametrelerini (ortalama, kovaryans) günceller (Maximization Adımı).
# Bu işlem, modelin log-olasılık değeri belirli bir eşik değere ulaştığında veya belirli bir iterasyon sayısı geçildiğinde sona erer.
# gaussian mixture moel aynı zamanda verinin olasılıksal dağılımını tanımlamak için de kullanılır
from sklearn.datasets import make_moons
X_moons,y_moons=make_moons(200,noise=0.05,random_state=0)

# grafipini çizdirelim
plt.scatter(X_moons[:,0],X_moons[:,1])
plt.show()

# kümeleme modeli olarak iki bileşeli gaussian mixture modelini kullanalım
# covariance_type="full":
# Her bir bileşenin kovaryans matrisinin türünü belirtir. Kovaryans matrisi, veri noktalarının yayılımını ve yönelimini gösterir.
# covariance_type için dört seçenek vardır:
# "full": Tam kovaryans matrisi kullanılır. Her bir bileşen için ayrı bir kovaryans matrisi hesaplanır ve verinin daha esnek bir şekilde modellenmesini sağlar. Bu, en genel ve güçlü yöntemdir.
gm2=GaussianMixture(n_components=2,covariance_type="full",random_state=0)
# eğer daha fazla n_component kullanırsak gşrdş verilerine daha yakın model kurabiliriz

# şimdi 16 komponentli moel kuralım
gm16=GaussianMixture(n_components=16,covariance_type="full",random_state=0)
# böylece girdi verilerine daha yakın oldu ama dikkat et n_componenti fazla arttırırsan overfitting olabilkir




print("\n**************\n")




# OPTİMAL NUMBER OF COMPONENTS
# gaussşan mixtire model ile veri setinin optimal bileşen sayısını tespit edebiliriz

# bunu göstermek için 1'den 21'e kadar olan sayıları n_component değişkenine atayalım
import numpy as np
n_components=np.arange(1,21)
models=[GaussianMixture(n,covariance_type="full",random_state=0).fit(X_moons)
        for n in n_components]
# şuan elimizde 20 model oldu


# şimdi herbir model için şunları görelim:

# m.bic(X_moons): Modelin BIC (Bayesian Information Criterion) skorunu hesaplar.
# BIC, modelin veri uyumunu ve karmaşıklığını dikkate alarak hesaplanan bir kriterdir;
# düşük BIC, daha iyi bir model anlamına gelir.
# label="BIC": Grafikteki bu çizgiye "BIC" etiketi verir. Bu, grafiğin açıklama kısmında (legend) görüntülenir.

# m.aic(X_moons): Modelin AIC skorunu hesaplar.
# AIC, modelin veriyle uyumunu ve parametre sayısını dikkate alarak hesaplanan bir kriterdir.
# Düşük AIC, daha iyi bir model anlamına gelir.
# label="AIC": Grafikteki bu çizgiye "AIC" etiketi verir.

plt.plot(n_components,[m.bic(X_moons) for m in models],label="BIC")
plt.plot(n_components,[m.aic(X_moons) for m in models],label="AIC")

# plt.legend(loc="best"): Grafiğin altına veya uygun bir yerine legend (açıklama kısmı) ekler.
# loc="best": En uygun konumu otomatik olarak seçer. Genellikle grafik içeriğine dayalı olarak en iyi yerleştirme yapılır.
plt.legend(loc="best")

plt.xlabel("n_components")
plt.show()

#yani hem bileşen sayısını bulur hem de kümeleme tahmincisi gibi çalışır




print("\n**************\n")




# uygulama
from sklearn.datasets import  load_digits
digits=load_digits()

# girdiye bakalım
print(digits.data.shape) # 1797 örneklem ve 64 boyutu var

# veri setinin boyutunu azaltalım
from sklearn.decomposition import PCA

# varyansın %99 kullanılcak şekilde boyut indirgeyelim
pca=PCA(0.99,whiten=True)
data=pca.fit_transform(digits.data)
print(data.shape) # verinin boyutu 1 oldu

# şimdi components satısıını hesaplayalım
# onar onar artan sayılar üretelim
n_components=np.arange(50,210,10) # 50'den başlayıp 210'a kadar gidiyor 10'ar 10'ar artıyor

# modelleri kuralım
models=[GaussianMixture(n,covariance_type="full",random_state=0)
        for n in n_components]

# şimdi model değişkeni iiçin bilgi kriterlerini hesaplayalım
aics=[model.fit(data).aic(data) for model in models]

# grafiğine bakalım
plt.plot(n_components,aics)
plt.show() # yani aic bilgi değerini minimize eden değer 110

# şimdi bu 110 değerine göre gaussian mixture kuralım
gm=GaussianMixture(110,covariance_type="full",random_state=0).fit(data)
# model kuruldu
# artık bu modeli bir yoğunluk dağılımı gibi kullanabiliriz
# bu dağılıma sahip yeni veriler üretebiliriz

data_new=gm.sample(100)
print(data_new) # digits veri setinin benzeirdir