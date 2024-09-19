# sınıflandırma regresyon ve aykırı değerleri bulur
# el yazısı tanıma zaman serisi analizi ve konuşma tanımada kullanılır
# Bu derste şunları görücez
# Lineer destek vektör makineleri, Kernel estek vektör makineleri, Hiperparametre ayarı

# her bir sınıfı diğer sınıflandar ayıraak için doğru, eğri ya da manifold kullanıcaz

import numpy as np
import pandas as pd

from sklearn.datasets._samples_generator import make_blobs
# 50 veri noktası oluştır n_camples=50
# 2 merkez(küme) oluştur veri nnoktalaru bu iki merkez etrafına toplanır yani iki farklı bölge olucak centers=2
# sonuçların tekrarlanabilir olması için rastgelelik kontrolü sağla random_state=0
# kümelerin standart sapmasını ayarla (değer arttıkça kümeler daha geniş olur  yani iki sınıf arasındaki noktalar birbirine yaklaşır = cluster_std
X,y=make_blobs(n_samples=50,centers=2,random_state=0,cluster_std=0.60)

import matplotlib.pyplot as plt
# saçılım grafiği çizelim
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="autumn")
plt.show()

print()

# grafikteki iki ayrı sınıf noktalarını lineer bir doğru ile birbirinden ayıralım
# hangi deoğrunun en iyi ayırmayı yaptuğunı bulmak için destek vektör makineleri bulunur
# sınıfların en yakın noktalarına göre herbir doğrunun margin yani sınıf çizgileri çizilir
# birsürü bu sınıfları ayıran doğrı çizilir ve sonra
# destek vektör makinelerinde bu margini en doğru olan modelin optimumu çizili. en son tek bir en doğru margin bulubur
# şimdi modelimizde destek vektör sınıflandırıcısını(doğruyu) bulalım
from sklearn.svm import SVC
# kernel=linear :  verilerin lineer (doğrusal) bir çizgiyle ayrılacağını ifade eder. Yani, verileri sınıflandırmak için bir doğru kullanılır.
# C=1E10 : Çok yüksek bir C değeridir, modelin veriyi çok sıkı şekilde ayırmasını sağlar ve bu durumda hatalara çok az izin verilir.
model=SVC(kernel="linear",C=1E10)
model.fit(X,y)

# modeli grafikte görelim
# kesik çizgiyl gösterilen marginler ile en iyi doğru çizildi
# çember içindeki noktalar destek vektör noktalardır
# veri setimi ayıran en iyi doğru SADECE destek vektör noktalarına göre çizilir

# Veri noktalarını çiz
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# X[:, 0]: İlk sütun, yani ilk özellik; x eksenindeki koordinatlar.
# X[:, 1]: İkinci sütun, yani ikinci özellik; y eksenindeki koordinatlar.
# c=y: Veri noktalarının renklerini sınıf etiketlerine (y) göre belirler.
# s=50: Veri noktalarının boyutunu belirler.
# cmap='autumn': Renk haritasını belirler; burada 'autumn' renk paleti kullanılır.

# mevcut ekseni (current axis) döndürür. ax değişkeni, grafiğin sınırlarını ayarlamak ve üzerine eklemeler yapmak için kullanılır.
ax = plt.gca()

# Grafiğin sınırlarını al
# ax.get_xlim() ve ax.get_ylim() fonksiyonları, x ve y eksenlerinin mevcut sınırlarını döndürür. Bu sınırlar, daha sonra karar sınırını ve margin'leri çizmek için kullanılır.
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Karar sınırı için meshgrid oluştur
# np.linspace() fonksiyonu, belirtilen aralıklar arasında eşit aralıklı 30 nokta oluşturur.
# xx: x ekseni için aralık.
# yy: y ekseni için aralık.
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)

# np.meshgrid() fonksiyonu, yukarıda oluşturulan xx ve yy vektörlerini kullanarak bir ızgara (meshgrid) oluşturur.
# Bu ızgara, karar sınırını ve margin'leri çizmek için kullanılır.
YY, XX = np.meshgrid(yy, xx)
# XX ve YY, np.meshgrid() fonksiyonu ile oluşturulan iki boyutlu ızgara matrisleridir. Bu matrisler, x ve y eksenlerinde farklı değerleri temsil eden noktalardan oluşur.
# Örneğin:
# XX matrisi, x koordinatları için bir ızgaradır.
# YY matrisi, y koordinatları için bir ızgaradır.

# ravel() Fonksiyonu: XX ve YY gibi çok boyutlu dizileri tek boyutlu bir diziye dönüştürür.
# Örneğin:
# XX.ravel() x koordinatlarının tek boyutlu bir listesi (vektörü) olur.
# YY.ravel() ise y koordinatlarının tek boyutlu bir listesi (vektörü) olur.
# Bu şekilde, ızgaradaki her noktayı (x, y) çiftleri olarak temsil edecek duruma getiririz.
xy = np.vstack([XX.ravel(), YY.ravel()]).T
# np.vstack([XX.ravel(), YY.ravel()]) Nedir?
# np.vstack() Fonksiyonu: İki veya daha fazla diziyi dikey olarak (vertically stack) istifler (üst üste koyar). Burada XX.ravel() ve YY.ravel() tek boyutlu dizilerini dikey olarak birleştirir.
# Bu işlem sonucu, iki satırdan oluşan bir 2D (iki boyutlu) dizi elde ederiz:
# Birinci satır, x koordinatlarının düzleştirilmiş dizisini içerir.
# İkinci satır, y koordinatlarının düzleştirilmiş dizisini içerir.
# .T (Transpose) Nedir?
# .T Operatörü: Matrisin transpozunu (dönüşümünü) alır. Bu, satırları sütunlara ve sütunları satırlara çevirir.
# Sonuç olarak, np.vstack([XX.ravel(), YY.ravel()]).T işlemi, her (x, y) çiftini bir sütun olarak temsil eden bir 2D dizi oluşturur.
# xy Nedir?
# Sonuçta xy, ızgaradaki her bir noktanın (x, y) koordinat çiftini içeren bir 2D dizi olur. Bu dizi, karar fonksiyonu hesaplamaları için kullanılır.

# Karar fonksiyonunu hesaplama:
# model.decision_function(xy) fonksiyonu, her bir grid noktası için karar fonksiyonunun değerini hesaplar.
# Sonuçlar, ızgara şeklinde çizilebilmesi için XX ile aynı şekle dönüştürülür.
Z = model.decision_function(xy).reshape(XX.shape)

# Karar sınırını ve margin'leri çiz : ax.contour() fonksiyonu, karar sınırlarını ve margin'leri çizer.
# XX, YY, Z: Grid noktaları ve hesaplanan karar fonksiyonu değerleri.
# colors='k': Çizgilerin rengi siyah ('k') olarak ayarlanır.
# levels=[-1, 0, 1]: Üç seviye belirlenir: -1 ve 1, margin'ler için; 0, karar sınırı için.
# alpha=0.5: Çizgilerin şeffaflık düzeyi.
# linestyles=['--', '-', '--']: -1 ve 1 seviyeleri için kesik çizgi, 0 seviyesi için düz çizgi kullanılır.
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# Destek vektörlerini çiz : ax.scatter() fonksiyonu, SVM tarafından belirlenen destek vektörlerini çizer.
# model.support_vectors_: Destek vektörlerinin koordinatları.
# s=100: Destek vektörlerinin boyutu.
# linewidth=1: Kenar çizgisinin kalınlığı.
# facecolors='none': Doldurulmamış işaretleyiciler (boş daireler).
# edgecolors='k': Kenar rengi siyah ('k').
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

plt.show()



print("\n-***************-\n")



# lineer destek vektör makineleri birçok durumda iyi çalışmasına rağmen çoğu gerçek veri lineer bir doğruyla ayrılamaz
# destek vektör makineleri ayrıca kernel ile birleşince çok güçlü olurlar

# make_circles Fonksiyonu: Makine öğrenmesi algoritmalarını test etmek ve görselleştirmek için kullanılan örnek veri kümeleri oluşturur.
# Bu fonksiyon, 2 boyutlu dairesel bir veri seti oluşturur.
from sklearn.datasets._samples_generator import make_circles

# girdi ve çıkıktı verilerini alalım
# 100 veri noktası olsın
# factor değeri küçük kullanıldığu için iç daire dış daireye göre daha küçük yapıldı
# Veriye rastgele gürültü ekler. Gürültü, veri noktalarının dairelerin üzerinde tam olarak değil, etrafında biraz dağılmasına neden olur.
X,y=make_circles(100,factor=.1,noise=.1)

# kernelin lineer olduğıu bir model alalım
clf_linear=SVC(kernel="linear").fit(X,y)
# ama veri seti liner bir doğruyla sınıflandırılamadı
# bu veri setini 3 boyutta gösterirsek lineer sınıflama yapabiliriz
# 3 boyutlu elde etmek için bir öznitelik seçmemiz gerek
# kernel_trick denilen matematiksel birşey kullanacağız
# bununla yüksek bpyutlu uzayda bir sınıflama öğrenilebilir:
clf_rbf=SVC(kernel="rbf",C=1E6,gamma="auto").fit(X,y)


# 2D SVM karar sınırını çizmek için fonksiyon
def plot_svm_boundary(clf, X, y, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    # Karar sınırı için meshgrid oluştur
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    # Karar sınırını ve margin'leri çiz
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.title(title)
    plt.show()

# 3D SVM karar sınırını çizmek için fonksiyon
def plot_svm_3d(clf, X, y, title):
    # 3D çizim için yeni bir figür oluştur
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Veriye üçüncü bir boyut ekleyelim (RBF çekirdek için)
    # Bu genellikle bir fonksiyon olarak seçilebilir; burada r^2 kullanılır
    r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)  # radius (distance to origin)
    # Veri noktalarını 3D olarak çiz
    ax.scatter(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    # 3D grid oluştur
    xx = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 30)
    yy = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 30)
    YY, XX = np.meshgrid(yy, xx)
    ZZ = np.sqrt(XX ** 2 + YY ** 2)
    # Karar fonksiyonunu hesapla
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    # Karar sınırını çiz
    ax.plot_surface(XX, YY, Z, color='k', alpha=0.3)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Radius (r)')
    plt.title(title)
    plt.show()

# Lineer kernel ile oluşturulan modeli görselleştir
plot_svm_boundary(clf_linear, X, y, "Lineer Kernel ile SVM")
# RBF kernel ile oluşturulan modeli 3 boyutlu görselleştir
plot_svm_3d(clf_rbf, X, y, "RBF Kernel ile 3D SVM")

# kernel stratejisi lineer olmayan metortları lineer yapmak için kullanılır



print("\n-***************-\n")



# veri setinde sınıfların arasındaki veri noktaşar9 birbirine çok yakın plabilir
# ilk grafiği yine çizdirelim
# # kümelerin standart sapmasını ayarla (değer arttıkça kümeler daha geniş olur yani iki sınıf arasındaki noktalar birbirine yaklaşır = cluster_std
X,y=make_blobs(n_samples=50,centers=2,random_state=0,cluster_std=1.2)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="autumn")
plt.show()

# veri noktaları arasındaki mesafe gaussian kernel ile ölçülür
# sınıfları ayıran çizgi ya da dairenin etrafındaki çizgiler c ve gama ile kontrıl edilir
# c parametresi regülerleştirmeyi kontrol eder
# gamma parametresi kenrellin genişliğini kontrol eder




print("\n-***************-\n")



# kanser veri setine bakalım
from sklearn.datasets import load_breast_cancer
kanser=load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(kanser.data,kanser.target,random_state=0)
svc=SVC(gamma="auto").fit(X_train,y_train)

# şimdi modelin eğitim ve test verileri için doğruluklarını bulalok
print(svc.score(X_train,y_train))
print(svc.score(X_test,y_test))
# görüldüğü gibi model overfit olmuş

print()

# bütün öznitelikleri aynı ölçek ile ölçekleyelim:
# bunun için eğitim setindeki herbir özelliğin(sütunun) min değerlerini bulalım
min_on_training=X_train.min(axis=0)

# şimdi herbir eğitim setindeki öznitelik aralığını hespalayalım
# Bu işlem, her bir özelliğin veri aralığını (maksimum - minimum) verir.
range_on_training=(X_train-min_on_training).max(axis=0)

# herbir eğitim veisinden minimum değeri çıkarıp aralığa bölelim
X_train_scaled=(X_train-min_on_training)/range_on_training

# test verisi için de yapalım:
X_test_scaled=(X_test-min_on_training)/range_on_training

"""
Bu işleme min-max normalization (min-max ölçeklendirme) veya min-max scaling (min-max ölçekleme) denir.
Min-Max Ölçeklendirme: bir veri setindeki her bir özelliği belirli bir aralığa (genellikle 0 ile 1 arasına) dönüştürmek için kullanılan bir normalizasyon tekniğidir. 
Bu yöntem, her bir özelliği kendi minimum ve maksimum değerlerine göre yeniden ölçeklendirir.
"""

# modeli tekrar kuralım ve doğruluk oranlarına bakalım
svc=SVC(gamma="auto").fit(X_train_scaled,y_train)
# şimdi modelin eğitim ve test verileri için doğruluklarını bulalok
print(svc.score(X_train_scaled,y_train))
print(svc.score(X_test_scaled,y_test))
# şimdi eğitim test oraanları yakın oldu ama şimdi de underfitting oldu


# bu problemin üstesinden gelmmek için c ve gamma güzenleyelim
# modelin daha komplex olması için c parametresini arttıraım
svc=SVC(C=100,gamma="auto").fit(X_train_scaled,y_train)
# şimdi modelin eğitim ve test verileri için doğruluklarını bulalok
print(svc.score(X_train_scaled,y_train))
print(svc.score(X_test_scaled,y_test))

# çok iyi oldu