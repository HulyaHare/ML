"""
3 çeşit naive bayes sınıflandırma çeşidi vardur:
Gaussian  Multinomial  Bernoulli

Gausian sürekli veriler için kullanılır
Multinominal çok kategorili veriler için kullanılr
Bernoulli ikili sınıflar için kullanılr

Bernoilli ve Multinominal genelde text verilerini sınıflandırmak için kullanılır
"""

# GAUSSİAN NAİVE BAYES
# özniteliklerin normal dağıldığı varsayılır
# normal dağıldıkları için bu modeldeki tüm öznitelikler sürekli olmalıdır
# modlin basitliği, hesaplama kolaylığı, sınıflandırma için iyi performansı binlerce öznitelik ile başa çıkması modelin avantajıdır

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import make_blobs

# girdi çıktıları alalım
X,y=make_blobs(100, #veri setinden 100 tane örnek veri noktası aldık
               2, # her örnek 2 boyutlu olcak
                centers=2, # 2 farklı küme yani sınıf olcak
               cluster_std=1.5) # Bu parametre, kümelerin standart sapmasını (dağılım genişliğini) belirtir. Burada 1.5 olarak ayarlanmış, yani her bir kümenin genişliği, merkezine olan uzaklıklarının ortalamasına göre değişecektir. Daha yüksek bir cluster_std değeri, veri noktalarının kümelerinin daha geniş dağılmasına ve kümelerin örtüşmesine neden olur.
# X (100,2) boyutunda olcak çünkü 100 örnek ve her örneğin 2 özelliği var
# y (100,) boyutunda olcak ve her veri noktasının (örneğin) hangi kümeye (sınıfa) ait olduğunu gösteren etiketleri içeren bir numpy dizisidir.

# saçılım grafiğine bakalım
# renklendirme için c=y olsun ve noktaların boyutu için s=50 olsun
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.title('make_blobs ile Oluşturulan Verilerin Görselleştirilmesi')
plt.show()

#herbir sınıftaki noktaların ortalaması ve standart sapması bulunarak bir model kurabiliriz
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X,y)

# şimdi rastgele veriler üretip bu verilerin sınıflarını tahmin edelim
rng=np.random.RandomState(0)
# yeni veriler oluşturalım
X_yeni=[-6,-14]+[14,18]*rng.rand(1000,2)
y_yeni=model.predict(X_yeni) # yeni veriler kurulan modele göre tahmin edildi

# şiödi yeni verilerin grafikte nereye düştüğümne bakalım
# plt.axis(): Grafik eksenlerinin sınırlarını verir. Bu komut ile eksen sınırları lim değişkenine atanır, böylece sonraki grafikte aynı eksen sınırları kullanılabilir.
# Bu, yeni verileri aynı ölçek ve görünümde göstermek için kullanılır.
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim=plt.axis()
plt.scatter(X_yeni[:, 0], X_yeni[:, 1], c=y_yeni, s=20, cmap='RdBu',alpha=0.2)
plt.axis(lim)
plt.xlabel('Özellik 1')
plt.ylabel('Özellik 2')
plt.title('make_blobs ile Oluşturulan Verilerin Görselleştirilmesi')
plt.show()



print("\n*************\n")



# SSL doğrulamasını devre dışı bırakmak için
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ÇOKLU NAİVE BAYES
#bu modelde özniteliklerin çok kategorili dağıldığı varsayılır
# eğer öznitelikler text'teki kelimeleirn sayısı ile ilişkili ise bu model tahmin edilir
from sklearn.datasets import fetch_20newsgroups
data=fetch_20newsgroups

# hedef değişkenindeki kateogrilerden birkaö tanesini seçip kategorilere atayalım
kategoriler=["talk.religion.misc","soc.religion.christian","sci.space","comp.graphics"]

# eğitim ve test veirleirni oluşturalım
train=fetch_20newsgroups(subset="train",categories=kategoriler)
test=fetch_20newsgroups(subset="test",categories=kategoriler)

# eğitim verisindeki beşinci texti görelim
print(train.data[5])


print("\n*************\n")


# makina öğrenmesinde bu veri setini kulanmak için her bir stringin içeriğini sayısal vektöre çevirmeliyiz
# bunun için TfidfVectorizer kullanlır
from sklearn.feature_extraction.text import TfidfVectorizer

# şimdi çok kategorili naive bayes sınıfını import edelim
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model=make_pipeline(TfidfVectorizer(),MultinomialNB())

# modeli fit edelim
model.fit(train.data,train.target)

# test verisimdeki değerleri tahmin edelim
etiketler=model.predict(test.data)

# şimdi modelin performansını değerlendirelim
# test verileri için gerçek ve tahmini etiketler arasındaki confusion matrise bakalım
from sklearn.metrics import confusion_matrix
mat=confusion_matrix(test.target,etiketler)

# gerçek ve tahmib değerlerin grafiğine bakalım
# mat: Karışıklık matrisi (confusion matrix), modelin gerçek ve tahmin edilen sınıfları karşılaştıran bir tablo.
# mat.T: mat'in transpozu; satır ve sütunları yer değiştirir.
# square=True: Hücreleri kare şeklinde yapar.
# annot=True: Hücrelere sayı ekler (annotate).
# fmt="d": Hücre içindeki değerleri tam sayı olarak gösterir.
# cbar=False: Renk çubuğunu (color bar) gizler.
# xticklabels=train.target_names: X eksenine sınıf adlarını ekler.
# yticklabels=train.target_names: Y eksenine sınıf adlarını ekler.
# plt.xlabel("Gerçek Değerler"): X eksenine "Gerçek Değerler" etiketi ekler.
# plt.ylabel("Tahmin Etiketleri"): Y eksenine "Tahmin Etiketleri" etiketi ekler.
sns.heatmap(mat.T,
            square=True,
            annot=True,
            fmt="d",
            cbar=False,
            xticklabels=train.target_names,
            yticklabels=train.target_names)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin etiketlşeri")
plt.show()
# grafijteki numarakar o bökğmğn ne kadrının tajmin edikdiini söyler
# köşegeneler de doğru tahminleir verir

# görüldüğpü gibi hıristiyanlık ve misc hakkında çok fazla yanlış veri var model o ikisinde çok fazla hata yapmış


# şimdi predict metodunu kullanarak bir stringin içeriğini tahmin etmek için fonksiyon oluşturalım
# predict_category(s, train=train, model=model): Tahmin fonksiyonu; bir metin girdisi alır.
# model.predict([s]): Model, verilen metni kullanarak bir sınıf tahmini yapar.
# train.target_names[pred[0]]: Tahmin edilen sınıfın adını (kategori ismini) döndürür.
# train=train ve model=model ifadeleri, fonksiyonun varsayılan olarak mevcut train ve model değişkenlerini kullanmasını sağlar. Bu, kodun esnekliğini artırır ve fonksiyon çağrısını daha basit hale getirir. İsterseniz fonksiyon çağrılırken farklı bir train veya model değeri de verebilirsiniz.
def predict_category(s,train=train,model=model):
    pred=model.predict([s])
    return train.target_names[pred[0]]

# şimdi bir stringin sınfını tahmin etmeyi deneyelim
print(predict_category("discussing islam vs ateism"))
print()
print(predict_category("determining the screen resolution"))


# naive bayes karışık modeller için kullanılmaz ama çok hızlıdır