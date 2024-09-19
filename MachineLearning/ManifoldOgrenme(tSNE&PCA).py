# boyut indirgeme ve çok boyutlu veri setlerini görselleştirme gibi görevler için PCA kullanlır ama
# lineer olmayan yöntemler için PCA yerine bu manifold öğrenme kullanılmalıdır

# manifoldı sayfanın yüzeyü gibi düşünebilirsiniz
# çok boyutlu veriyi iki boyuta çevirirsek bir sayfa düzlemi gibi olur

# manifold algpritmalaro genelde veriyi görselleştirmek için kullanılırlar
# bu sebeple bu algoritmalar 22den faha fazka öznitelik nadiren oluştururlar

# t-SNE manifold öğrenme tekniği
# t-SNE veir noktaları arasındaki mesafeyi mümkün olduğunda en iyi şekilde koruyan verinin iki boyutlu temsilini bulmaktır
from sklearn.datasets import load_digits
digits=load_digits()

# önce pca ile iki boyuta indirgeyelim
from sklearn.decomposition import PCA
pca=PCA(n_components=2).fit(digits.data)
digits_PCA=pca.transform(digits.data)
print(digits_PCA)

# şimdi grafik çizelim
import matplotlib.pyplot as plt

# grafik nesnesi oluşturalım
plt.figure(figsize=(10,10))

# grafiği daha iyi görmek için veri setindeki sınıfları renklendirelim
renkler=["#476A2A","#7851B8","#BD3430","#4A2D4E","#875525","#A83683","#4E655E","#853541","#3A3120","#535D8E"]

# x eklsenini ve y eksenininin eksen sınırlarını ayarlayalım
# plt.xlim() ve plt.ylim() işlevleri, x ve y eksenlerinin minimum ve maksimum değerlerini belirleyerek, grafikte hangi aralığın görüneceğini kontrol eder.
plt.xlim(digits_PCA[:,0].min(),digits_PCA[:,0].max())
plt.ylim(digits_PCA[:,0].min(),digits_PCA[:,0].max())

"""
for döngüsü ile veri noktalarını görselleştirelim
len(digits.data), veri setindeki toplam örnek sayısını döndürür

plt.text() Fonksiyonu:
Bu fonksiyon, belirli bir noktada metin yazdırmak için kullanılır. 
Burada, matplotlib kütüphanesinin pyplot modülünden text() fonksiyonu kullanılarak PCA sonucu elde edilen iki boyutlu düzlemde her bir örneğin hedef etiketinin (sınıfının) bir görselleştirmesi yapılır.

digits_PCA[i, 0] x-koordinatını, digits_PCA[i, 1] ise y-koordinatını temsil eder. 
Yani, her veri noktasının x ve y konumunu belirler.

str(digits.target[i]): Görselleştirme üzerine yazılacak metin. 
digits.target[i], i. örneğin sınıf etiketini (0-9 arası rakamlar) verir. 
str() kullanılarak bu etiket bir metne dönüştürülür. 
"""

for i in range(len(digits.data)):
    plt.text(digits_PCA[i,0],digits_PCA[i,1],
             str(digits.target[i]),
             color=renkler[digits.target[i]],
             fontdict={"weight":"bold","size":9})
plt.show()



print("\n************\n")



# şimdi bu veri setini TSNE ile görselleşrirelim
from sklearn.manifold import TSNE
tsne=TSNE(random_state=42)
# TSNE yeni veriye dönüşüm yapmadığı için transform() metodu yoktur direk fit_transform() metodu vardır
digits_TSNE=tsne.fit_transform(digits.data)
print(digits_TSNE)

plt.figure(figsize=(10,10))
renkler=["#476A2A","#7851B8","#BD3430","#4A2D4E","#875525","#A83683","#4E655E","#853541","#3A3120","#535D8E"]
plt.xlim(digits_TSNE[:,0].min(),digits_TSNE[:,0].max())
plt.ylim(digits_TSNE[:,0].min(),digits_TSNE[:,0].max())
for i in range(len(digits.data)):
    plt.text(digits_PCA[i,0],digits_TSNE[i,1],
             str(digits.target[i]),
             color=renkler[digits.target[i]],
             fontdict={"weight":"bold","size":9})
plt.show()

# AMA MANİFOLD TEKNİĞİ EKSİK VERİLERİN VARLIĞINDA İYİ BİR YÖNTEM DEĞİLDİR
# BU YÜZDEN VERİ SETİNİ ARAŞTIRIRKEN ÖNCE PCA DENENİR EĞER OLMAZSA MANİFOLD YAPILIR