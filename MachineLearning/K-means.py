# kümeleme  veri seti içindeki benzer ilişkileri ve patternleri bulur ve benzer özellikleri taşıyan verileri gruplar
# etikltler bilinmez bu yüzden unsüpervizeddir
# k-means kümelere ayırır ve küme merkezlerini bulur
# ilk önce ayrılacak küme sayısı kadar nokta belirlenir (bu noktalar merkez kabul edilir)
# bu merkese en yakın noktaların ortalaması bulunur
# bulunnan ortalama merkez kabul edilerek küme merkezleri güncellenir
# sonra tekrar bu merkezlere yakın noktalar bulunur ve bunkarın ortalaması bulunur
# bu ortalamalar merkez kabul edilip tekrar küme merkelerş bulunur
# bu adımlar küme merkezleri artık değişmeyene kadar devam eder

import matplotlib.pyplot as plt

# bu adımları örnek üzerş den görelim
# mglearn makine öğrenmesini öğrenmek için yapılmış bir kütüphanedir
import mglearn
mglearn.plots.plot_kmeans_algorithm()
plt.show() # küme merkezleri üçgen ile, veri noktaları daire ile gösterilir

mglearn.plots.plot_kmeans_boundaries()
plt.show() # kümelerin karar sınırlarını gördük
# yeni veri noktalarının herbirini k-means algoritması en yakın kümeye atayacak



print("\n*****************\n")




from sklearn.datasets._samples_generator import make_blobs

# n_samples : 300 veri noktası oluşturulacak (300 örnek)

# centers=4: Kaç tane küme (cluster) oluşturulacağını belirtir.

# cluster_std=0.60: Her bir kümenin standart sapmasını (yayılımını) belirtir.
# Bu değer, kümelerin ne kadar yayılacağını kontrol eder.
# Daha düşük bir değer, kümelerin birbirine daha yakın ve sıkı olmasını sağlar;
# daha yüksek bir değer, daha geniş ve dağınık kümeler oluşturur.

# random_state=0: Aynı kod her çalıştırıldığında aynı sonuçların elde edilmesini sağlar.
X,y_gercek=make_blobs(n_samples=300,centers=4,cluster_std=0.60,random_state=0)

# şimdi bu bölgelerin grafikleirni çizdirelim
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1],s=50)
plt.show() #şekilde 4 küme oldu

#kmeans algpritması bu 4 kümeyi otomatik bulur
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4).fit(X)
y_kmeans=kmeans.predict(X)

# şimdi bu kümelerin grafiğine bakalım
# yani her bir kümeyi ayrı ayrı renklendşrelim
plt.scatter(X[:,0],X[:,1],s=50,c=y_kmeans,cmap="viridis")
plt.show()

plt.scatter(X[:,0],X[:,1],s=50,c=y_kmeans,cmap="viridis")
# ayrıca kmeans tahmincisiyle belirlenen küme merkezlerini de grafikte görelim:
centers=kmeans.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],c="black",s=200,alpha=0.5) # boyutu 200, görünürlüğü 0.5
plt.show()



print("\n*****************\n")




# SPECTİRAL CLUSTERİNG ALGORİTHM
# kmeans algoritması kümeler geometrik olarak komplex olduğu zaman etkisizdir
# yani kmeans tekniğini uygulamak için kümeler arasındaki sınıflar lineer olmalıdır3
# eğer sınıflar komplex ise bu algoritma veriyi kümelemekte başarısız olur
# bunu gösterelim örnekle

# n_samples=200: Üretilecek veri noktası sayısını belirtir. Burada, toplam 200 veri noktası oluşturulur.
# noise=0.5: Veriye eklenen gürültü seviyesini belirtir. Yüksek gürültü değeri, verilerin orijinal yarım ay şekillerine daha az bağlı olmasına ve daha dağınık olmasına neden olur.
from sklearn.datasets import make_moons
# n_clusters=2: Oluşturulacak küme sayısını belirtir. Burada, 2 küme oluşturulacaktır.
X,y=make_moons(200,noise=0.5,random_state=0)
# fit_predict(X): Bu metod, hem K-Means modelini X verisi üzerinde eğitir (fit) hem de bu veriyi kullanarak her bir veri noktası için küme etiketlerini (predict) döndürür.
labels=KMeans(2,random_state=0).fit_predict(X)
# labels: K-Means algoritması tarafından tahmin edilen küme etiketlerini içeren bir dizi. Bu etiketler, her bir veri noktasının hangi kümeye ait olduğunu gösterir (0 veya 1).

# şimdi grafiğini çizdirelim
plt.scatter(X[:,0],X[:,1],s=50,c=labels,cmap="viridis")
plt.show()

# görüldüğü gibi karar sınıfrı lineer olmadığı için başarısız oldu
# kernel dönüşüm yaparsan yüksek boyutl verileri yansıtabilirsim böylece linner dönüşüm mümkün olur
# yani linner olmayan sınırlar içn KERNEL dönüşüm yap
# bunun için kmeans algoritmasının bir tahmincisi olam SpectralClusteeirng tahmincisi uygulanır
# bu tahminci verinin yükse boyutlu temsilini hesaplamak için en yakın komşuların graohını kulanır
# daha sonra kmeans algoritmasını kullanarak etiketlere atar
# bunu gösterelim:
# SpectralClustering, grafik tabanlı bir kümeleme algoritmasıdır ve
# veri noktalarını bir grafiğin düğümleri olarak düşünerek kümelere ayırır.
# Bu algoritma, genellikle karmaşık ve doğrusal olmayan kümeleri daha iyi tanımlamak için kullanılır.
from sklearn.cluster import SpectralClustering
model=SpectralClustering(n_clusters=2, # 2 küme oluşturulacaktır
                         affinity="nearest_neighbors",
                         assign_labels="kmeans")
"""
affinity="nearest_neighbors":
Kullanılacak benzerlik ölçütünü (affinity) belirler. 
nearest_neighbors seçeneği, her bir veri noktası için en yakın komşularına göre bir benzerlik matrisi oluşturur.
Bu, verinin yerel yapısını yakalamak için kullanılır.
En yakın komşularına dayalı bir benzerlik matrisi oluşturmak, özellikle karmaşık ve doğrusal olmayan veri kümeleri için etkilidir.

assign_labels="kmeans":
Kümeleri atamak için hangi yöntemin kullanılacağını belirtir. 
kmeans seçeneği, spektral kümelerin belirlenmesi için K-Means algoritmasını kullanır.
Alternatif olarak, discretize yöntemi de kullanılabilir. 
kmeans genellikle daha hızlıdır ve çoğu durumda yeterince iyi çalışır.
"""

# şimdi oluşturduğumuz modele göre X verilerini tahmin edip etiketlerini atayalım
labels=model.fit_predict(X)

# grafipini çizelim
plt.scatter(X[:,0],X[:,1],s=50,c=labels,cmap="viridis")
plt.show()




print("\n*****************\n")




# kmeans algoritması kümöelerin merkezi bulununcaya kadar devam eder
# örneklem sayısı büyük ise bu tekrarlardan solayı algoritma yavaş çalışır
# algoritmayı hızlı çalıştırmak için MiniBatchKMeans kullanılabilir
# bu teknin KMeans algoritmasının çeşididir ve hesaplama zamanını azaltmak için mini batchler kullanılır
# yani herbir adımda küme merkezleri güncellenirken verinin bir alt kümesini kullanır
# mini matchler girdi verisinin alt kümeleridir

# şinmdi bir resimdeki baskın renkleri göstermek için bir algoritma yazalım
from sklearn.datasets import load_sample_image
china=load_sample_image("china.jpg")

# resmin eksenlerini gösterelim
ax=plt.axes(xticks=[],yticks=[])
ax.imshow(china)
plt.show()

# değişkenin yapısına bakalım
print(china.shape)
# resşm 3 boyuttan oluşuyor
# birinci boyut yğksekliği, ikinci boyut genişliği, üçüncü boyut da kırmızı mavi ve yeşilin tonlarını gösteren 0'dam 255'e kadar tam sayıları içeriyor

# ilk olaarak makine öğrenmesi algoritmalarını kullanmak için verinin 3 boyutlu yapısını 2 boyuta indirgeyelim
# bunun için öncelikle renkleri 0 ile 1 arasonda göstermek için değişkeni 255'e bölelim
veri=china/255

# verinin boyutlarını 2 boyuta çevirelim
veri=veri.reshape(427*640,3)
print(veri.reshape)

# resimdeki yaklaşık 16 milyon rengi kmeans kümeleme algoritmasını kullanarak 16 renge indirgeyelim
# veri seti çok büyük olduğundan mini-batch kullanalım
# bu algoritme sonuçları çok daha hızlı hesaplamak için verinin alt kümelerinde çalışacak
from sklearn.cluster import MiniBatchKMeans

# n_clusters=16: Küme sayısını belirtir. Bu durumda, renk sayısını 16'ya indirgemek için kullanılır.
kmeans=MiniBatchKMeans(16)

# kmeans.fit(veri): Model, veri üzerinde eğitilir.
# Burada, her bir piksel rengi (RGB) bir veri noktası olarak kabul edilir ve K-Means algoritması bu noktaları 16 küme etrafında gruplayarak veri noktalarını en yakın küme merkezine atar.
kmeans.fit(veri)

# böylece verideki baskın renkler ortaya çıkartıldı
ax.imshow(china)
plt.show()
"""
Bu kod, bir resmi yükleyip görüntülemek ve ardından bu resimdeki renklerin sayısını K-Means Kümeleme algoritması kullanarak 16 renge indirgemek için kullanılmaktadır. 
İşlem, resmi 2 boyutlu bir veri setine dönüştürerek, her bir pikselin rengini bir veri noktası olarak ele alır ve daha sonra bu verileri K-Means ile kümeleyerek renkleri azaltır. 
MiniBatchKMeans algoritması, büyük veri setleriyle çalışırken performansı artırmak için kullanılır.
"""
