# BİRLEŞTİRİCİ KÜMELEME:
# aynı pernsiplere dayanan kümeleme algoritmalarını bir araya getirir
# herbir birim başlangıçta küme kabul edilir
# bazı durma kriterleri sağşanıncaya kadar çok benzer kümeler bir araya getirilir
# kümeleme sayısı gibi değerler scşkştleannda durma kriteridir

# scikitlearnda çok benzer kümelerin tam olarak nasıl ölçüleceğini belirlemek için birkaç bağlantı kriteri vardur:
# birleştirici kümeleme tek bağlantı, ortalama bağlantı tam bağlantı ve rard stratejilerini destekler

# tek bağlantıya en yakın komşuluk da denir:
# bu yönteöde ilk önce en küçük uzaklığa sahşp iki gözlem birleştirilir
# sınra bir sonraki en küçük uzaklık bulunur
# ardından yeni gözlem ya ilk iki gözlemin oluşturduğu kümeye katılır ya da iki elemanlı başka yeni küme oluşturulur
# bu süreç tüm gözlemler için devam eder

# ortalama bağlantı:
# kümelerin bütün gözlem çiftleri arasında mesafenin ortalamasını minimize eder

# tam bağlantıya maximum mağlantı da denir en uzek komşuluk olarak de bilinir:
# tam bağlantı noktalar arasındaki en küçük maximum mesafeye sahip iki kümeyi birleştirir

# ward yöntemi:
# bu bir en küçük varyans yaklaşımıdır ve öntanımlı olarak seçilidir
# bütün kümelerin içindeki farkların kareleri toplamını minimize eder
# bu yaklaşım az gözlemli kümeleri birleştirme eğilimindedir
# ayrıca bu yaklaşım eşit sayıda gözlemden oluşan kümeler oluşturma gibi bir eğilimi de vardur
# bu nedenşe gözlem sayısı birbirine yakın kümeler oluşturmak için bu yöntem kullanılır

# örneğin bir kümenin üyeleri diğerlerşnden daha fazla ise ortalama veya tam bağlantı daha iyi çalışır

# örneklerde ward yaklaşımı kullancaz

import matplotlib.pyplot as plt

# algoritmanın nasıl çalıştığına bakalım
import mglearn
mglearn.plots.plot_agglomerative_algorithm()
plt.show() # her adımda en yakın kümeler birleşe birleşe gitti



print("\n***************\n")



# şimdi bir veri setinde birleştirici kümeyi inceleyelim:
from sklearn.cluster import AgglomerativeClustering # birleştirici kümeyi import ettik
from sklearn.datasets import make_blobs # kullanılacak veri setini import ettik
X,y=make_blobs(random_state=42)
agg=AgglomerativeClustering(n_clusters=3) # 3 küme olsun

# birleştirici kümeleme veri noktalarını tahmin etmez
# bu yüzden bunun tahmin etme yani predict() metofdu yojtur
# bunun yerine modeli kurup eğitmek ve eğitim setindedki veirleri kümelemek için fit_predict() kullanlır
k=agg.fit_predict(X)
print(k)

# şimdi veri setindeki noktaların grafipğine bakalım
# discrete_scatter fonksiyonu, 2 boyutlu bir dağılım grafiği oluşturur ve her bir küme farklı bir renk ve sembol ile gösterilir.
# k: Her bir veri noktasının ait olduğu kümeyi belirten tahmin edilen etiketlerin bulunduğu bir dizidir. Değerler, 0, 1, 2 gibi sayısal etiketlerdir ve hangi veri noktasının hangi kümeye ait olduğunu belirtir.
mglearn.discrete_scatter(X[:,0],X[:,1],k)
plt.show()



print("\n***************\n")



# HİYERARŞİKAL KÜMELEME
# benzer özniteiklere sahip nesnelerin adıma sdım bir aray getirilmesi veya tam tersine bir bütünden adım adım ayrılmasıdır
# hiyerarşikal kümeleme analizinde birleştirici ve bölücü olmak üzere 2 teml yaklaşım vardyr

# önce hşyerarşikal küme analizini görelim
mglearn.plots.plot_agglomerative()
plt.show() # görüldüpü gibi benzer özniteliği olanlar adım adım bir araya geldiler

print()

# eğer veri seti ikiden fazla boyutlu ise:
# # hiyerarşikal kümeleme analizinde kullanılan en etkili görselleştirme aracı dendrıgramdyr
# ama scikitlearn dendrogram çizemez ama SciPy kullanacak çizebiliriz
# ilk önce dendrogram ve ward yöntemini scipy'den import edelim
from scipy.cluster.hierarchy import dendrogram,ward
X,y=make_blobs(random_state=0,n_samples=12)
# X için ward kümeleme yapalım:
linkange_array=ward(X)

# scipy ward() fonksiyonu mesafeleri belirleyen bir dizi dönderir (dict yapısındadır)
# mesafeleri belirleyen linkage_array için bir dendrogram çizdirelim
kumeler=dendrogram(linkange_array)
for kume in kumeler:
    print(kume,":",kumeler[kume])
plt.show() # bu şekilde grafikte de ayrıca görelim



print("\n***************\n")



# DBSCAN:
# bu algoritma yopun bölgeleri bularak kümeleme yapar,
# seyrek bölgeleri ise parazit olarak algılar

# bu algoritmanın esas yararı BAŞLANGIÇTA KÜMELEME SAYISINI BELİRLEMEYE GEREK YOKTUR

# yoğun bölgeler içindeki noktalar core sample yani çekirdek örnek olarak adlandırılır
# bu algoritmanın min_samples ve eps yani yarıçap isminde iki parametresi vardır

# veri uzayında keyfi olarak bir nokta seçilir ve bu nokta merkez kabul edilir
# alınan yarıçap içerisinde kalan noktaların sayısı tanımlanan min_samples yani minimum nokta değerine eşit veya büyük ise bu bölge yoğun olarak nitelendirilir
# bu yoğun noktalar çekirdek örnek olarak sınıflandırılır
# başlangıç noktasını merkez kabul edip belirlenen yarıçap içerisinde kalan noktalar belirlediğimiz minimum nokta sayısından küçük ise parazit olarak etiketlenir
# bu başlangıç noktası herhangi bir kümeye ait değil anlamına gelir
# yarıçap içerisinde minimum nokra sayısından daha çok nokta varsa bu nokta çekirdek örnek olarak etiketlenir ve yeni bir küme etiketine atanır
# ardından yeni bir nokta seçilir ve sürece devam edilir
# parazit olarak etiketlenen bir nokta daha sonraki aşamalarda bir başka çekirdek örnek yarıçapı içerisinde bir kümenin üyesi olabilir

# kısaca burada üç nokta çeşidi vardır:
# bunlar çekirdek noktalar, çekirdek noktaların yarıçap mesafesinde kalan noktalar yani kenar noktalar, parazit noktalardır

# dbscan algoritması veri seti üzerinde birçok kez çalışır
#çekirdek noktaların kümeleri her zaman aynıdır
# fakar kenar noktalar, birden fazla çekirdek örneklerin komşusu olabilir

# şimdi veri seti üzerinde dbscan algorimasına bakalım
from sklearn.cluster import DBSCAN
X,y=make_blobs(random_state=0,n_samples=12)
dbscan=DBSCAN()

# dbscan algoritması yeni verilerde tahmin yapmaz bu yüzden fit_predict() metodu kulllanılmalıdır
kumeler=dbscan.fit_predict(X)
print(kumeler)
# bütün veri noktaları -1 etiketine atandı ki bu paraziti ifade eder
# bu durum yarıçap ve minimum nokta sayısının öntanımının bir sonucudur

# şimdi veri dönüşümünü kullanarak dbscan algoritmasının perfırmansına bakalım
from sklearn.datasets import make_moons
X,y=make_moons(n_samples=200,noise=0.06,random_state=42)

# şimdi veriyi standartlaştıralım
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit()
X_scaler=scaler.transform(X)
dbscan=DBSCAN()
kumeler=dbscan.fit_predict(X_scaler)

# şimdi veriyi görselleştirelim
import matplotlib.pyplot as plt
plt.scatter(X_scaler[:,0],X_scaler[:,1],
            c=kumeler, # her veri noktasının renginin hangi kümeye ait olduğunu belirtir. kümeler de 0,1,2 gibi veri noktalarını kümelere göre temsil eder
            cmap=mglearn.cm2, # mglearn kütüohanesinde tanımlaanan özel bir renk haritasıdır
            s=60) # noktaların büyüklüğü 60 olur
plt.show()

# yarıçapın öntanımlı değeri eps=0.5 ama istersen değiştir analizine göre