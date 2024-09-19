# sınıflandırma yaparız
# önceden belirlenmiş listeden seçilecek sınıf etiketini seçeriz
# ikili sıfnılandırma (categoric) evet-hayır şeklimdedir
# mısır tarlasının ürün miktarını farklı özellikleri kullanarak tahmin etme Regresyondur
import matplotlib.pyplot as plt
# bu algpritmada yeni bir veri öznitelik değerlerine göre en uyakın komşu sınıflarına atanır
import mglearn
from sklearn.model_selection import train_test_split

#herbiri tek bir komşu seçsin diye n_neighnors=1 oldu
mglearn.plots.plot_knn_classification(n_neighbors=1)
#plt.show()

# şimdi 3 en yakın komşuluğa göre sınıflandırma yapalım
mglearn.plots.plot_knn_classification(n_neighbors=3)
#plt.show()

print("\n**********\n")

#veri setini 2 parçaya ayırıp eğitim ve tewst şeklimde ikiye ayıralım
X,y=mglearn.datasets.make_forge()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
snf=KNeighborsClassifier(n_neighbors=3)
#eğitim verileriyle modeli oluşturalım
snf.fit(X_train,y_train)

# tahmin yapalım
print(snf.predict(X_train))

print()

# doğrıluk oranına bakalım
print(snf.score(X_test,y_test))

print("\n**********\n")

# şimdi gerçek veri setiyle uygulama yapalım
from sklearn.datasets import load_breast_cancer
kanser=load_breast_cancer()
print(kanser.keys())

print()

# şimdi veri setinin özet bilgilerine bakalım
print(kanser["DESCR"])

print()

# veri setini parçalayalım
# stratify ile hedef değişkene yani kanser.target'e göre orantılı bölünmesi sağlanır
# yani kanser türü verisinde 'iyi huylu' ve 'kötü huylu' sınıflarının oranı eğitim ve test setlerinde de aynı kalır.
X_train,X_test,y_train,y_test=train_test_split(kanser.data,kanser.target,
                                               stratify=kanser.target,
                                               random_state=66)
train_truth=[]
test_truth=[]

#model için kaç komşuğuğun en iyi performansı sergileyeceğini bilmediğimiz için 1 den 10 a kadar en yakın komşuluklara bakalım
neighbor_nums=range(1,11)
for n_neighbor in neighbor_nums:
    snf=KNeighborsClassifier(n_neighbors=n_neighbor)
    snf.fit(X_train,y_train) # modeli kuralım
    train_truth.append(snf.score(X_train,y_train)) # eğitim verisine göre doğruluk oranlarını bulalım
    test_truth.append(snf.score(X_test,y_test)) # teste verilerine göre doğruluk oranlarını bulalım

# bununan geğerleri grafikte görelim
plt.plot(neighbor_nums,train_truth,label="Eğitim Doğruluk Oranları")
plt.plot(neighbor_nums,test_truth,label="Test Doğruluk Oranları")
plt.ylabel("Doğruluk")
plt.xlabel("n-komşuluk")
# grafiklerin isimleri için legend kullan
plt.legend()
#plt.show()

# görüldüğü gibi az komşulukla kurulan model komplex modeldir bu  modelin doğruluk oranı düşüktür
# görüldüğü gibi azçok komşulukla kurulan model basit modeldir bu  modelin performansı  düşüktür

# grafiğe göre en iyi model 6 komşuluk sayısına göre oluşturulan modeldir

print("\n**********\n")

# şimdi k en yakın komişuluğun regresyon çeşidine bakalım
mglearn.plots.plot_knn_regression(n_neighbors=1)
# plt.show()

# şimdi 3'e çıkartalım
mglearn.plots.plot_knn_regression(n_neighbors=3)
# plt.show()

print("\n**********\n")

from sklearn.neighbors import KNeighborsRegressor
# girdi çıktı verilerini oluşturalım
X,y=mglearn.datasets.make_wave(n_samples=40)

# eğitim test değişkenlerini olluşturalım
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

# 3 en yakın komşuluk algoritmalarını oluşturalım
reg=KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train,y_train)

print(reg.score(X_train,y_train))
