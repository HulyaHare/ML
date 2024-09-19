from sklearn.ensemble import RandomForestClassifier
# random_state ile kullanılan random verileri sabitledik yani her seferinde yanı veriler lucak
clf=RandomForestClassifier(random_state=0)

# 3 öznitelik ve iki örneklemli girdi değişkeni oluşturalım
X=[[1,2,3],[11,12,13]]

# iki sınıflı öznitelik değişkeni oluşturalom
y=[0,1]

# modeli veriye uyduralımc
clf.fit(X,y)

# bu modeli kullanara yeni verilerin sınıfnı tahmin edelim
print(clf.predict(X)) # çıktı değerini y bulmalı çünkü öyle yani

print("\n**********\n")

# şimdi modelin yeni veriyi nasıl tahmin ettipine bakalım
print(clf.predict([[4,5,6],[14,15,16]]))

print("\n**********\n")

# şimdi veri önişleme yapalım
# veri önişlemeyle daha iyi model lurulır ve daha hızlı işlem yağalom

# önce verileri aynı ölceklte ölçeklemek içim standartlaştırma yapalım
#  Farklı değişkenler (features) genellikle farklı birimlerde ve ölçeklerde olabilir. Örneğin, biri 0 ile 100 arasında bir sınav puanı, diğeri 0 ile 1 arasında bir olasılık olabilir. Bu durumda, büyük değerler içeren özellikler modelde daha fazla ağırlık kazanabilir ve bu, modelin performansını olumsuz etkileyebilir. Standartlaştırma, tüm özellikleri aynı ölçeğe getirir.
from sklearn.preprocessing import StandardScaler
X=[[0,15],[1,-10]]

# şimdi fit() ile verileri eğitelim ve transfor() ile verileri ölçekleyelim
print(StandardScaler().fit(X).transform(X))

print("\n**********\n")

# pipeline bakalım
# pipeline ile maline öğrenmesi proje adımlarını birbirine bağlarız
# şiödi lojistik regresyon modeli kurmak için pipeline kullancaz

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# şimdi bir pipeline nesnesi oluşturuo bunun içine tüm adımları yazalım
pipe=make_pipeline(
    StandardScaler(), # veriyi ölçekler
    LogisticRegression() # model kurar
)

print("\n**********\n")

from sklearn.datasets import load_iris
# iris veri setinden girdi çıktı değerlerini oluşturalım
# X ve y girdi çıktı değişkenlerini döndğrmek için return_X_y=True diyelim
X,y=load_iris(return_X_y=True)

# eğitim verileriyle model kurup test verileriyle modelin performansını görelim
from sklearn.model_selection import train_test_split
# veri setini %25 test ve %75 eğitim olarak parçaşaualım
# istediğin oranda parçalamak için test_size kullanılabilir
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

# fit metoduyla modeli eğtielim ve işlem adımlarını ekrana yazdıralım
print(pipe.fit(X_train,y_train))

print("\n**********\n")

# modelin performansını accuracy_score ile görelim
from sklearn.metrics import accuracy_score
# X_test değerlerini tahmin edip bu değerleri y_test ile yani gerçek değerşer ile karşılaştırıp doğrulığunu görelim
print(accuracy_score(pipe.predict(X_test),y_test))
# böylece doğrulul skorunu gördül

print("\n**********\n")

# şimdi modeli değerlendirelim

# çapraz doğrulama yani cross_validation ile görelim
# veri setini parçaşarken 4^de birini test kalanını eğitim yapmıştık
# ama cross validation yaparak kalan parçaların test olmamasını yaptuğımız adaletsizlği düzeltir

#önce bir oyuncak veri oluşturalım
from sklearn.datasets import make_regression

# bu make_regression  fonksiyonunu kullanarak girdi çıktı değişkenlerini oluşturalım
X,y=make_regression(n_samples=1000,random_state=0)

# sonuç değişkeni sayısal tipte olduğundan Lineerregression kullanarak modeli kuralım
from sklearn.linear_model import LinearRegression

# bu sınıftan örnek alalım
lr=LinearRegression()

# modeli veri setine göre eğitmek için cross_validation yöntemini kullanalım
from sklearn.model_selection import cross_validate
# şimdi bu fonksiyonla modeli kuralım
# öntanımlı olarak cross_validate fonksiyonu veri setini beş parçaya ayırır ve bu parçalara göre işlem uyapar
result=cross_validate(lr,X,y)

# bu parçaşarın test skorlarına bakalım
print(result["test_score"])
# hepsi 1 olduğıuna göre model tüm verileri %100 oranında doğru tahmin etti

print("\n**********\n")

# parametlerleri oromatik seçebiliriz
# modelin ayarlanabilir parametrelerine hiperparametre denir (hiper_parametere ile model için eğitilen parametreler farklıdır)
# iyi bir model kurmak içim iyi bir parametre ayarı yapmalıyız
# şimdi randomforest modelinin en iyi kombinasyon ayarını bulalım
# bunun içim kaliforniadaki ev fiyatalarına bakalım

from sklearn.datasets import fetch_california_housing
# bu fonksiyonu kullanarak girdi ve sonuç değişkenlerini bulalum
X,y=fetch_california_housing(return_X_y=True)

# şimdi veri setini eğitim test olarak parçala
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

# şimdi en iyi parametre kombinasyonunu bullalım,
# bu teknikte rastgele değerler kullanılır bu değerleri önceden beliticez
from sklearn.model_selection import RandomizedSearchCV

# rastgele değerler üretmek için randint import edelim
from scipy.stats import randint
# şimdi parametreler için rastgele değerler üretmek için randint kullanıp sözlük yapısı kullanalım
param_distributions={"n_estimators":randint(1,5),
                    "max_depth":randint(5,10)}

from sklearn.ensemble import RandomForestRegressor
# makine öğrenmesi algoritmalarına estimator denir
search=RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=0),
    n_iter=5, # bu değer parametre sayısını ifade eder
    param_distributions=param_distributions,
    random_state=0
)

search.fit(X_train,y_train)

# em iyi model için en iyi parametlreleri göreilim
print(search.best_params_)

print()

# şimdi bu en iyi parametreye göre test verileri üzeriinde modelin performansını görelim
print(search.score(X_test,y_test))

