# pipeliene ile hem model hem de algoritma arasında ilşki kurulur
# birden çok tahminciyi zincirler
# mesela öznitelik seçme, normalleştirme ve sınıflama gibi işlemleri bir arada yapabiliriz

from sklearn.datasets import _samples_generator

# girdi çıktıları yapalım (çıktı sınıflardan oluşsun)
# bu kod yapay sınıflandırma veri seti oluşturur.
# make_classification fonksiyonu, makine öğrenmesi ve özellikle sınıflandırma algoritmalarını test etmek için kullanılan çeşitli özelliklere sahip yapay veri setleri oluşturmak için kullanılır.
X, y = _samples_generator.make_classification(
                                             n_features=20,       # 20 öznitelik olsun
                                             n_informative=3,     # 3 bilgi verici öznitelik
                                             n_redundant=0,       # Aralarında lineer ilişki olan (gereksiz) öznitelik olmasın
                                             n_classes=4,         # 4 sınıf
                                             n_clusters_per_class=2  # Her sınıf için 2 küme
                                             )
"""
n_features=20:
eri setinde her bir veri noktası için 20 özellik (öznitelik) olacağı anlamına gelir. Yani, X matrisi (num_samples, 20) boyutunda olacaktır.
n_informative=3:
20 öznitelikten yalnızca 3 tanesinin sınıflar arasında ayırt edici bilgiye sahip olduğunu belirtir. 
Bu, bu özelliklerin sınıfları ayırt etmek için gerçekten faydalı olduğunu ve modelin öğrenmesi gereken asıl öznitelikler olduğunu gösterir.
n_redundant=0:
Gereksiz (redundant) özniteliklerin sayısını belirtir.
Gereksiz öznitelikler, bilgilendirici özniteliklerden lineer olarak oluşturulan özniteliklerdir ve model için aslında anlamlı bilgi taşımazlar.
n_classes=4:
Veri setindeki sınıf (class) sayısını belirtir.
Bu durumda, sınıflandırma problemi 4 sınıf içerecek şekilde ayarlanır. y vektöründeki etiketler bu 4 sınıfı temsil eder.
n_clusters_per_class=2:
Her bir sınıf için oluşturulacak küme (cluster) sayısını belirtir.
Burada, her sınıfın iki farklı küme (veya alt grup) içereceği anlamına gelir.
"""
print(X.shape)
print(X)
print()
print(y.shape)
print(y)

print("\n******\n")

# girdi ve çıktıları parçalayalım
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

# 20 tane öznitelik var
# bunlardan f değerine göre en iyi 3 tanesini seçelim
from sklearn.feature_selection import SelectKBest,f_regression
anova_filter=SelectKBest(f_regression,k=3) # 3 tane öznitelik seççez

# sınıflamak için destek vektör makineleri kullanalım
from sklearn.svm import LinearSVC
clf=LinearSVC()

# şimdiye kadar en iyi öznitelik seçmek ve tahmin yapmak için nesneleri oluşturduk
# bu yaptıklarımızı bir araya getirelim
from sklearn.pipeline import make_pipeline
anova_svm=make_pipeline(anova_filter,clf)
anova_svm.fit(X_train,y_train)


# test verisinin sınıflarını tahmin edelim
y_pred=anova_svm.predict(X_test)
print(y_pred)
print()

# kurulan modelin performansına bakalım
print(anova_svm.score(X_test,y_test)) # %56
# modelin skoru düşük
print()


# modelin precision,recall ve fp scor değerlerini görelim
# bunu classification report ile görürüz
from sklearn.metrics import classification_report
# şimdi classification report kllanarak metrikleri ekrana yazdıralım
print(classification_report(y_test,y_pred))
# çıkan sonuçlardan modelin iyi tahmin yapamadığını gördük



print("\n********\n")



# şimdi gerçek veri setinde pipeline ve GridSearchCV tekniğini kullanalım
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()

# şimdi veri setini eğitim ve test şeklimde parçalayalım
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=0)

# veri setini destek vektör makineleriyle analiz edelim
from sklearn.svm import SVC

# kernel destek vektör makineleri performansını iyileştirmek için minmax scaler dönüşümünü kullanalım
from sklearn.preprocessing import MinMaxScaler

# normalde bu sınıflardan nesneler oluşturup ayrı ayrı veriye uygulayıp modeli kuracaktık
# ama pipeline ile tek hamlede yapabiliriz
pp=make_pipeline(MinMaxScaler(),SVC(gamma="auto"))

# şimdi modeli kuralım
pp.fit(X_train,y_train)
# böylece scaler ile önce eğitim verisi dönüştürüldü
# ardından bu dönüğştürülen veir için destek vektör makineleri modelini kurdu

# şimdi modeli değerlendirelim
print(pp.score(X_test,y_test)) # %95 doğru



print("\n********\n")



# adlında pipeline ile öznitelik çıkartma, öznitelik seçme, ölçekleme ve model kurma gibi 4 işlem aynı anda yapılabilir
from sklearn.datasets import load_iris
iris=load_iris()

X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=0)

# PolinomialFeatures ile öznitelikler polinoösal terimlerle genişletilir (x,x^2,x^3...)
# böylece model daha karmaşık ve doğrısal olmayan ilişkileri öğrenebilir
from sklearn.preprocessing import PolynomialFeatures

# StandartScaler ile veriler ölçeklendirilir yani mesela tüm öznitelikler 0-1 arasına standartlaştırılır(getirilir)
from sklearn.preprocessing import StandardScaler

# Ridge ile regülerleştirme eklenerek overfitting engellenir yani modelin karmaşıklığı sınırlanıp genelleme yeteneği artar
from sklearn.linear_model import Ridge

# şimdi pipeline ile bu sınıflaradn örnekleri al
pp=make_pipeline(StandardScaler(),PolynomialFeatures(),Ridge())

# şimdi en iyi polinom derecesini ve etkileşimleri bulalım
# ayrıca ridge tahmincisi için en iyi alpha değerini belirleyelim
# yani modeli en iyi yapabilecek parametreleri yazalım
param_grid={"polynomialfeatures__degree":[1,2,3], # polinomsal özelliklerin genişletileceği derecelerdir
            "ridge__alpha":[0.001,0.01,0.1,1,10,100]} # alpha büyükse düşük performans yani underfititng, alpha küçükse overfitting olur

# şimdi grid search tekniğini kullanalım
from sklearn.model_selection import GridSearchCV
# cv=5 yani veri seti 5 eşit parçaya bölünür ve her seferinde bir parçası test olur yani 5 kez deneme doğrulama yapılır
# b_jobs=-1 yani tüm mevcut işlemci çekirdekleri kullanılır (hiperparametre arama işlemi ve büyük veri setiyse bunu kullan)
grid=GridSearchCV(pp,param_grid=param_grid,cv=5,n_jobs=-1)

# modeli kuralım
grid.fit(X_train,y_train)

# şimdi en iyi model için kullanılan parametrelere bakalım
print(grid.best_params_)

print()

# modelin test verisi üzerindeli performansına bakalım
print(grid.score(X_test,y_test))