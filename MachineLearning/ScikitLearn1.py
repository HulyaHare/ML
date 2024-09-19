import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris=pd.read_csv("iris.txt")
print(iris.head())

print("\n**********\n")

# veri setindeki features'leri yani öznitelikleri gösteren matrire öznitelik matrisi denir ve X ile gösterilir
# öznitelik matrisi numpay dizisi ya da pandas Dataframesi olabilir
# y ile gösterilen label yani target denilen istatistikle bağımlı değişken olarak adlandırılan hedef dizisi vardır
# bu y hedef dizisi bir boyutlu numpy ya da pandas Series yapısındadir (sayısal ya da kategorik olabilir)


# iris veri setinden öznitelik matrisi ve hedef dizisini oluşturaLIM
# özniteik matrisi:
# species'i görmek istiyoruz ve bu bir sütun olduğundan axis=1 oldu (iki boyutlu)
X_iris=iris.drop("species",axis=1)
print(X_iris.head())
print(X_iris.shape)

print()

# şimdi hedef diziyi veri setşnden çekelim (tek boyytlu)
y_iris=iris["species"]
print(y_iris.head())
print(y_iris.shape)

print("\n**********\n")

# doğrusal regresyonla model işelemi yap
# rastgele veriler oluştur
rng=np.random.RandomState(42)

# şimdi x değişkenini oluştır
# rand(50) : 0-1 arasında 50 rastgele sayu oluşturur
# 10 ile çarparak 0-10 arasında 50 sayı oluşturmuş oluruz
x=10*rng.rand(50)
print("x =",x)

# şimdi y değişkenini oluştur
# y değişkeni x'e bağlı şöyle bir fonksiyondur
# 2*x+1
# y değişkenine rastgeele 0-1 arasında 50 değer ekleyerek  verileri biraz değiştirir tam doğru olmasını engeller
y=2*x-1+rng.rand(50)
print("y =",y)

plt.scatter(x,y)
#plt.show()

print("\n**********\n")

# regresyon hesaplayalım
from sklearn.linear_model import LinearRegression
# model kurılmadan ömce hiperparametreleri belirleyip onlara göre düzemşeyelim

# fir_intercept : Modelin veriye uyması sırasında bir sabit terim (intercept) hesaplamasını sağlar. Bu, modelin y-eksenini kestiği noktayı öğrenmesini sağlar.
model=LinearRegression(fit_intercept=True)
print(model.get_params())

print("\n**********\n")

# öznitelik matrisi ve hedef dizinini düzenleelim
# newaxis ile dizi iki boyutlu hale gelir yani:
# Örneğin, x tek boyutlu bir diziyse ([1, 2, 3, ...] gibi), np.newaxis ile bu dizi iki boyutlu hale getirilir ([[1], [2], [3], ...] gibi). Bu, bir sütun vektörü oluşturur.
X=x[:,np.newaxis]
print(X.shape)

print("\n**********\n")

# şimdi veri için modeli fit() ile kuralım
model.fit(X,y)

print("\n**********\n")

# sabit değeri ve ve eğimi bulalım
print(model.coef_) # sabit değer
print(model.intercept_) # eğim

print("\n**********\n")

# modle kurduktan sonra yeni verileri tahmin edelim

# bu kod ile doğrusal regresyon modelinin belirli bir aralıktaki x değerleri için y tahminleri yapmasını sağlanır

# x eksenini -1 ile 11 arasında 50 parçaya bölelim
# x_fit bu sayıların tutulduğu bir liste gibidir
x_fit=np.linspace(-1,11)
# x_fit değişkenini iki boyutlu hale getirir bunu yapmazsal makşne öğrenimi modeline uymaz
X_fit=x_fit[:,np.newaxis]
#X_fit için y değerleirni tahmin edelim. doğrusal regresyon kullanılır
# bu tahmin edilen y değerleri y_fit'de saklanır
y_fit=model.predict(X_fit)
print(y_fit)

print("\n**********\n")

# x ve y saçılım grafiğini çizdirelim
plt.scatter(x,y)
# bu grafiğin üstüne tahmin yaptığımız doğruyu çizleim yani X_fit ve y_fit
plt.plot(x_fit,y_fit)
plt.show()

print("\n**********\n")

# şimd denetimli öğrenmedeki sınıflandırma çeşidine bakalım
# eğitim verisini kullanıp bir model oluştırucaz
# sonra bu modeli kullanarak test verisinindeki verilerin etiketini tahmin edicez
# bayes kullanalım

# veri setini eğitim ve test olarak ayıralım
from sklearn.model_selection import train_test_split

# eğitim ve test setleri için değişkenler oluşturalım
X_train,X_test,y_train,y_test=train_test_split(X_iris,y_iris,random_state=1)

# model sınıfnını seçelim yani bayes modelini seçelim
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()

# şimdi fit metoduyla eğitim değişkenlerini kullanarak modeli kuralım
model.fit(X_train,y_train)

# şimdi yeni veri için tahmin yapalım
y_model=model.predict(X_test)

# şimdi modelin doğruluk oranını bulalum
from sklearn.metrics import  accuracy_score
# şimdi etiketler ile modelin tahmin ettiği değeri karşılaştıralım
print(accuracy_score(y_test,y_model))

# böylece bu mmodeli kullanarak yeni bir iris çiçeğinin etiketini yğzde 97 oranla doğru tahmin edşyoruz