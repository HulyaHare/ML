# yapay sinir ağları zor problemleri çözer
# ses tanıma, resim sınıflandırma, milyonlarca videoadan en iyisini önerme gibi şeyler yapılır
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons

X,y=make_moons(n_samples=100,noise=0.25,random_state=3)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=42)

# modelin komplexliğini hidden_layer_sizes ile azaltalım onun öntanımlı değeri 100'dür
# azaltmasaydık eğitim setinin skoru çok yüksek olurdu
# alpha öntanımlı olarak düşük regülerleştirme ile gelir
mlp=MLPClassifier(hidden_layer_sizes=[10],max_iter=10000,random_state=0).fit(X_train,y_train)
print(mlp.score(X_train,y_train))
print(mlp.score(X_test,y_test))


print("\n*************\n")


# kanser seti için yapalım
from sklearn.datasets import load_breast_cancer
kanser=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(kanser.data,kanser.target,random_state=0)
mlp=MLPClassifier(random_state=42).fit(X_train,y_train)

# şimdi modelim eğitim ve test verisi üzerşndeki performansını görelim
print(mlp.score(X_train,y_train))
print(mlp.score(X_test,y_test))

print()

# veriyi yeniden ölçeklendiereiliö
# StandardScaler, her bir özelliği (kolonu), ortalaması 0 ve standart sapması 1 olacak şekilde ölçeklendirir.
# Bu tür ölçeklendirme, özellikle makine öğrenmesi modellerinin daha iyi performans göstermesi için önemlidir.
# scaler.transform(X_train) ifadesi, eğitim verisini ölçeklendirilmiş değerlere dönüştürür.
# fit() işleminden elde edilen ortalama ve standart sapmayı kullanarak X_train verisi standartlaştırılır (her bir özellik (sütun) için ortalama 0 ve standart sapma 1 olacak şekilde ölçeklendirilir).
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
scaler.fit(X_test)
X_train_scaled=scaler.transform(X_train)
X_test_scales=scaler.transform(X_test)
mlp=MLPClassifier(max_iter=1000,random_state=42).fit(X_train_scaled,y_train)
print(mlp.score(X_train_scaled,y_train))
print(mlp.score(X_test_scales,y_test))

print()

# şimdi modelin performansı iyi oldu ama eğitim ve test verileri arasındaki fark çok fazla oldu
# genelleştirmeyi arttırmak için komplexliği azaltalım bunun için alpha düşer
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
scaler.fit(X_test)
X_train_scaled=scaler.transform(X_train)
X_test_scales=scaler.transform(X_test)
mlp=MLPClassifier(alpha=1,max_iter=1000,random_state=42).fit(X_train_scaled,y_train)
print(mlp.score(X_train_scaled,y_train))
print(mlp.score(X_test_scales,y_test))