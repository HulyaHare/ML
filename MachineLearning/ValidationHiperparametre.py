# model parametresi ve hiperparametre farklıdır
# model parametresi veriden tahmin edilir ve model için gereklidir
# mesela regresyon modeli kurulduktan sonra olan modeldeki katsayılar model parametreleridir
# hiperparametreler model parametrelerini tahmin etmek için kullanılır ve bunları modeli kuran kişi belirler

from sklearn.datasets import load_iris
iris=load_iris()

#öznitelikr matrisi ve hedef değişkeni belirleyelim
X=iris.data
y=iris.target

# şimdi modeli ve modelin hiperparametrelerini seçelim
# en yakın komşuluk kulanalım
from sklearn.neighbors import KNeighborsClassifier

# n_neighbors bir parametredir
# n_neighbors=1 olduğunda model en yakın komşunun etiketine bakarak sınıflandırılır
model=KNeighborsClassifier(n_neighbors=1)
model.fit(X,y)

# verinin etiketlerini tahmin edelim
y_model=model.predict(X)

# performansına bakalım
from sklearn.metrics import accuracy_score
print(accuracy_score(y,y_model))
# ama modelin %100 doğru tahmin yağtığı ortaya çıktı yani overfitting var
# çünkü model kurmayı ve değerlendirmeyi aynı modelde yaptık
# veri setini eğitim ve test olarak bölüp yapmamız gerekiyordu:



print("\n****************\n")



from sklearn.model_selection import train_test_split
# %50 test ve %50 eğitim olsun
X1,X2,y1,y2=train_test_split(X, y, random_state=0, train_size=0.5)

# modeli kuralım
model.fit(X1, y1)

# modelin performansına bakalım
y2_model=model.predict(X2)

# modlein doğruluk skoruna bakalım
print(accuracy_score(y2, y2_model)) # %91



print("\n****************\n")



# az önce veri setini bölmek için tam ortadan ikiye böldük yani %50 test ve %50 eğitim yaptık
# sonra da verinin ilk parçasoyla modeli kurxuk
# ama acaba bu bölme optimal bölme mi diyre şöyle buluruz: cross_validation kullannılır
# yani şimdi bir de verinin ikinci parçasoyla modeli kurup hangi parçayla kurmanın daha iy olduğuna bakıcaz
y1_model=model.fit(X2, y2).predict(X1)

# soğruluk skoruna bakalım
print(accuracy_score(y1,y1_model)) # %96 ile daha yüksek çıktı yani daha iyi bör bölmedir



print("\n****************\n")



# acaba az önce yapılan gibi %50 ve %50 değil de %20 ve %80 bölme yaparsak performans daha mı iyi olur

from sklearn.model_selection import cross_val_score
# modeli 5'e parçalayalım
print(cross_val_score(model,X,y,cv=5)) # beş parçanın doğruluk skorlarını gördük

# en iyi parametreleri bulmamız gerek
# en iyi model bias ve varyans arasında tradeof denilen optimum yeri bulmakla seçilir
# eğer yüksek bias olursa underfitting olur, yüksek varyans olursa da overfitting olur
