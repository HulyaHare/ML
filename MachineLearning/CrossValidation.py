import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris=load_iris()

# girdi ve çıktıları oluşturslım
X,y=iris.data,iris.target

# veri setini parçalayalım
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

# veri setinin ne kadarıının parçalandığına bakıcaz:
# ama tabii istersen ön tanımlı oranları değiştirip eğitim ve test verilerinin oranlarını değiştirelim

# X değişkeninin boyutuna bakalım
print(X.shape) # 150 örneklem ve 4 öznitelik var
print(X_train.shape) # (112,4)  %75
print(X_test.shape) # (38,4)  %25
print()

# şimdi modeli kuralım
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(solver="lbfgs",multi_class="auto",max_iter=1000)
logreg.fit(X_train,y_train)

# modeli değerlerndirelim
print(logreg.score(X_test,y_test)) # yani yeni bir iris çiçeğinin türü %97 oranla doğru tahmin edilir
print()

# şimdi modeli değerlendirmenin iki yönüne bakıcaz
# cross validation ile genelleştirme performansını değerlendirmede çok robast yani dayanıklı bir yoldur
# score metodu ile bulunan r karedir


print()


# cross valıdation
# bir modeli kurduktan sonra aynı veri üzerinde modeli test etmek doğru değildir overfitting olur
# overtittingden kaçınmak için mpodel eğitim ve test olarak ikiye ayrılır
# eğitim verisiyle kurulur ve kurulan model daha önce görmediği test verisi üzerşnde test edilir,
# cross validation modelin performansını değerlendirmek için kullanılan istatistiksel  bir metottut
# cross validationda veri hızlı şekilde parçalanır ve çoklu modeller eğitilir

# en çok kullanılan versiyonu k-fold crıss validationdur
# k burada belirli bir sayıdır
# mesela 5 fold cross validation uygulandıysa veri eşit büyüklükte 5 parçaya ayrılır
# ardından modeller eğitilir
# 5 kere işlem yapılır her birinde şöyle olur:
# 1.de  -->  test,eğitim,eğitim,eğitim,eğitim
# 2.de  -->  eğitim,test,eğitim,eğitim,eğitim
# 3.de  -->  eğitim,eğitim,test,eğitim,eğitim
# 4.de  -->  eğitim,eğitim,eğitim,test,eğitim
# 5.de  -->  eğitim,eğitim,eğitim,eğitim,test
# bölyece herbir adımda birer doğrulama skoru bulunarak toplamda 5 doğrulama skoru elde edildi

from sklearn.model_selection import cross_val_score

# bu metodu kullanarak iris veri seti için doğrulama skorlarını bulalım
# cv argumanı ile kaç fold istediğimiz yazılır
scores=cross_val_score(logreg,X,y,cv=5)
print(scores)
print()

# şimdi bu değerlerin ortalamasını alalım
print(scores.mean())

# cross validation şu yüzden kullanılır:
# train_test-split olarak aldığımız parça herhangi bir tane kısmını bölüp işlem yapıcak
# ama cross_val_score ise veri setinin her kısmını işleme spkucak sırayla her paröçasını test olarak yapıp kullancak
# böylece en iyi performansı bulucaz çünkü zaten herbr parça için denedik

# scikit learn eğitim ve test verisini orantılı böler bunu şöyle anlarız:


print("\n********************\n")


# cross validationa örnek gösterelim
# burada birinci yaklaşımda test ve eğitim verileri standart olarak ayrılıyor
# ikinci yaklaşımda test ve eğitim veirleri tabakalı olarak ayrılıyor
import mglearn
mglearn.plots.plot_stratified_cross_validation()
plt.show()

# fols sayısını ayarlayalım:
# Bu kod, k-katlı çapraz doğrulama (k-fold cross-validation) işlemi için bir KFold nesnesi oluşturur.
# KFold çapraz doğrulama, bir makine öğrenmesi modelinin performansını değerlendirmek ve doğrulamak için kullanılan popüler bir yöntemdir.
# Bu yöntem, veri setini k adet alt kümeye ayırarak modelin eğitim ve test performansını daha iyi anlamamıza olanak tanır.
from sklearn.model_selection import KFold
# shuffle veriyi karıştırıp öğrenmeyi iyileştirir
# n_splits veriyi 3 kümeye ayırır ve bu kümeler 2 train, 1 test olurlar
kfold=KFold(n_splits=3,shuffle=True,random_state=0)

# şimdi iris veri setilye kurulan modeli değerlendirelim
cross_val_score(logreg,iris.data,iris.target,cv=kfold)
print(cross_val_score)

# cross validationda kulanılan diğer yöntem de new one out yöntemidir
# kfold gibi düşünebililrsin ama burada her bir fold ayrı bir örnektir
# herbir örnek için eğitim ve test oluşur
# küçük örneklerde daha iyi işe yarar
from sklearn.model_selection import LeaveOneOut
loo=LeaveOneOut()
scores=cross_val_score(logreg,iris.data,iris.target,cv=loo)
print(scores.mean())


print("\n********************\n")



# veri setinde birbirleriyle ilişkili grupların varlığında cross validatin ayarlarına bakalım
# mesela veri setinde aynı ada sahip birçok veri olabilir
# eğitim ve test verileri için tabakalı seçim yaparsak muhtemelen aynı lişinin verileri bir yerde toplanır ve bu yüzden kurulan model sağlıklı olmaz
# bu problemin üstesinden gelmek için grup kfold tekniği kullanlır
# bu teknik arguman olarak grupların dizisini alır

# bu tekniği import edelim
from sklearn.model_selection import GroupKFold

# 12 örnekli yapay veri seti oluşturalım
from sklearn.datasets import make_blobs

# girdi çıktıları oluşturalım
X,y=make_blobs(n_samples=12,random_state=0)

# grup değişkenini oluşturalım
groups=[0,0,0,1,1,1,1,2,2,3,3,3]

# scores yazarak parametlekeri oluşturalım
scores=cross_val_score(logreg,X,y,groups,cv=GroupKFold(n_splits=3))
print(scores)