# şimdi modeli kurarken en iyi parametreleri nasıl  bulacağımıza bakıcaz (grid search)
# modeli kurarken parametreleri iyi ayarlamalıyız böylece modelişn performansı yükselşr
# başta paramtetreler öntanımlı olarak gelir ama bunları kendşmşz ayarlayarak daha iyi halde getirebiliriz
# modelşn performansını en iyi yapan parametreyi grid search ile buluruz
# grid search ilgili paramterelerin bütün kombinasyonlarını dener

# mesela kernel destek vektör makinelerinin gamma ve c parametresş vardur
# bu iki parametre için tüm değerler tek tek denenerie mesela:
# gamma için 0.001,0.01,0.1,1,10,100 ve c için 0.001,0.01,0.1,1,10,100 parameterleri var
# yani toplamda 6*6 36 olasılık var
# grid search kullanarak 36 modelşn ehepsş denenrş ve en iyi modeli ve parametrelerşnş böylece elde ederiz

# grid search ile eğitim ve test verilerini ayırırken bir kez parçalama yerine eğitim ve test verilerini ayırırken cross validation kullanılır

# cross validation ile grid search birlikte şöyle kullanılır: GridSearchCV

from sklearn.datasets import load_digits
digits=load_digits()

print(digits.DESCR) # 5620 örneklem ve 64 öznitelik var
print()

# paramtereleri belirleyelim
# bu veri seti için destek vektör makinelerini kullanalım
# önce paramtereleri tek tek yazalım çünkü her parameteriy kullanıp en iyi yaklaşımı bulucaz
# kernelin rbf yaklaşımında gamma ve C parametreleri vardı
# kernelin linner yaklaşımında sadece C parametresi vardı
param_grid=[{"kernel":["rbf"],"gamma":[1e-3,1e-4],"C":[1,10,100,1000]},
            {"kernel":["linear"],"C":[1,10,100,1000]}]

#şimdi cross val ile kullanacağımız grid search import edelim
from sklearn.model_selection import GridSearchCV
# destek vektör makinelerini de impoart edelim
from sklearn.svm import SVC

# şimdi destek vektör makinelerini ve oluşturdığu parametreleri kullancarak grid search sınıfından bir örnek alalım
grid_search=GridSearchCV(SVC(),param_grid,cv=5)
# gris search cv cross validation tekniğini kullandık
# bu teknşk veri setini eğitim ve validation olarak ikiye ayırır ama
# modeli daha önce görmedğiğ verilerler değerlendirmek için veriyi eğitim ve test şeklinde parçalamalıyız

# yani eğitim ve validation iile en iyi paramterlereri grid_search ile bulduk ve bu parametreler ile modeli kurcaz
# ardından da modelin parameterlerini bulurken ve modeli kurarken kullanmadığımız verilerle modeli test edicez

# şimdi veri setini eğitim ve test olrak parçala
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,random_state=0)

# oluşturdupğumuz grid search nesnesi bir sınıflayıcı gibi davranır
grid_search.fit(X_train,y_train)

# modelin skoruna bakalım
print(grid_search.score(X_test,y_test)) # model %99 oranında çalılır
print()

# en iyi parametreleri görelim
print(grid_search.best_params_)
print()

# en iyi skoru görelim
print(grid_search.best_score_)
print()

# score ve best score farkı şudur:
# score metodu modeli kurduktan sonra modeliin tahmin test seti üzerindeki performansını gösterir
# best score ise  modeli kurarken eğitim setinde cross validtion ile elde edilen skorların ortalamalarını veirir

# en iyi parametreleri oluşturan modelin argumanlarını görelim
print(grid_search.best_estimator_)

print()


# şimdi diğer parametreler ile kurulan tüm modellerin skorlarını görelim
print(grid_search.cv_results_)
print()
# ama çok anlaşılır görünmedi


# bunu pandas ile dataframe çevirelim
import pandas as pd
# Pandas görüntüleme ayarlarını değiştir
pd.set_option('display.max_columns', None)  # Tüm sütunları göster
pd.set_option('display.max_rows', None)     # Tüm satırları göster
sonuclar=pd.DataFrame(grid_search.cv_results_)
# değerleri daha iyi görmek için transposesini alıp gösterelim
print(sonuclar.T) # params kısmında parametreler var


print("\n*****\n")


# şimdi başta oluşturdığumuz eğitim ve test veileriye cross validation kullanarak farklı kombinasyonlar bulabiliriz
# yani aslında en başta eğitim ve test şeklinde bölerken de cross validation kullanalım
# yani hem modeli kurarken hem de değerlerndirirken cross validation kullanmış olucaz
from sklearn.model_selection import cross_val_score
skor=cross_val_score(GridSearchCV(SVC(),param_grid,cv=5),
                     digits.data,digits.target,cv=5)
print(skor)
print()
print(skor.mean()) # modek %97 doğruluğa sahip