# eğitim ile model kurulur ve kurulan odel ile test verisi tahmin edilir
import pandas as pd

# id sütunu index sütunu olsun
train_data=pd.read_csv("train.csv",index_col="Id")
print(train_data.head()) # en spn sütun spnuç değişkenidir yani SalePrice
print()

# veris eti yapısına bakalım
print(train_data.shape) # 1460 satır ve 80 sütun var
print()

# girdi ve sonuç değişkenlerini oluşturakım (son satır sonuç değişkeni)
X=train_data.drop(["SalePrice"],axis=1)
y=train_data.SalePrice

print(X.shape) # 1460 satır ve 79 sütun
print(y.shape) # 1460 satır 0 sütun
print()

# girdi verilerinin genel yağısına bakalım
print(X.info()) # çok eksik veri var
print()

# eksik veri olan sütunlara ve eksik veri sayılarına bakalım
print(X.isnull().sum())
print()

# toplam eksik veri içeren sütun sayısına bakalım
print((X.isnull().sum()!=0).sum()) # 19 sütunda eksik veri var
print()

# sayısal tipteki sütunlardaki eksik için ortalama ya da medyan kullancaz
# kategorik tipteli sütunlardaki eksik veriler için en fazla tekrar eden değerleri kullanalım

# sayısal ve kategorik tipteki sütunları ayrı ayrı seçelim:

# sayısal sütunlar:
numerical_cols=[cname for cname in X.columns if X[cname].dtype in ["int64","float64"]]

# kategorik sütunlar:
# kategorisi 10'dan fazla olan sütunları analize katmıcaz çümkü fazla kategorili sütunlar analize zarar verir
categorical_cols=[cname for cname in X.columns if X[cname].nunique()<10 and X[cname].dtype=="object"]

# bu sütunları birleştirelim
my_cols=categorical_cols+numerical_cols

# bu sütunları veri setinden seçelim
X_data=X[my_cols].copy()
print(X_data.shape) # 1460 satır ve 76 sütun

# sayısal tipteki sütunlardaki eksik için ortalama ya da medyan kullancaz
# bunun  için SimpleImputer kullanılır
from sklearn.impute import SimpleImputer
numerical_transformer=SimpleImputer(strategy="median")

# kategorik sütunlar için önce eksik veirler ele alınacak sonra ONeHotCoding kullancaz
from sklearn.pipeline import Pipeline # pipeline ile işlem adımlarını bağlayalım
from sklearn.preprocessing import OneHotEncoder
categorical_transformer=Pipeline(steps=[
                        ("imputer",SimpleImputer(strategy="most_frequent")), # eksik verileri dolsuralım ve bu adımın adı imputer olsun ve en çok tekrar eden değerle dolsurmak için most_frequent stratejisi kullanalım
                        ("onehot",OneHotEncoder(handle_unknown="ignore")) #kategorik veirleri one hot kodlamaua çevilerim ve adımın adına onehot diyelim ve handle unknown ile bilinmeyen alt kategori olduğunda bu kategoriye 0 değerini verelim
                    ])

# şimdi hem sayısal hem kategorik veriler için belirlenen işlemleri bir araya getirelim
from sklearn.compose import ColumnTransformer
preprocessor=ColumnTransformer(transformers=[
                        ("num",numerical_transformer,numerical_cols), #numerik dönüşümü numerik sütunlara uyguladık ve bu iişleme num ismini verdik
                        ("cat",categorical_transformer,categorical_cols) # kategorik dönüşümü kategorik sütunlara uyguladık ve bu işleme cat ismini verdik
                    ])

# bu işlemleri veri setimize uygulayalım
X_data_pre=preprocessor.fit_transform(X_data) # kategroik sütunlar onehot çevrildi ve eksik veriler atıldı
print(X_data.shape) # 1470 satır ve 232 sğtun çünkü kategorik sütunlar one hot olunca 232 arttı
print()

# VERİ ÖNİŞLEME BİTTİ

# RandomForest ile modeli kuralım
# sonuç değişkeni sayısal olduğundan regresser yaptuk classifier yapmadık
from sklearn.ensemble import RandomForestRegressor
# 100 karar ağacı olsun
model=RandomForestRegressor(n_estimators=100,random_state=1)

# pipeline oluşturalım
my_pipeline=Pipeline(steps=[
    ("preprocessor",preprocessor), # veri önişleme yağılır
    ("model",model) # model kurulur
])

# veri setini eğitim ve test olarak parçala
from sklearn.model_selection import train_test_split
# %20 test olsun
X_train,X_test,y_train,y_test=train_test_split(X_data,y,train_size=0.2)

# eğitim verisi ve pipeline ile model kuralım
my_pipeline.fit(X_train,y_train)

# test verisini tahmin edelim
preds_test=my_pipeline.predict(X_test)

# modelin performansına bakalım
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(preds_test,y_test)) # 16961.51
print()

# modelin eğitim verisini nasıl tahmin ettiğine bakalım
preds_train=my_pipeline.predict(X_train)
print(mean_absolute_error(preds_train,y_train)) # 6444.37
print()

# biz modelin eğitim ve test verisindeki tahmin hatalarının yakın olmasını isteriz
# peki en iyi model bu mu yoksa değil mi bunu görmk için farklı ağaç sayılarıyla modeli kuralım
# bunun için bir fonksiyon yazalım
from sklearn.model_selection import cross_val_score
def get_score(n_estimators):
    my_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),  # veri önişleme yağılır
        ("model", RandomForestRegressor(n_estimators=n_estimators,random_state=1))  # model kurulur
    ])
    # bu fonksiyona cross val score ekleyerek veri seitni parçalara bölelim ve sırayla her parçayı tek tek test parçası yapıp tüm parçaları deneyelim
    # bölyece en iyi hangi bölünmüş parçanın iyi skor yapacağını bulalım
    # ayrıca veriyi 5'e parçalayalım
    # çıkan sonuç negatif plcağaından -1 ile çarptuk
    scores = -1 * cross_val_score(my_pipeline,X_data,y,cv=5,scoring="neg_mean_absolute_error")
    return scores.mean() # scores'te beş ayrı değer olduğundan ortalamasını alalım

# 100'den 400'e ağaç sayılarının performansını görelim
result={}
for i in range(2,8):
    result[50*i]=get_score(50*i)
print(result)
print()

# herbir ağaç için sonuçların grafiğini çizdirelim
import matplotlib.pyplot as plt
plt.plot(list(result.keys()),list(result.values()))
plt.show()
# grafiğe göre ağaç sayısı 350 iken model en az hata yapıyor


# şimdi bu 350 ağaç syısına göre model kuralım çünkü en iyi model odur
model=RandomForestRegressor(n_estimators=350,random_state=1)
# pipeline oluşturalım
my_pipeline=Pipeline(steps=[
    ("preprocessor",preprocessor), # veri önişleme yağılır
    ("model",model) # model kurulur
])

# şimdi orjinal girdi veriler ile modlei kuralım
my_pipeline.fit(X_data,y)


# şimdi test veirlerini tahmin edelim
test_data=pd.read_csv("test.csv")

# test verisindeki kolonlaraı seçelim
X_test=test_data[my_cols].copy()

# kurduğumuz modele göre test verilerini tahmin edelim
preds_test=my_pipeline.predict(X_test)

# kod bitti