import pandas as pd
df=pd.read_csv("heart.csv",sep=";")
print(df.head()) # death hedef değişkendir

print("\n********\n")

# 200 satır ve 16 sütun var
print(df.shape)

print("\n********\n")

# veri setini eğitim ve test şeklinde parçalayalım

# veri seitnin %95 seçelim
data=df.sample(frac=0.95,random_state=0)
print(data.head()) # indexler karışık seçildi yani idler karışık karışık
print()

# veri setinin kalanını atayalım
data_unseen=df.drop(data.index)
print(data_unseen.head())
print()

# veri setlerinin indexlerini resetleyellim
# böylece id'ler 0'dan başlar tekrar
data.reset_index(inplace=True,drop=True)
print(data.head()) # karışık ,d sayıları 0'dan başlatılcak şekilde düzenlendş
print()
data_unseen.reset_index(inplace=True,drop=True)
print(data_unseen.head())
print()

print("\n********\n")

# şimdi veri önişleme yapalım

# pycarette veri önişleme setup() fonksiyonu ile yapılır
# bu fonksiyon ile veri ölçekleme, verileri eğitim ve test set olarak ayırma ve sınıfları dengeleme gibi adımlar yapılır

# death sonuç değişkeninin etiketlerine ve kaçar tane olduklarına bakalım
print(data["DEATH"].value_counts())
# dengeli dağılmamış çünkü 131 ve 59 oldu

# bunu dengeli hale getirelim
from pycaret.classification import *
# balance ayarı yapmak için randomoversampler import edelim
from imblearn.over_sampling import RandomOverSampler

# şimdi setup ile veri önişleme adımlarını yapalım
model=setup(data=data,target="DEATH",
            normalize=True, # veri setini ölceklemek için normalize true oldu
            normalize_method="minmax", # veri setini minmax metodu ile ölçekleyelim
            train_size=0.8, # veri setinin %80 eğitim olarak kullanalım
            fix_imbalance=True, # böylece sonuç değişkenini dengeledik böylece data["DEATH"].value_counts() dediğimiz kısım dengeli olur
            fix_imbalance_method=RandomOverSampler(), # sonuç değişkeni bu sınıf ile dengelenecek
            session_id=0 # bu arguman aynı random_statedir ve sabitlemek için kullanılır
            )
print(model) # veri önişlemeden sonraki tanımları gördğk direk
