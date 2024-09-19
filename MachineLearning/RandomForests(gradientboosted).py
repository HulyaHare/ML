# karar ağaçları tekniği aşırı uydurmaya meğillidir
# bunun etkisini azaltmak için random forest kullnılır

# ensemble nedir:
# bimlerce kişiye soru sordunuz ve bu cevapları topladınız bu bir kişinin görüşünden daha iyidir. buna kalabalığın zekası denir
# bir grup kestiricinin tahminlerini toplarsan daha iyi tahmin yaparsın
# kestricilerin grubuna ensemble denir
# mesela bir karar ağaçşları sınıflandırıcıları grubu alalım ve her sınıflandırıcıyı eğitim verisinin rastgele alt kümesinde eğitelim
# böylece hervir ağacın tahminini buluruz. bu tahminerden en çok oy alan sınıftahmin olarak alınır.
# böyle karar ağaçlarının ensamblına random forest denir

# mesela lojistik regresyon, destek vektör makineleri, random forest  sınıflandırıcısı ve K en yakın komşuluğu sınıflandırıcısını bir araya getirererk tahmin yapmayı düşünelim
# her bir algoritmanın tahmin sonuuçlarına bakılır ve en çok hangisi tahmin edildiyse o seçilir
# tahmincilerin herbiri birbirinden bağımsız çalışır yani böyle farklı algoritmalar kullanılır

from sklearn.datasets import make_moons
# noise eklemek, makine öğrenmesi modellerinin gerçek dünyadaki daha karmaşık ve hatalı veriler üzerinde nasıl performans gösterdiğini test etmeye yarar.
# 100 örnek alınacak, noise ile veri noktalarının dağılımında küçük rastgele sapmalar yapar
X,y=make_moons(n_samples=100,noise=0.25,random_state=3)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y)

# şimdi kullanılacak tahmincileri içeri aktaralım
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# modelleri eğitelim(kuralım):

# solver, modelin katsayılarını en uygun hale getirmek için kullanılan algoritmadır.
# lbfgs, küçük ve orta ölçekli veri setleri için uygun olan bir optimizasyon algoritmasıdır
# ve özellikle lojistik regresyon gibi problemler için oldukça etkilidir.
# Aynı zamanda, çok sınıflı sınıflandırma problemlerinde de iyi performans gösterir.
log=LogisticRegression(solver="lbfgs").fit(X_train,y_train)

# 10 adet karar ağacı oluşturulur
# Daha fazla ağaç genellikle daha iyi bir performans ve genelleme sağlar, ancak hesaplama maliyetini de artırabilir.
rnd=RandomForestClassifier(n_estimators=10).fit(X_train,y_train)

# fit() Metodu: X_train (özellik matrisi) ve y_train (etiketler) kullanılarak SVM modelini eğitir. Bu metod, eğitim verisine göre modelin parametrelerini ayarlar ve en uygun hiperdüzlemi bulur.
# gamma parametresi, RBF (Radial Basis Function) gibi doğrusal olmayan çekirdeklerde (kernel) kullanılır ve karar sınırının eğriliğini kontrol eder.
svm=SVC(gamma="auto").fit(X_train,y_train)

# estimators: Bu parametre, ansamblda kullanılacak baz sınıflandırıcıları ve bunların isimlerini içeren bir liste alır.
# İki tür oylama (voting) stratejisi vardır:
# hard Voting: Her bir modelin tahminine eşit ağırlık verir ve çoğunluk oylaması (majority voting) ile nihai sınıfı belirler. Her modelin tahmini bir oy olarak sayılır ve en fazla oyu alan sınıf tahmin edilir.
# soft Voting: Her modelin sınıf olasılıklarını kullanır ve bu olasılıkların ortalamasını alarak tahmin yapar. Olasılıkların toplamı en yüksek olan sınıf seçilir. Bu yöntem, modellerin doğruluklarına göre ağırlık verilmesini sağlar.
voting=VotingClassifier(estimators=[("lr",log),("rf",rnd),("svc",svm)],
                        voting="hard").fit(X_train,y_train)

# şimdi modellerin skorlarına bakalım
print(log.score(X_test,y_test))
print(rnd.score(X_test,y_test))
print(svm.score(X_test,y_test))
print(voting.score(X_test,y_test)) # oylamanın doğruluk değeri diğerierine eşit ya da daha yüksektir
# voting'de sınıflandırıcıların sınıf tahminlerinin ortalaması alınarak tahmin yapıldu
# herbir sınıflayıcının sınıf olasılıkları hesaplanırsa daha yüksek tahmin yapılır
# buna soft_voting denir
# bu opsiyonu kullanmak için herbir tahmincinin sınıflama tahminlerini hesaplayıp voting argumanına "soft" yazılır



print("\n*************\n")



# biraz önce birbirinden farklı sınıflandırıcıları gruplayarak bir tahmin yapmayı öğrendik

# DİĞER BİR YÖNTEM:
# her tahminci için aynı eğitim algoritmasını kullanmak ama bu tahmincileri eğitirken eğitim verilrinin farklı alt kümelerini kullanmaktır

# örneklem yerine koymalı seçiliyorsa bu metoda baggy denir:
# yani eğitim veri setinden rasgele alt kümeler seçilirken, bir örnek birden fazla kez seçilebilir.
# yani her bir alt küme, orijinal veri setinden rasgele örneklenir. Örnekler, seçildikten sonra tekrar geri koyulabileceği için bazı örnekler birden fazla kez seçilebilirken, bazıları hiç seçilmeyebilir.

# örneklem yerine koymasız seçilirse buna pasting denir:
# yani bir alt kümeye seçilen örnekler bir sonraki alt küme için kullanılmaz.
# bu şekilde, her bir alt küme benzersiz örnekler içerir.
# yani eğitim veri seti, örneklem yerine koymasız (without replacement) rastgele alt kümelere bölünür ve her bir alt küme farklı bir modelin eğitilmesi için kullanılır.
# bu teknik kullanıldıktan sonra kestiriciler eğitilir ve bunların tahminleri bir araya getirilir. ardından yeni bir gözkem için bir tahmin yapılır

from sklearn.ensemble import BaggingClassifier
# bu tekniği karşılaştormak için karar ağaçşarını da import edelim
from sklearn.tree import DecisionTreeClassifier

# şimdi veri setini import edelim ve girdi çıktılarını X,y değişkenlerine atalım
from sklearn.datasets import make_blobs
X,y=make_blobs(n_samples=300,centers=4,random_state=0,cluster_std=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y)

# şimdi karar ağaçşarından örnek alalım ve modeli fit ile kuralım
tree=DecisionTreeClassifier().fit(X_train,y_train)

# bagging sınıfından örnek alalım
bag=BaggingClassifier(tree, # karar ağacıdır
                      n_estimators=100, # 100 karar ağacı eğitilecek
                      max_samples=0.8, # Her bir karar ağacının eğitiminde kullanılacak örnek sayısının, toplam eğitim örneklerinin %80'i kadar
                      n_jobs=-1, # Modelin eğitiminde kullanılacak işlemci çekirdeği sayısını belirler. -1 değeri, kullanılabilir tüm çekirdeklerin kullanılacağını ifade eder.
                      random_state=1).fit(X_train,y_train)

print(tree.score(X_test,y_test)) # tek ağaçla kurulan modelin doğruk oranı
print(bag.score(X_test,y_test)) # 100 ağaçla kurulan modelin doğruk oranı



print("\n*************\n")



# RANDOM FORESTS:
# karar ağaçlarının ensemblesidir
# yine moon kullanalım
X,y=make_moons(n_samples=100,noise=0.25,random_state=3)
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y)

# 5 ağaç kullan
# bu ağaçşarın herbiri kullanılarak random forest tahmini oluşur
forest=RandomForestClassifier(n_estimators=5).fit(X_train,y_train)

print()

# göğüs kanseri veri setiyle örnek
from  sklearn.datasets import load_breast_cancer
kanser=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(kanser.data,kanser.target,random_state=0)

# 100 ağaçtan oluşan random forestten bir örnek alalım
forest=RandomForestClassifier(n_estimators=100).fit(X_train,y_train)
print(forest.score(X_test,y_test))



print("\n*************\n")



# gradient boosted regresyon ağaçşarı daha güçlü karar ağaçları oluşturmak için çoklu karar ağaçlarını toplayan diğer bir ensemple metottur
# hem regresyon hem de sınıflama için kullanılır
# herbir yeni ağaç, önceki ağaçların hatalarını doğrulamaya çalışır
# herbir ağaç verinin bir parçasını daha iyi tahmin ed3er
# ağaç eklendikçe modelin performansı artar

from sklearn.ensemble import GradientBoostingClassifier
gbrt=GradientBoostingClassifier(random_state=0).fit(X_train,y_train)
print(gbrt.score(X_train,y_train))
print(gbrt.score(X_test,y_test))
print()
# eğitim verisindeki doğruluğu çok fazla çıktı
# burada overfitting var
# yani ağaç çok derin ve komplex oldu  ve bu modelin genelleştirmesinde prblem ortaya çıkartır
# bunu önlemek için karar ağacının derinliğini sınırlamalıyız
# bunu aşmak için max_depth ayarlayalım
gbrt=GradientBoostingClassifier(max_depth=1,random_state=0).fit(X_train,y_train)
print(gbrt.score(X_train,y_train))
print(gbrt.score(X_test,y_test))
print()
# aslında learning_rate kullanarak önceki ağacın hatasını doğrulamak için ne kadar çaba harcanacağını gösterir
# max_depth yerine bunu kullanarak overtittingi biraz daha düzeltebikiriz
gbrt=GradientBoostingClassifier(learning_rate=0.01,random_state=0).fit(X_train,y_train)
print(gbrt.score(X_train,y_train))
print(gbrt.score(X_test,y_test))
