# sorular sorulur onlara göre oluşturulur
# mesela şahin ve devekuşunu ayırmak için uçabiliyor mu sorusu sorulur
# eğer karar ağacı bütün yapraklar yalın olana kadar devam ederse model koplex olur yani eğitim overfit(aiırı uydurm) problemini ortaya çıkarır
# overfit engellemek için iki strateji vardır:
# İlk Budama : ağaç tamamlanmadan dallanmayı durdurur
# Son Budama : ağaç tamamlanır ama az bilgi içeren yapraklar kaldırılır

# decisiontreeregressir veya desiciontreeclasifier vardır
# sicikit learn yalnızca ilk budamayı uygular, son budamayı uygulamaz

from sklearn.datasets import load_breast_cancer
kanser=load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(kanser.data,kanser.target,stratify=kanser.target)

from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier().fit(X_train,y_train)

print(tree.score(X_train,y_train)) # 1.0
print(tree.score(X_test,y_test)) # 0.923

print()

# eğitim skoru testten çok yüksek oldu çünkü ağaç inebileceği en derine indi ve eğitim verisindeki tüm etiketleri mğkemmmel etiketledi
# yani ağaç çok derin ve komplex oldu  ve bu modelin genelleştirmesinde prblem ortaya çıkartır
# bunu önlemek için karar ağacının derinliğini sınırlamalıyız
# o yüzdem şimdi ön budama yapalım
# yani model eğitim verisine tamamen mükemmel fit edilmeden önce ağacın gelişimini durduralım
# bunun için max_depth=4 diyerek yalnız 4 süğüm kullanalım
# böylece eğitim verisindeki doğruluk düşer ama test verisindeki artar

tree=DecisionTreeClassifier(max_depth=4).fit(X_train,y_train)
print(tree.score(X_train,y_train)) # 0.981
print(tree.score(X_test,y_test)) # 0.965


print("\n**********\n")


# karar ağaçlarını daha iyi anlamak için görselleştirelim
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data[:,2:]
y=iris.target
tree=DecisionTreeClassifier(max_depth=2).fit(X,y)

# görseleştirelim
# tree.dot dosyasını dizinde oluşturalım ve grafiği buna ekleyelim
from sklearn.tree import export_graphviz
export_graphviz(tree,
                out_file="tree.dot",
                class_names=True,
                filled=True)
# karar ağacını görelim
import graphviz
with open ("tree.dot") as f:
    dot_graph=f.read()
source=graphviz.Source(dot_graph)
source.view()


print("\n**********\n")


# regresyon için karar ağaçları
from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor(max_depth=2).fit(X,y)
# görselleştirelim (her düğüde tahmin sınıfı yerine tahmin değeri vardır)
export_graphviz(tree_reg,
                out_file="tree.dot",
                class_names=True,
                filled=True)
# karar ağacını görelim
with open ("tree.dot") as f:
    dot_graph=f.read()
source=graphviz.Source(dot_graph)
source.view()