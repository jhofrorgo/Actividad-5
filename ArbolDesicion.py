#Se importa la librería de pandas para el uso de la data 
import pandas as pd
#Se importa la  data necesaria para la actividad
df = pd.read_csv('https://raw.githubusercontent.com/sotastica/data/main/uso_internet_espana.csv')
#Se genera un ejemplo de la data de 10 registros 
df.sample(10)
#Se realiza la transformación de los datos en ceros y unos para los calculos y se elimina la primer fila 
pd.get_dummies(data=df, drop_first=True)
df = pd.get_dummies(data=df, drop_first=False)
df

#Se definen las variables explicativa y objetivo necesarias para el modelo 
explicativas = df.drop(columns='uso_internet')
objetivo = df.uso_internet

#Se importa la librería DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth = 3)
model.fit(X=explicativas, y=objetivo)
#se dine la ramificación hasta tres niveles
DecisionTreeClassifier(max_depth = 3)
#Se importa la librería para pintar el arbol de desiciones
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(28, 8))
plot_tree(decision_tree=model, feature_names=explicativas.columns, filled=True, fontsize=12);
#Para validar el modelo se toma un registro como ejemplo 
a = explicativas.sample()
a
#Se valida el modelo y la cantidad de registros correctos 
model.predict_proba(a)
y_pred = model.predict(explicativas)
y_pred.shape
#Como se visualiza en el modelo el dato mas importante para determinar el uso del internet es la edad, esto debido a que como se ve en el siguiente histograma es el dato mas completo 
import seaborn as sns
sns.histplot(x=df.edad, hue=df.uso_internet)

# Se genera la predicción del modelo y se valida con el dato esperado vs el real determinando una confiabilidad en el modelo del 80%
df['pred'] = y_pred
df.sample(100)[['uso_internet', 'pred']]
df['uso_internet'] == df['pred']
(df['uso_internet'] == df['pred']).sum()
(df['uso_internet'] == df['pred']).sum()/2454
(df['uso_internet'] == df['pred']).mean()