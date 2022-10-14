import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/sotastica/data/main/uso_internet_espana.csv')
df.sample(10)
pd.get_dummies(data=df, drop_first=True)
df = pd.get_dummies(data=df, drop_first=False)
df
explicativas = df.drop(columns='uso_internet')
objetivo = df.uso_internet
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth = 3)
model.fit(X=explicativas, y=objetivo)
DecisionTreeClassifier(max_depth = 3)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(28, 8))
plot_tree(decision_tree=model, feature_names=explicativas.columns, filled=True, fontsize=12);
a = explicativas.sample()
a
model.predict_proba(a)
y_pred = model.predict(explicativas)
y_pred.shape
import seaborn as sns
sns.histplot(x=df.edad, hue=df.uso_internet)
df['pred'] = y_pred
df.sample(100)[['uso_internet', 'pred']]
df['uso_internet'] == df['pred']
(df['uso_internet'] == df['pred']).sum()
(df['uso_internet'] == df['pred']).sum()/2454
(df['uso_internet'] == df['pred']).mean()