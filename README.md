# Data preprocessing

#### Create DataBase
```python
from sqlalchemy import create_engine

engine = create_engine('sqlite:///Database', echo=False)

def create_dabase(df):
    try:
        df.to_sql(name='data', con=engine, if_exists='replace')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        print(f'Operation closed')
```

#### Nan-values

```python
# fill Nan-values with 0
data['station_id'] = data['station_id'].fillna(0)

# Take mean value and round to nearest whole number (3.97 -> 4.0) and fill Nan-values with it
data['input_doors_count'] = data['input_doors_count'].fillna(4)

# others Nan values proportionally fill
for i in data.columns:
    probs = data[i].value_counts(normalize=True)
    data[i] = data[i].apply(lambda x: np.random.choice(probs.index, p=probs.values) if pd.isna(x) else x)
    print(f'{i} id done')

# Visualization unique values of our columns
for column in data.columns:
    print(f'{column} - {data[column].unique()}')
```

#### Value adjustment
```python
# Drop text values, replaced by int values
result_data['input_escalator_count'] = (result_data['input_escalator_count']
                                        .apply(lambda x: 21 if x == '7м х 3' else x))
```


#### Eilbow method for Kmeans 
```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as pl

inertia = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X_pca)
    inertia.append(kmeanModel.inertia_)

# visualization
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()
```

#### Visualization clusters distribution
```python
plt.figure(figsize=(12, 6))
plt.subplot(122)
plt.scatter(np.array(X_pca)[:, 0], np.array(X_pca)[:, 1], c=predictions, cmap='viridis')
plt.title("Кластеризация")
plt.show()
```


#### Correlation Matrix visualization
```python
import seaborn as sns
labeledMetro['date'] = pd.to_datetime(labeledMetro['date']).astype('int64')

correlation_matrix = labeledMetro.corr()
sns.heatmap(correlation_matrix, annot = False, cmap='coolwarm')
plt.title("Correlation")
plt.show()
```


#### Features Importance with RandomForest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

X = labeledMetro.drop({'cluster_id'}, axis=1)
y = labeledMetro['cluster_id']

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

importances = clf.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': X.columns, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False) 
print(feature_imp_df)


plt.figure(figsize=(8, 4))
plt.barh(X.columns, importances, color='skyblue')
plt.xlabel('Gini Importance')
plt.title('Feature Importance - Gini Importance')
plt.gca().invert_yaxis()  # Invert y-axis for better visualization
plt.show()
```



#### Statistical Analysis
```python
import scipy

for i in pred_itog_df.columns:
    stat, p = scipy.stats.normaltest(pred_itog_df[i]) # Критерий согласия Пирсона
    print(f'{i}: Statistics=%.3f, p-value=%.3f' % (stat, p))
    
    alpha = 0.05
    if p > alpha:
        print('Принять гипотезу о нормальности')
    else:
        print('Отклонить гипотезу о нормальности')
```


#### Graph analysis 
```python
# histograms
for i in pred_itog_df.columns:
    if i == 'cluster_id':
        continue
    else:
        sns.histplot(pred_itog_df[i], kde=True, color = 'blue')
        plt.title(f'{i} histogram')
        plt.show()

# Box-plot graphs
for i in pred_itog_df.columns:
    if i == 'cluster_id':
        continue
    else:
        sns.boxplot(x=pred_itog_df[i])
        plt.title(f'{i} box_plot')
        plt.show()
```

#### Some specific values
```python
# Spicific
# Total station load
zagr = result_data.groupby('station_id')['num_val'].sum() / result_data.groupby('station_id')['num_val'].count() 

# Bandwidth
bwidth = result_data.groupby('station_id')['input_stairs_total_bandwidth'].sum() / result_data.groupby('station_id')['input_stairs_total_bandwidth'].count() / labeledMetro.groupby('station_id')['input_stairs_total_bandwidth'].mean()  

#Specific values
count_d_i = result_data[['station_id', 'input_doors_count']]
result_data['output_doors_count'] = result_data['output_doors_count'].astype(float)
count_d_o = result_data[['station_id', 'output_doors_count']]
count_t_i = result_data[['station_id', 'input_turnstile_count']]
result_data['output_turnstile_count'] = result_data['output_turnstile_count'].astype(float)
count_t_o = result_data[['station_id', 'output_turnstile_count']]
result_data['input_escalator_count'] = result_data['input_escalator_count'].astype(float)
count_e_i = result_data[['station_id', 'input_escalator_count']]
count_e_o = result_data[['station_id', 'output_escalator_count']]
```
