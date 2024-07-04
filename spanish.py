# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from scipy import stats
import seaborn as sns

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble # RandomForestClassifier()
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.inspection import PartialDependenceDisplay
from sklearn import neighbors

from statsmodels.tools.eval_measures import mse, rmse
from math import sqrt

# %%
import os
for dirname, _, filenames in os.walk('data_sets'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %%
# Obtener datos de temperaturas globales
temperature_df = pd.read_csv('data_sets/GlobalTemperatures.csv')

# Ver la forma del dataframe
temperature_df.shape

# %%
temperature_df.head()

# %%
temperature_df.info()

# %%
def convertTemptToFahrenheit(temp):
  tempInFahrenheit = (temp * 1.8) + 32
  return tempInFahrenheit

# Convertir columnas numéricas específicas de Celsius a Fahrenheit
# temperature_df_numerical_features = temperature_df.select_dtypes(include='number') # ¡No queremos convertir las columnas de incertidumbre!
numerical_cols = ['LandAverageTemperature','LandMaxTemperature','LandMinTemperature','LandAndOceanAverageTemperature']
temperature_df[numerical_cols] = temperature_df[numerical_cols].apply(convertTemptToFahrenheit)

# %%
def converToDateTime(df):
  df = df.copy()
  df['dt'] = pd.to_datetime(df['dt'])
  df['Month'] = df['dt'].dt.month
  df['Year'] = df['dt'].dt.year
  return df

# Convertir fecha a un objeto DateTime
new_temp_df = converToDateTime(temperature_df)
# Eliminar Fecha y Meses
new_temp_df = new_temp_df.drop(['dt', 'Month'], axis=1)
# Establecer el índice del dataframe en Año
new_temp_df = new_temp_df.set_index('Year')
new_temp_df.head()

# %%
new_temp_df.isnull().sum()

# %%
print("Los registros de tierra comienzan desde:", new_temp_df[new_temp_df.LandAverageTemperature.notna()].index.min())
print("Los registros de océano comienzan desde:", new_temp_df[new_temp_df.LandAndOceanAverageTemperature.notna()].index.min())

# %%
temp_df_cleaned = new_temp_df[new_temp_df.index >=1850]
temp_df_cleaned.isnull().sum()

# %%
temp_df_cleaned.head()

# %%
temp_df_cleaned.shape

# %%
def plot_average_temp(df,col1,col2,label):
  cols = [col1,col2]
  temp_df = df[cols]
  average_per_year = temp_df.groupby(temp_df.index)[cols].mean()
  average_per_year['lower temp'] = average_per_year[col1] - average_per_year[col2]
  average_per_year['upper temp'] = average_per_year[col1] + average_per_year[col2]

  plt.figure(figsize=(12,6))
  plt.plot(average_per_year.index, average_per_year[col1], color='red', label='Average')
  plt.plot(average_per_year.index, average_per_year['upper temp'], color='blue', alpha=0.3)
  plt.plot(average_per_year.index, average_per_year['lower temp'], color='blue', alpha=0.3)
  plt.fill_between(average_per_year.index, average_per_year['upper temp'], average_per_year['lower temp'], color='lightblue', alpha=0.3, label='Temperature Uncertainty Boundaries')
  plt.xlabel('Year')
  plt.ylabel('Average Temperature')
  plt.title(label)
  plt.legend(loc='best')

# Temperatura promedio de la tierra
plot_average_temp(temp_df_cleaned,'LandAverageTemperature','LandAverageTemperatureUncertainty','Temperatura promedio de la tierra por año')

# %%
# Temperatura promedio de la tierra y el océano
plot_average_temp(temp_df_cleaned,'LandAndOceanAverageTemperature','LandAndOceanAverageTemperatureUncertainty','Temperatura promedio de la tierra y el océano por año')

# %%
year_intervals = [1850, 1890, 1930, 1970, 2010]
temp_df_every_40_years = temp_df_cleaned[temp_df_cleaned.index.isin(year_intervals)]

temp_df_cleaned['YearInterval'] = pd.cut(temp_df_cleaned.index, bins=year_intervals + [2020], right=False, labels=year_intervals)

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=temp_df_cleaned, x='YearInterval', y='LandAndOceanAverageTemperature', ax=ax)
ax.set(ylabel='Average Temperature', title="Temperatura promedio en intervalos de 40 años")
for item in ax.get_xticklabels():
    item.set_rotation(90)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8,6))
corr_matrix = np.abs(temp_df_cleaned.corr())
sns.heatmap(temp_df_cleaned.corr()[['LandAndOceanAverageTemperature']].sort_values('LandAndOceanAverageTemperature').tail(10),
 vmax=1, vmin=-1, cmap='YlGnBu', annot=True, ax=ax);
ax.invert_yaxis()

# %%
def reduce_df(df):
  # Crear una copia del dataframe original
  df = df.copy()

  # Eliminar LandMaxTemperatureUncertainty, LandAndOceanAverageTemperatureUncertainty,
  # LandMinTemperatureUncertainty y LandMinTemperatureUncertainty
  cols_to_drop = ['LandMaxTemperatureUncertainty','LandAndOceanAverageTemperatureUncertainty',
                'LandAverageTemperatureUncertainty','LandMinTemperatureUncertainty']
  df = df.drop(cols_to_drop,axis=1)
  return df

reduced_temperature_df = reduce_df(temp_df_cleaned)

# %%
# Características, X
X = reduced_temperature_df.drop('LandAndOceanAverageTemperature',axis=1)
# Objetivo, Y
Y = reduced_temperature_df['LandAndOceanAverageTemperature']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=40)
print("Tamaño de X_train: ", X_train.shape)
print("Tamaño de X_test: ", X_test.shape)
print("Tamaño de y_train: ", y_train.shape)
print("Tamaño de y_test: ", y_test.shape)

# %%
y_pred = [y_train.mean()] * len(y_train)

# MAE
print('MAE base (en grados):', round(mean_absolute_error(y_train,y_pred),2))

# %%
# Capturar salida k-fold en un dataframe para comparaciones
kfold_df = pd.DataFrame(columns=['Modelo','Fold_1','Fold_2','Fold_3','Fold_4','Fold_5'])
kfold_mae_df = pd.DataFrame(columns=['Modelo','Fold_1','Fold_2','Fold_3','Fold_4','Fold_5'])

# %%
# Devuelve un diccionario de estadísticas para comparar la varianza del modelo, siempre que
# se definan cinco pliegues.
def kfold_xval(model,train_data,target_data,**kwargs):
    num_folds = kwargs.get('num_folds',10)
    ret_5 = kwargs.get('ret_5',num_folds==5)
    print("Validación cruzada usando {} pliegues".format(num_folds))
    cross_val_array = cross_val_score(model, train_data, target_data, scoring="explained_variance",cv=num_folds)
    if ret_5:
        ret_dict = {'Modelo': str(model),
                    'Fold_1': cross_val_array[0],
                    'Fold_2': cross_val_array[1],
                    'Fold_3': cross_val_array[2],
                    'Fold_4': cross_val_array[3],
                    'Fold_5': cross_val_array[4],
                   }
        print("Varianza explicada:", ret_dict)
        return(ret_dict)
    else:
        print(cross_val_array)

# %%
# Devuelve un diccionario de estadísticas para comparar el error absoluto del modelo, siempre que
# se definan cinco pliegues.
def kfold_xval_mae(model,train_data,target_data,**kwargs):
    num_folds = kwargs.get('num_folds',10)
    ret_5 = kwargs.get('ret_5',num_folds==5)
    print("Validación cruzada usando {} pliegues".format(num_folds))
    # Minimizar MAE es equivalente a maximizar el MAE negativo
    mae_val_array = cross_val_score(model, train_data, target_data, scoring="neg_mean_absolute_error", cv=num_folds)
    if ret_5:
        ret_dict = {'Modelo': str(model),
                    'Fold_1': mae_val_array[0],
                    'Fold_2': mae_val_array[1],
                    'Fold_3': mae_val_array[2],
                    'Fold_4': mae_val_array[3],
                    'Fold_5': mae_val_array[4],
                   }
        print("Error absoluto medio negativo:", ret_dict)
        return(ret_dict)
    else:
        print(mae_val_array)

def comp_train_test(model,X_train,y_train,X_test,y_test):
    model.fit(X_train,y_train)

    # Hacer predicciones
    y_preds_train = model.predict(X_train)
    y_preds_test = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    tips = sns.load_dataset("tips")
    ax = sns.regplot(data=tips,x=y_test,y=y_preds_test,scatter_kws={"color": "blue", 'alpha':0.3}, line_kws={"color": "red"})
    ax.set(xlabel='Valor de prueba', ylabel='Valor predicho',title='Valores de prueba vs. valores predichos,\n{}'.format(str(model)))
    plt.show()

    print("------------------------- Estadísticas del conjunto de prueba -------------------------")
    print("R-cuadrado del modelo en el conjunto de prueba es: {}".format(model.score(X_test, y_test)))
    print("Error absoluto medio de la predicción es: {}".format(mean_absolute_error(y_test, y_preds_test)))
    print("Error cuadrático medio de la predicción es: {}".format(mse(y_test, y_preds_test)))
    print("Raíz del error cuadrático medio de la predicción es: {}".format(rmse(y_test, y_preds_test)))
    print("Error porcentual absoluto medio de la predicción es: {}".format(np.mean(np.abs((y_test - y_preds_test) / y_test)) * 100))

# %%
linear = LinearRegression()
cv_results = kfold_xval(linear,X_train,y_train,num_folds=5)
kfold_df = pd.concat([kfold_df, pd.DataFrame.from_records([cv_results])])

# %%
mae_results = kfold_xval_mae(linear,X_train,y_train,num_folds=5)
kfold_mae_df = pd.concat([kfold_mae_df, pd.DataFrame.from_records([mae_results])])

# %%
comp_train_test(linear,X_train,y_train,X_test,y_test)

# %%
# Encontrar el RMSE para trazar una curva de codo
rmse_val = [] # Almacenar valores de RMSE para diferentes k
for k in range(20):
    k = k+1
    model = neighbors.KNeighborsRegressor(n_neighbors = k)

    model.fit(X_train, y_train)  # Ajustar el modelo
    pred=model.predict(X_test) # Hacer predicción en el conjunto de prueba
    error = sqrt(mean_squared_error(y_test,pred)) # Calcular RMSE
    rmse_val.append(error) # Almacenar valores de RMSE

figsize=(8, 6)
curve = pd.DataFrame(rmse_val) # Curva de codo
curve.plot(title="Curva de codo del RMSE", xlabel="k", ylabel="RMSE", legend=None)

# %%
knn_model = neighbors.KNeighborsRegressor(n_neighbors = 8) # Por la curva de codo arriba, establecemos k = 8
cv_results = kfold_xval(knn_model,X_train,y_train,num_folds=5)
kfold_df = pd.concat([kfold_df, pd.DataFrame.from_records([cv_results])])

# %%
mae_results = kfold_xval_mae(knn_model,X_train,y_train,num_folds=5)
kfold_mae_df = pd.concat([kfold_mae_df, pd.DataFrame.from_records([mae_results])])

# %%
comp_train_test(knn_model,X_train,y_train,X_test,y_test)

# %%
rf_model = ensemble.RandomForestRegressor(n_estimators=10, n_jobs=-1)
cv_results = kfold_xval(rf_model,X_train,y_train,num_folds=5)
kfold_df = pd.concat([kfold_df, pd.DataFrame.from_records([cv_results])])

# %%
mae_results = kfold_xval_mae(rf_model,X_train,y_train,num_folds=5)
kfold_mae_df = pd.concat([kfold_mae_df, pd.DataFrame.from_records([mae_results])])

# %%
comp_train_test(rf_model,X_train,y_train,X_test,y_test)

# %%
svm_model = SVR(kernel = 'rbf')
cv_results = kfold_xval(svm_model,X_train,y_train,num_folds=5)
kfold_df = pd.concat([kfold_df, pd.DataFrame.from_records([cv_results])])

# %%
mae_results = kfold_xval_mae(svm_model,X_train,y_train,num_folds=5)
kfold_mae_df = pd.concat([kfold_mae_df, pd.DataFrame.from_records([mae_results])])

# %%
comp_train_test(svm_model,X_train,y_train,X_test,y_test)

# %%
# Buscar el número óptimo de n_estimators y max_depth
from sklearn.model_selection import GridSearchCV
params = {'n_estimators': range(50,500,50),
          'max_depth': [2,3,4]}
gb_model = ensemble.GradientBoostingRegressor(random_state=99)
gscv = GridSearchCV(gb_model,params,cv=5)
gscv.fit(X_train,y_train)
gscv.best_params_, gscv.best_score_

# %%
# Buscar el learning_rate y la loss óptimos
params = {'learning_rate': [x/10 for x in range(1,11,1)],
          'loss': ['squared_error','absolute_error','huber']}
gb_model = ensemble.GradientBoostingRegressor(max_depth=4,random_state=99)
gscv = GridSearchCV(gb_model,params,cv=5)
gscv.fit(X_train,y_train)
gscv.best_params_, gscv.best_score_

# %%
# Buscar el subsample y min_samples_split óptimos
params = {'subsample': [x/20 for x in range(10,20,1)],
          'min_samples_split': [2**x for x in range(1,9)]}
gb_model = ensemble.GradientBoostingRegressor(max_depth=4,random_state=99)
gscv = GridSearchCV(gb_model,params,cv=5)
gscv.fit(X_train,y_train)
gscv.best_params_, gscv.best_score_

# %%
# Estos son parámetros razonablemente buenos.
params = {'loss': 'squared_error',
          'random_state': 99,
          'max_depth': 4,
          'n_estimators': 100,
          'learning_rate': 0.1,
          'subsample': 0.75,
          'min_samples_split': 2,
         }
gb_model = ensemble.GradientBoostingRegressor(**params)
cv_results = kfold_xval(gb_model,X_train,y_train,num_folds=5)
kfold_df = pd.concat([kfold_df, pd.DataFrame.from_records([cv_results])])

# %%
mae_results = kfold_xval_mae(gb_model,X_train,y_train,num_folds=5)
kfold_mae_df = pd.concat([kfold_mae_df, pd.DataFrame.from_records([mae_results])])

# %%
comp_train_test(gb_model,X_train,y_train,X_test,y_test)

# %%
pd.set_option('display.float_format', lambda x: '%.4f' % x)
fold_cols = ['Fold_1','Fold_2','Fold_3','Fold_4','Fold_5']
kfold_df['avg_explained_variance'] = kfold_df[fold_cols].mean(axis=1)
kfold_df.sort_values(['avg_explained_variance'],ascending=False)
kfold_df.style.hide(axis="index")

# %%
pd.set_option('display.float_format', lambda x: '%.4f' % x)
kfold_mae_df['avg_neg_means_abs_error'] = kfold_mae_df[fold_cols].mean(axis=1)
kfold_mae_df.sort_values(['avg_neg_means_abs_error'],ascending=False)
kfold_mae_df.style.hide(axis="index")

# %%
# MAE
print('MAE del Regresor KNN en el conjunto de entrenamiento (en grados):', round(mean_absolute_error(y_train,knn_model.predict(X_train)),2))
print('MAE del Regresor KNN en el conjunto de prueba (en grados):', round(mean_absolute_error(y_test,knn_model.predict(X_test)),2))

# %%
# Buscar el subsample y min_samples_split óptimos
params = {'subsample': [x/20 for x in range(10,20,1)],
          'min_samples_split': [2**x for x in range(1,9)]}
gb_model = ensemble.GradientBoostingRegressor(max_depth=4,random_state=99)
gscv = GridSearchCV(gb_model,params,cv=5)
gscv.fit(X_train,y_train)
gscv.best_params_, gscv.best_score_

# %%
# Estos son parámetros razonablemente buenos.
params = {'loss': 'squared_error',
          'random_state': 99,
          'max_depth': 4,
          'n_estimators': 100,
          'learning_rate': 0.1,
          'subsample': 0.75,
          'min_samples_split': 2,
         }
gb_model = ensemble.GradientBoostingRegressor(**params)
cv_results = kfold_xval(gb_model,X_train,y_train,num_folds=5)
kfold_df = pd.concat([kfold_df, pd.DataFrame.from_records([cv_results])])

# %%
mae_results = kfold_xval_mae(gb_model,X_train,y_train,num_folds=5)
kfold_mae_df = pd.concat([kfold_mae_df, pd.DataFrame.from_records([mae_results])])

# %%
comp_train_test(gb_model,X_train,y_train,X_test,y_test)

# %%
pd.set_option('display.float_format', lambda x: '%.4f' % x)
fold_cols = ['Fold_1','Fold_2','Fold_3','Fold_4','Fold_5']
kfold_df['avg_explained_variance'] = kfold_df[fold_cols].mean(axis=1)
kfold_df.sort_values(['avg_explained_variance'],ascending=False)
kfold_df.style.hide(axis="index")

# %%
pd.set_option('display.float_format', lambda x: '%.4f' % x)
kfold_mae_df['avg_neg_means_abs_error'] = kfold_mae_df[fold_cols].mean(axis=1)
kfold_mae_df.sort_values(['avg_neg_means_abs_error'],ascending=False)
kfold_mae_df.style.hide(axis="index")

# %%
# MAE
print('MAE del Regresor KNN en el entrenamiento (en grados):', round(mean_absolute_error(y_train,knn_model.predict(X_train)),2))
print('MAE del Regresor KNN en la prueba (en grados):', round(mean_absolute_error(y_test,knn_model.predict(X_test)),2))

# %%
mape = mean_absolute_percentage_error(y_test, knn_model.predict(X_test)) * 100
accuracy = 100 - mape
print("Exactitud del Modelo de Gradient Boosting:", round(accuracy,2), "%.")

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Efecto de la Temperatura Promedio de la Tierra y la Temperatura Promedio de la Tierra y el Océano")
# La característica que queremos comparar contra nuestro modelo para ver el efecto neto en la predicción del modelo (nuestro objetivo)
feature = ['LandAverageTemperature' ]
gb_disp = PartialDependenceDisplay.from_estimator(knn_model, X_test, feature, ax=ax)

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Efecto de la Temperatura Máxima de la Tierra y la Temperatura Promedio de la Tierra y el Océano")
feature = ['LandMaxTemperature']
gb_disp = PartialDependenceDisplay.from_estimator(knn_model, X_test, feature, ax=ax)

# %%
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Efecto de la Temperatura Mínima de la Tierra y la Temperatura Promedio de la Tierra y el Océano")
# La característica que queremos comparar contra nuestro modelo para ver el efecto neto en la predicción del modelo (nuestro objetivo)
feature = ['LandMinTemperature']
gb_disp = PartialDependenceDisplay.from_estimator(knn_model, X_test, feature, ax=ax)

# %%
features = [('LandAverageTemperature','LandMaxTemperature')];
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(knn_model, X_test, features,ax=ax)
plt.show()

# %%
features = [('LandAverageTemperature','LandMinTemperature')];
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(knn_model, X_test, features,ax=ax)
plt.show()