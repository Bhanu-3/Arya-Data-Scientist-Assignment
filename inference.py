import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,ConfusionMatrixDisplay
model = xgb.XGBClassifier()
model.load_model('./model/bestModel.json')

test_file = input('Enter test filepath : ')
data = pd.read_csv(f'./{test_file}')

if 'Unnamed: 0' in data.columns:
    data.drop('Unnamed: 0',axis=1,inplace=True)

select_df = data[['X12', 'X19', 'X21', 'X50', 'X55', 'X56', 'X57']].copy()

test_x_scaled = min_max_scaler.fit_transform(select_df.values)
test_pred = model.predict(test_x_scaled)

data['preds'] = test_pred
select_df['preds'] = test_pred

data.to_csv('./results/test_preds_allfeats.csv',index=False)
select_df.to_csv('./results/test_preds.csv',index=False)


