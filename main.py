import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import plotly.express as px
import datetime
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


st.set_page_config(page_title="AI dataset Dashboard",
                   page_icon=":bar_chart:",
                   layout="wide"
)

st.markdown("# SIT Independent Study Visualization")


df2 = pd.read_csv("./dataset/DataSet_Time.csv")
st.write(df2.shape)

df2["Date"]= pd.to_datetime(df2["Date"], infer_datetime_format=True)

#df2['Year'] = df2['Date'].dt.year
#df2['Month'] = df2['Date'].dt.month
#df2['Day'] = df2['Date'].dt.day
#df2['Quarter'] = df2['Date'].dt.quarter


fig1 = plt.figure(figsize=(15,1))
sns.lineplot(x=df2["Date"], y=df2["Total"], data=df2)
plt.title("Total Death")

st.pyplot(fig1)

df2 = df2.set_index('Date')
df2.index =  pd.to_datetime(df2.index)

#st.write(df2)

#------------------------------------------------------
#Train / Split

st.markdown("## Select the Train/Test point")
dindex = st.date_input("Select Date",
        datetime.date(2019, 1, 1))
st.write('Your Seperate Train/Test date point is:', dindex)

dindex= pd.to_datetime(dindex)

train = df2.loc[df2.index < dindex]
test = df2.loc[df2.index >= dindex]

figTrainTest = plt.figure(figsize=(15,1))
sns.lineplot(x=train.index, y=train["Total"], data=train, color='green', legend='auto', label="Train")
sns.lineplot(x=test.index, y=test["Total"], data=test, color='orange', legend='auto', label="Test")
plt.title("Train / Test DataSet")
st.pyplot(figTrainTest)



def create_features(df2):
        """
        Create Time series features based on time series index
        """
        df2['hour'] = df2.index.hour
        df2['dayofweek'] = df2.index.day_of_week 
        df2['quarter'] = df2.index.quarter
        df2['month'] = df2.index.month
        df2['year'] = df2.index.year
        df2['dayofyear'] = df2.index.day_of_year
        return df2

df2 = create_features(df2)

#Vistualize Features / Target Relationship

fig2 = plt.figure(figsize=(10,5))
sns.boxplot(data=df2, x='month', y='Total',)
plt.title('Dead')
#st.pyplot(fig2)

#------------------------------------------------------
# Create Model
train = create_features(train)
test = create_features(test)

FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
TARGET = 'Total'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


st.write("training....")

reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_round=50,
                        learning_rate=0.001)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
)

st.write("complete",)

# Feature Importance
fi = pd.DataFrame(data=reg.feature_importances_,
                index=reg.feature_names_in_,
                columns=['importance'])

fi.sort_values('importance')
#st.write(fi)

#Forecast on Test

test['prediction'] = reg.predict(X_test)
df2 = df2.merge(test[['prediction']], how='left', left_index=True, right_index=True)

st.write(df2)

# TEST

score = np.sqrt(mean_squared_error(test['Total'], test['prediction']))
st.markdown(f'#### Mean Squared Error : {score:0.2f}')

# Calculate Error

test['error'] = np.abs(test[TARGET] - test['prediction'])
test['date'] = test.index.date
chk = test.groupby('date')['error'].mean().sort_values(ascending=False).head(5)
chk2 = test.groupby('date')['error'].mean().sort_values(ascending=False).tail(5)

l_column, r_column = st.columns(2)
l_column.write("Top 5 Max Error")
l_column.write(chk)
r_column.write("Top 5 Max Correct")
r_column.write(chk2)


#------------------------------------------------------
# Time series cross validation

tss = TimeSeriesSplit(n_splits=5, test_size=365*1, gap=1)
df2 = df2.sort_index()

fold = 0
for train_idx, val_idx in tss.split(df2):
        train = df2.iloc[train_idx]
        test = df2.iloc[val_idx]
        fold += 1


#Forecast Horizon Explained

def create_features(df2):
        """
        Create Time series features based on time series index
        """
        df2['hour'] = df2.index.hour
        df2['dayofweek'] = df2.index.day_of_week 
        df2['quarter'] = df2.index.quarter
        df2['month'] = df2.index.month
        df2['year'] = df2.index.year
        df2['dayofyear'] = df2.index.day_of_year
        return df2

df2 = create_features(df2)


#LAG Features
def add_lag(df2):
        target_map = df2['Total'].to_dict()
        df2['lag1'] = (df2.index - pd.Timedelta('364 days')).map(target_map)
        df2['lag2'] = (df2.index - pd.Timedelta('728 days')).map(target_map)
        df2['lag3'] = (df2.index - pd.Timedelta('1092 days')).map(target_map)
        return df2

tss = TimeSeriesSplit(n_splits=5, test_size=365*1, gap=1)
df2 = df2.sort_index()

df2 = add_lag(df2)

fold = 0
pred = []
scores = []
for train_idx, val_idx in tss.split(df2):
        train = df2.iloc[train_idx]
        test = df2.iloc[val_idx]

        train = create_features(train)
        test = create_features(test)

        FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1', 'lag2', 'lag3']
        TARGET = 'Total'
        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_test = test[FEATURES]
        y_test = test[TARGET]

        reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                                n_estimators=1000,
                                early_stopping_rounds=50,
                                objective='reg:linear',
                                max_depth=3,
                                learning_rate=0.01)
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=100
        )
        y_pred = reg.predict(X_test)
        pred.append(y_pred)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(score)

st.write(f'Scores across fols {np.mean(scores):0.4f}')
st.write(f'Fold scores : {scores}')

#-------- PREDICT FUTURE DATA
# Retrain on all data

df2 = create_features(df2)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1', 'lag2', 'lag3']
TARGET = 'Total'

X_all = train[FEATURES]
y_all = train[TARGET]


reg = xgb.XGBRegressor(base_score=0.5, 
                        booster='gbtree',
                        n_estimators=1000,
                        objective='reg:linear',
                        max_depth=3,
                        learning_rate=0.01)
reg.fit(X_all, y_all,
        eval_set=[(X_all, y_all)],
        verbose=100)

#Create Future Dataframe

future = pd.date_range('2022-10-01', '2024-12-31', freq='1d')
future_df2 = pd.DataFrame(index=future)
future_df2['isFuture'] = True
df2['isFuture'] = False
df2_and_future = pd.concat([df2, future_df2])
df2_and_future = create_features(df2_and_future)
df2_and_future = add_lag(df2_and_future)

future_w_features = df2_and_future.query('isFuture').copy()



#Predict the future

future_w_features['pred'] = reg.predict(future_w_features[FEATURES])

future_w_features.columns = ['Date', 'G0', 'G1', 'G2', 'G3', 'G4', 'Total', 'hour', 
'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'prediction', 'lag1', 'lag2', 'lag3'
'isFuture', 'prediction total']
#st.write(future_w_features)
st.write(future_w_features['prediction total'])

#plot chart

figpred = plt.figure(figsize=(15,1))
sns.lineplot(x=train.index, y=train["Total"], data=train, color='green', legend='auto', label="Train")
sns.lineplot(x=test.index, y=test["Total"], data=test, color='orange', legend='auto', label="Test")
sns.lineplot(x=future_w_features.index, y=future_w_features["prediction total"], data=future_w_features, color='grey', legend='auto', label="Forecast")
plt.title("Train / Test / Forecast DataSet")
st.pyplot(figpred)


#----HIDE STREAMLIT STYLE----
hide_st_syle = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_syle, unsafe_allow_html=True)