import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import plotly.express as px
import datetime
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


st.set_page_config(page_title="AI dataset Dashboard",
                   page_icon=":bar_chart:",
                   layout="wide"
)

st.markdown("# SIT Independent Study : Artificial intelligence (AI) prediction 64130700318")


df2 = pd.read_csv("./dataset/DataSet_Time.csv")
dfr = pd.read_csv("./dataset/data_time_r.csv")
#st.write(dfr.shape)

dfr["Date"]= pd.to_datetime(dfr["Date"], infer_datetime_format=True)


fig1 = plt.figure(figsize=(15,1))
sns.lineplot(x=dfr["Date"], y=dfr["Total"], data=dfr)
plt.title("Total Death")

#figp1 = plt.figure(figsize=(15,1))
#sns.lineplot(x=dfp["Date"], y=dfp["Total"], data=dfp)
#plt.title("Total Death by Province")
#st.pyplot(figp1)

#st.pyplot(fig1)

#Set Index
dfr = dfr.set_index('Date')
dfr.index =  pd.to_datetime(dfr.index)

#st.write(df2)

#------------------------------------------------------
#Train / Split

st.markdown("## Select the Train/Test seperate point")
dindex = st.date_input("Select Date",
        datetime.date(2019, 1, 1))
st.write('Your Seperate Train/Test seperate date point is:', dindex)

dindex= pd.to_datetime(dindex)

train = dfr.loc[dfr.index < dindex]
test = dfr.loc[dfr.index >= dindex]

figTrainTest = plt.figure(figsize=(15,1))
sns.lineplot(x=train.index, y=train["Total"], data=train, color='green', legend='auto', label="Train")
sns.lineplot(x=test.index, y=test["Total"], data=test, color='orange', legend='auto', label="Test")
plt.title("Train / Test DataSet")
st.pyplot(figTrainTest)



def create_features(dfr):
        """
        Create Time series features based on time series index
        """
        dfr['dayofweek'] = dfr.index.day_of_week 
        dfr['quarter'] = dfr.index.quarter
        dfr['month'] = dfr.index.month
        dfr['year'] = dfr.index.year
        dfr['dayofyear'] = dfr.index.day_of_year
        return dfr

dfr = create_features(dfr)

#Vistualize Features / Target Relationship

fig2 = plt.figure(figsize=(10,3))
sns.boxplot(data=dfr, x='month', y='Total',)
plt.title('Dead')
#st.pyplot(fig2)

#------------------------------------------------------
# Create Model
train = create_features(train)
test = create_features(test)

FEATURES = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear']
TARGET = 'Total'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


#Select N_Estimators
n_est = st.select_slider('Select N_Estimators parameter', options=[300, 500, 800, 1000, 1500, 2000, 2500, 3000])
st.write('Your N_Estimators is', n_est)

#Select train time
lrn_rate_pre = st.slider('Select Learning rate (milliseconds)', 1, 30, 5)
lrn_rate = lrn_rate_pre/1000
st.write('Your Learning rate is', lrn_rate, 'second')


st.write("training....")

reg = xgb.XGBRegressor(n_estimators=n_est, early_stopping_round=50,
                        learning_rate=lrn_rate)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
)

st.markdown('#### :green[Complete]')


# Feature Importance
fi = pd.DataFrame(data=reg.feature_importances_,
                index=reg.feature_names_in_,
                columns=['importance'])

fi.sort_values('importance')
#st.write(fi)

#Forecast on Test

test['prediction'] = reg.predict(X_test)
dfr = dfr.merge(test[['prediction']], how='left', left_index=True, right_index=True)

#st.write(df2)


#Mean Square Error

score = np.sqrt(mean_squared_error(test['Total'], test['prediction']))
st.markdown(f'#### Mean Square Error : :green[{score:0.2f}]')



#Coefficient of Determination
r2score = np.sqrt(r2_score(test['Total'], test['prediction']))
st.markdown(f'#### R-squared : :green[{r2score:0.2f}]')


#Create Future Dataframe

st.markdown("## Select the endpoint forecast date")
endpre = st.date_input("Select Date",
        datetime.date(2023, 9, 30))
st.write('Your end forecast date:', endpre)

endpre= pd.to_datetime(endpre)


futurepred = pd.date_range('2022-10-01', endpre, freq='1d')
future_df3 = pd.DataFrame(index=futurepred,)
future_df3 = create_features(future_df3)
future_df3['prediction'] = reg.predict(future_df3[FEATURES]).round()


#plot future table
#st.write(future_df3)

#plot chart dead

figpred = plt.figure(figsize=(15,2))
sns.lineplot(x=train.index, y=train["Total"], data=train, color='green', legend='auto', label="Train")
sns.lineplot(x=test.index, y=test["Total"], data=test, color='orange', legend='auto', label="Test")
sns.lineplot(x=future_df3.index, y=future_df3["prediction"], data=future_df3, color='grey', legend='auto', label="Forecast")
plt.title("Train / Test / Forecast DataSet")
st.pyplot(figpred)

st.markdown(f'##### Prediction Chart on Time')

# Make left chart  timeline predict death

dead_by_year = (
        future_df3['prediction']
)
 
time_chart_line = pd.DataFrame(
        dead_by_year,
)



# Make a right bar chart (Vertical) 

dead_by_month = (
        future_df3.groupby(by=["month"]).sum()[["prediction"]].sort_values(by="month")
)
fig_dead_month = px.bar(
        dead_by_month,
        x=dead_by_month.index,
        y="prediction",
        text="prediction",
        title="<b>Prediction of each month death</b>",
        color_discrete_sequence=["#0083B8"] * len(dead_by_month),
        template="plotly_white",
)

fig_dead_month.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig_dead_month.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),
)



#plot chart

left_column, right_column = st.columns(2)
left_column.line_chart(time_chart_line)
right_column.plotly_chart(fig_dead_month, use_container_width=True)





r_show = st.selectbox(
        "Select Region to show:", ('North', 'NorthEast', 'Central', 'East', 'West', 'South')
)

#st.write(dfr)
#st.write(future_df3)
#st.write(r_show)

futurepred = pd.date_range('2022-10-01', endpre, freq='1d')
future_dfr = pd.DataFrame(index=futurepred,)
future_dfr = create_features(future_dfr)


TARGETR = r_show

X_trainr = train[FEATURES]
y_trainr = train[TARGETR]

X_testr = test[FEATURES]
y_testr = test[TARGETR]


st.write("training....")

reg2 = xgb.XGBRegressor(n_estimators=n_est, early_stopping_round=50,
                        learning_rate=lrn_rate)

reg2.fit(X_trainr, y_trainr,
        eval_set=[(X_trainr, y_trainr), (X_testr, y_testr)],
        verbose=100
)

st.markdown(f'#### Prediction Death of Region : :red[{r_show}]')

future_dfr['prediction'] = reg2.predict(future_dfr[FEATURES]).round()

#Write Table
#st.write(future_dfr)


#plot chart death region

figpredr = plt.figure(figsize=(15,1))
sns.lineplot(x=train.index, y=train[r_show], data=train, color='green', legend='auto', label="Train")
sns.lineplot(x=test.index, y=test[r_show], data=test, color='orange', legend='auto', label="Test")
sns.lineplot(x=future_dfr.index, y=future_dfr["prediction"], data=future_dfr, color='grey', legend='auto', label="Forecast")
plt.title('Train / Test / Forecast DataSet of Region')
st.pyplot(figpredr)

# Make chart 3 timeline predict death by region

dead_by_year_region = (
        future_dfr['prediction']
)
 
time_chart_line_r = pd.DataFrame(
        dead_by_year_region,
)

# Create an area chart with a gradient of colors
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']

st.markdown(f'##### Prediction of Death on Time Region : {r_show}')
st.area_chart(time_chart_line_r)