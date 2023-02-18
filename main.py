import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import plotly.express as px
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import mean_squared_error


st.set_page_config(page_title="AI dataset Dashboard",
                   page_icon=":bar_chart:",
                   layout="wide"
)

st.markdown("# SIT Independent Study Visualization")

st.markdown("## Select the dataset")


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

train = df2.loc[df2.index < '01-01-2019']
test = df2.loc[df2.index >= '01-01-2019']

figTrain = plt.figure(figsize=(15,1))
sns.lineplot(x=train.index, y=train["Total"], data=train, color='orange',)
plt.title("Train Set")
st.pyplot(figTrain)

figTest = plt.figure(figsize=(15,1))
sns.lineplot(x=test.index, y=test["Total"], data=test, color='green',)
plt.title("Test Set")
st.pyplot(figTest)

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

reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_round=50,
                        learning_rate=0.001)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
)

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


df = pd.read_csv("./dataset/DataSet_clean_CSV.csv",
        skiprows=0, #use when have blank row before data table
        #nrows=10000     #use when want to limit dataset  
        ) # read a CSV file inside the 'data" folder next to 'main.py'
        # df = pd.read_excel(...)  # will work for Excel files
        

st.markdown("# Dataset Dashboard")  # add a title
#st.dataframe(df) #visualize my dataframe in the Streamlit app

st.sidebar.header("Please filter here")

cluster = st.sidebar.multiselect(
        "Select Group Clustering:",
        options=df.sort_values(by="Assignments")["Assignments"].unique(),
        default=df.sort_values(by="Assignments")["Assignments"].unique()
)

deadyear = st.sidebar.multiselect(
        "Select year:",
        options=df.sort_values(by="DeadYear")["DeadYear"].unique(),
        default=df.sort_values(by="DeadYear")["DeadYear"].unique()
)

deadmonth = st.sidebar.multiselect(
        "Select month:",
        options=df.sort_values(by="DeadMonth")["DeadMonth"].unique(),
        default=df.sort_values(by="DeadMonth")["DeadMonth"].unique()
)

region = st.sidebar.multiselect(
        "Select Region:",
        options=df.sort_values(by="Region")["Region"].unique(),
        default=df.sort_values(by="Region")["Region"].unique()
)

#province = st.sidebar.selectbox(
        #"Select Province:",
        #options=df["Province"].unique(), 
        #default=df["Province"].unique()
#)


df_selection = df.query(
        "Assignments == @cluster & DeadYear == @deadyear & DeadMonth == @deadmonth & Region == @region"
)

st.dataframe(df_selection)

st.markdown("---")
#------------Main Page----------------
st.markdown("## :bar_chart: Amount of Death from accident Dashboard")


#----TOP Dead
total_dead = int(df_selection["id"].count())
average_age = round(df_selection["Age"].mean(), 2)

left_column, left2_column = st.columns(2)
with left_column:
        st.subheader("Total Death:")
        st.subheader(f"{total_dead:,}")
with left2_column:
        st.subheader("Average Age (Years):")
        st.subheader(f"{average_age}")

st.markdown("---")

# Make a bar chart (Vertical) 

dead_by_month = (
        df_selection.groupby(by=["DeadMonth"]).count()[["id"]].sort_values(by="id")
)
fig_dead_month = px.bar(
        dead_by_month,
        x=dead_by_month.index,
        y="id",
        text="id",
        title="<b>Death by Month</b>",
        color_discrete_sequence=["#0083B8"] * len(dead_by_month),
        template="plotly_white",
)

fig_dead_month.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig_dead_month.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),
)

# Make a bar chart2 (Horizon) 

dead_by_region = (
        df_selection.groupby(by=["Region"]).count()[["id"]].sort_values(by="id")
)
fig_dead_region = px.bar(
        dead_by_region,
        x="id",
        y=dead_by_region.index,
        text="id",
        orientation="h",
        title="<b>Death by Region</b>",
        color_discrete_sequence=["Green"] * len(dead_by_region),
        template="plotly_white",
)

fig_dead_region.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig_dead_region.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False)),
)

left_column, right_column = st.columns(2)
left_column.plotly_chart(fig_dead_month, use_container_width=True)
right_column.plotly_chart(fig_dead_region, use_container_width=True)


# Make chart 3 (Death by Year)

dead_by_year = (
        df_selection.groupby(by=["DeadYear"]).count()[["id"]].sort_values(by="id")
)
 
time_chart_line = pd.DataFrame(
        dead_by_year,
)

st.markdown("##### Death Year on Year")
st.area_chart(time_chart_line)

#Make Chart 4 (Death by province)

dead_by_province = (
        df_selection.groupby(by=["Province"]).count()[["id"]].sort_values(by="id", ascending = True)
)

fig_dead_province = px.bar(
        dead_by_province,
        x="id",
        y=dead_by_province.index,
        text="id",
        orientation="h",
        title="<b>Death by Province</b>",
        color_discrete_sequence=["orange"] * len(dead_by_province),
        template="plotly_white",
)

fig_dead_province.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig_dead_province.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False)),
)

st.plotly_chart(fig_dead_province, use_container_width=True)





#----HIDE STREAMLIT STYLE----
hide_st_syle = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_syle, unsafe_allow_html=True)