# SITprojectIS-318
## require
- pip install streamlit
- pip install scikit-learn
- pip install matplotlib

# Run
'streamlit run main.py' http://localhost:8501

 # Add some matplotlib code !
if dataset_name == "Death from road accident":
    fig, ax = plt.subplots()
    df.hist(
        bins=12,
        column="DeadMonth",
        grid=False,
        figsize=(10, 10),
        color="#86bf91",
        zorder=3,
        rwidth=0.5,
        ax=ax,
    )
    st.pyplot(fig)


df2["Date"]= pd.to_datetime(df2["Date"], infer_datetime_format=True)

df2['Year'] = df2['Date'].dt.year
df2['Month'] = df2['Date'].dt.month
df2['Day'] = df2['Date'].dt.day
df2['Quarter'] = df2['Date'].dt.quarter


fig1 = plt.figure(figsize=(15,1))
sns.lineplot(x=df2["Date"], y=df2["Total"], data=df2)
plt.title("Total Death")

st.pyplot(fig1)

________________________________________________


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


_______________________________
Cross Validation

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


