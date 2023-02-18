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