import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="AI dataset Dashboard",
                   page_icon=":bar_chart:",
                   layout="wide"
)

st.markdown("# SIT Independent Study Visualization")

st.markdown("## Select the dataset")

dataset_name = st.selectbox("Select Dataset", ("Death from road accident", "Electric use in BKK"))

if dataset_name == "Death from road accident":
        df = pd.read_csv("./dataset/DataSet_clean_CSV.csv",
                         skiprows=0, #use when have blank row before data table
                         #nrows=10000     #use when want to limit dataset  
        ) # read a CSV file inside the 'data" folder next to 'main.py'
        # df = pd.read_excel(...)  # will work for Excel files
elif dataset_name == "Electric use in BKK":
        df = pd.read_csv("./dataset/Electric use in BKK.csv") # read a CSV file inside the 'data" folder next to 'main.py'
else:
        None

st.markdown(dataset_name)  # add a title
#st.dataframe(df) #visualize my dataframe in the Streamlit app

st.sidebar.header("Please filter here")
deadyear = st.sidebar.multiselect(
        "Select year:",
        options=df["DeadYear"].unique(),
        default=df["DeadYear"].unique()
)

deadmonth = st.sidebar.multiselect(
        "Select month:",
        options=df["DeadMonth"].unique(),
        default=df["DeadMonth"].unique()
)

region = st.sidebar.multiselect(
        "Select Region:",
        options=df["Region"].unique(),
        default=df["Region"].unique()
)

#province = st.sidebar.selectbox(
        #"Select Province:",
        #options=df["Province"].unique(), 
        #default=df["Province"].unique()
#)


df_selection = df.query(
        "DeadYear == @deadyear & DeadMonth == @deadmonth & Region == @region"
)

st.dataframe(df_selection)

st.markdown("---")
#------------Main Page----------------
st.markdown("## :bar_chart: Amount of Dead from accident Dashboard")


#----TOP Dead
total_dead = int(df_selection["id"].count())
average_age = round(df_selection["Age"].mean(), 2)

left_column, left2_column = st.columns(2)
with left_column:
        st.subheader("Total Dead:")
        st.subheader(f"{total_dead:,}")
with left2_column:
        st.subheader("Average Age (Years):")
        st.subheader(f"{average_age}")

st.markdown("---")

# Make a bar chart

dead_by_month = (
        df_selection.groupby(by=["DeadMonth"]).count()[["id"]].sort_values(by="id")
)
fig_dead_month = px.bar(
        dead_by_month,
        x="id",
        y=dead_by_month.index,
        orientation="h",
        title="<b>Dead by Month</b>",
        color_discrete_sequence=["#0083B8"] * len(dead_by_month),
        template="plotly_white",
)
fig_dead_month.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
)

dead_by_region = (
        df_selection.groupby(by=["Region"]).count()[["id"]].sort_values(by="id")
)
fig_dead_region = px.bar(
        dead_by_region,
        x="id",
        y=dead_by_region.index,
        orientation="h",
        title="<b>Dead by Region</b>",
        color_discrete_sequence=["Green"] * len(dead_by_region),
        template="plotly_white",
)
fig_dead_region.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
)

left_chart1_column, right_chart1_column = st.columns(2)
with left_chart1_column:
        st.plotly_chart(fig_dead_month)
with right_chart1_column:
        st.plotly_chart(fig_dead_region)


# Make a bar chart2
