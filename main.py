import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

sts

st.markdown("# SIT Independent Study Visualization")

st.markdown("## 1. Select the dataset")

dataset_name = st.selectbox("Select Dataset", ("Death from road accident", "Electric use in BKK"))

if dataset_name == "Death from road accident":
        df = pd.read_csv("./dataset/DataSet_clean_CSV.csv") # read a CSV file inside the 'data" folder next to 'main.py'
        # df = pd.read_excel(...)  # will work for Excel files
elif dataset_name == "Electric use in BKK":
        df = pd.read_csv("./dataset/Electric use in BKK.csv") # read a CSV file inside the 'data" folder next to 'main.py'
else:
        None
    #X = df.data
    #Y = df.target
    #return X, Y

st.markdown(dataset_name)  # add a title
#st.write(df) # visualize my dataframe in the Streamlit app

deadm = df["DeadMonth"]
#st.write(deadm)

fig, ax = plt.subplots()

fruits = ['apple', 'blueberry', 'cherry', 'orange']
counts = [40, 100, 30, 55]
bar_labels = ['red', 'blue', '_red', 'orange']
bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

ax.set_ylabel('fruit supply')
ax.set_title('Fruit supply by kind and color')
ax.legend(title='Fruit color')

plt.show()