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