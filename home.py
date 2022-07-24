import streamlit as st

def launch():
    with st.container():
        st.header('Information :bulb:')

        st.subheader('KMEANS Clustering (KMC) :mag:')
        st.write('KMC group certain data into some cluster')
        st.write('Data in same cluster tend to have similarity')
        st.write('KMC group dataset into predefined K of choice')
        st.write('So there will be K group in the end')

        st.subheader('DBSCAN Clustering :mag_right:')
        st.write('DBSCAN is density-based spatial clustering of applications with noise')
        st.write('It is able to cluster group of arbitrary shape')
        st.write('And it can detect noise or outliers')

    pass