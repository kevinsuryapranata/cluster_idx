import streamlit as st

def launch():
    with st.container():
        st.header('Information :bulb:')
        st.write('This program is an application of skripsi experiment')
        st.write('The experiment conducted on KOMPAS100 using its fundamental data and ESG Risk Rating')
        st.write('The fundamental data was obtained from yahoo while the ESG Risk Rating from the following source')
        st.write('https://www.sustainalytics.com/esg-rating')

    with st.container():    
        st.subheader('KMEANS Clustering (KMC) :mag:')
        st.write('KMC group certain data into some cluster')
        st.write('Data in same cluster tend to have similarity')
        st.write('KMC group dataset into predefined K of choice')
        st.write('So there will be K group in the end')
        
    with st.container(): 
        st.subheader('KMC Experiment result and analysis :mag:')    
        st.write('Our experiment result suggest that indeed there is outlier on IDX')
        st.write('Using fundamental data (and ESG) to cluster')
        st.write('We manage to find 6 outlier using KMC K=4')
        st.write('Then we remove this 6 data and perform KMC again with K=5')
        st.write('The result, the company in KOMPAS100 seperated evenly into 5 cluster')


    with st.container():    
        st.subheader('DBSCAN Clustering :mag_right:')
        st.write('DBSCAN is density-based spatial clustering of applications with noise')
        st.write('It is able to cluster group of arbitrary shape')
        st.write('And it can detect noise or outliers')

    with st.container(): 
        st.subheader('DBSCAN Experiment result and analysis :mag_right:')    
        st.write('From the uniqueness of the method itself, compared with KMC')
        st.write('DBSCAN catch several noise or outlier')
        st.write('And after removing outlier and performing DBSCAN 2.0')
        st.write('The cluster yielding 5 cluster, that is similar with KMC K=5')
    
    pass