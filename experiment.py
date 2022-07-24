from sklearn import cluster
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

cluster = 'cluster'
df_kompas = pd.read_csv('kompas.csv').set_index('company')
df_testing = pd.read_csv('testing.csv').set_index('company')
df_price = pd.read_csv('idxx_price_2020.csv').set_index('Date')

def remove_jk(list_index):
    return [index[:4] for index in list_index]

def merge_df(x,y):
    z = x.copy().append(y.copy())
    i = np.unique(z.index.values, return_index = True)[1]
    return z.iloc[i].copy()

def validate_index(df_target, df_test):
    return [index for index in df_test.index if index in df_target.index]

df_price = df_price.T
df_price.index = remove_jk(df_price.index)
df_kompas.index = remove_jk(df_kompas.index)
df_testing.index = remove_jk(df_testing.index)

def display_price(df_price, df_sample, df_target, title = ""):
    df_price[cluster] = df_sample[cluster].copy()
    df_price_target = df_price.loc[validate_index(df_price, df_target)].copy()
    df_price_target_cluster_mean = df_price_target.groupby('cluster').mean().copy()

    sns.lineplot(data=df_price_target_cluster_mean.T, palette="tab10", linewidth=1)
    plt.title(title)
    st.pyplot(plt.gcf())

    plt.clf()
    return

def pca_df(df):
    df_pca = df.copy() 
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_pca)
    df_pca = pd.DataFrame(data = df_pca, columns = ['PCA1', 'PCA2'], index = df.index)
    df_pca['cluster'] = df['cluster']
    return df_pca.copy()

def display_cluster(df_sample, title = ""):
    df_sample_pca = pca_df(df_sample)

    # display DBSCAN cluster
    sns.scatterplot(data = df_sample_pca, 
                x = "PCA1", 
                y = "PCA2",
                style = cluster,
                hue = cluster
    )
    plt.title(title)
    st.pyplot(plt.gcf())

    # reset df_sample for kmeans.predict()
    df_sample = df_sample.drop(columns = cluster)
    plt.clf()

    return df_sample

def process(n_input, n_k, n_eps, n_minPts):
    df_sample = df_testing.sample(n_input).copy()
    kmeans = KMeans(n_clusters = n_k)
    dbscan = DBSCAN(eps = n_eps, min_samples = n_minPts)
    
    df_sample[cluster] = kmeans.fit_predict(df_sample)
    display_price(df_price.copy(), df_sample, df_sample, 'SAMPLE performance of KMEANS')
    st.table(df_sample[cluster])
    df_sample = display_cluster(df_sample, 'KMEANS Cluster')
    

    df_sample[cluster] = dbscan.fit_predict(df_sample)
    display_price(df_price.copy(), df_sample, df_sample, 'SAMPLE performance of DBSCAN')
    st.table(df_sample[cluster])
    df_sample = display_cluster(df_sample, 'DBSCAN Cluster')
    return

def launch():

    n_input = 0
    n_k, n_eps, n_minPts = 0, 0, 0
    button = False

    experiment_title = st.text_input('Insert experiment name', 'IDX Clustering (An Experiment)')
    if experiment_title == "":
        st.info('Insert experiment name')
    if experiment_title != "":
        st.markdown(
            f"""
            This is a clustering attempt\n
            Choose number of company to cluster\n
            The following slider will determine number of company\n
            That will be cluster using experiment model\n
            """
        )
        
        n_input = st.slider('Pick number of company for testing',
                            min_value = 1,
                            max_value = 400,
                            step = 1,
                            value = 10  
        )
        st.write('Minimum input : 1')
        st.write('Maximum input : 400')
        st.write('Number of company for input:', n_input, '👈 you choose')

        n_k = st.slider('Pick number of K for KMC',
                            min_value = 2,
                            max_value = n_input,
                            step = 1,
                            value = 2  
        )
        st.write('Minimum input : 2')
        st.write('Maximum input :', n_input)
        st.write('Number of K for KMC:', n_k, '👈 you choose')

        n_eps = st.slider('Pick number of eps for DBSCAN',
                            min_value = 0.01,
                            max_value = 2.00,
                            step = 0.01,
                            value = 0.33  
        )
        st.write('Minimum input : 0.01')
        st.write('Maximum input : 2.00')
        st.write('Number of eps for DBSCAN:', n_eps, '👈 you choose')

        n_minPts = st.slider('Pick number of minPts for DBSCAN',
                            min_value = 2,
                            max_value = n_input,
                            step = 1,
                            value = 2  
        )
        st.write('Minimum input : 2')
        st.write('Maximum input :', n_input)
        st.write('Number of minPts for DBSCAN:', n_minPts, '👈 you choose')

        button = st.button('Process ' + experiment_title,
            help = 'This will cluster the choice of input'
        )

    if button:
        process(n_input, n_k, n_eps, n_minPts)
    # if experiment_title != "":
    #     output_orientation = st.radio('Pick output orientation', ('Vertical', 'Horizontal'))    
    #     button = st.button('Process ' + experiment_title,
    #                     help = 'This will cluster the choice of input'
    #     )
