import streamlit as st

import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA




# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="light",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection




# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# global variable
cluster = 'cluster'

# utility function
def remove_jk(list_index):
    return [index[:4] for index in list_index]

def merge_df(x,y):
    z = x.copy().append(y.copy())
    i = np.unique(z.index.values, return_index = True)[1]
    return z.iloc[i].copy()

def pca_df(df):
    df_pca = df.copy() 
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_pca)
    df_pca = pd.DataFrame(data = df_pca, columns = ['PCA1', 'PCA2'], index = df.index)
    df_pca['cluster'] = df['cluster']
    return df_pca.copy()

def show_cluster(df, orientation = ""):
    if orientation == 'Vertical':

        
        aggrid_interactive_table(df)
        return 


        # N = len(df)    
        # if N < 10:
        #     st.table(df['cluster'].sort_values())  
        #     return
        # N_split = N // 2
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.table(df[:N_split])
        # with col2:
        #     st.table(df[N_split:])
    elif orientation == "Horizontal":
        st.table(df)

def transform_label(df_mix, df_test, orientation = ""):
    df_temp = df_mix.copy()
    df_testing = df_temp.loc[df_test.index]
    # show_cluster(df_testing['cluster'].sort_values(), orientation)
    show_cluster(df_testing, orientation)
    for index in df_test.index:
        df_temp.at[index, 'cluster'] = 'Input'
    df_temp['cluster'] = df_temp['cluster'].apply(lambda x: 'Cluster ' + str(x))
    return df_temp

def display_cluster(df_mix, df_test, title = "", orientation = ""):
    df_mix_pca = pca_df(df_mix)
    df_mix_pca = transform_label(df_mix_pca, df_test, orientation)

    # display DBSCAN cluster
    sns.scatterplot(data = df_mix_pca, 
                x = "PCA1", 
                y = "PCA2",
                style = cluster,
                hue = cluster
    )
    plt.title(title)
    st.pyplot(plt.gcf())

    # reset df_mix for kmeans.predict()
    df_mix = df_mix.drop(columns = cluster)
    plt.clf()

    return df_mix

def validate_index(df_target, df_test):
    return [index for index in df_test.index if index in df_target.index]

def display_price(df_price_mix, df_mix, df_target, title = ""):
    df_price_mix[cluster] = df_mix[cluster].copy()
    df_price_target = df_price_mix.loc[validate_index(df_price_mix, df_target)].copy()
    df_price_target_cluster_mean = df_price_target.groupby('cluster').mean().copy()

    sns.lineplot(data=df_price_target_cluster_mean.T, palette="tab10", linewidth=1)
    plt.title(title)
    st.pyplot(plt.gcf())

    plt.clf()
    return

def launch():
    button = False
    application_title = st.text_input('Insert application name', 'IDX Clustering')
    if application_title == "":
        st.info('Insert experiment name')
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

    if application_title != "":
        output_orientation = st.radio('Pick output orientation', ('Vertical', 'Horizontal'))    
        button = st.button('Process ' + application_title,
                        help = 'This will cluster the choice of input'
        )

    # load data
    df_kompas = pd.read_csv('kompas.csv').set_index('company')
    df_testing = pd.read_csv('testing.csv').set_index('company')
    df_price = pd.read_csv('idxx_price_2020.csv').set_index('Date')

    dbscan = pickle.load(open('dbscan.pkl', 'rb'))
    kmeans = pickle.load(open('kmeans.pkl', 'rb'))

    # processing
    if button:  
        # DATA PREPARATION - mini adjustment from last experimentation
        # will simplify in the future # fitting pipeline
        df_price = df_price.T
        df_price.index = remove_jk(df_price.index)
        df_kompas.index = remove_jk(df_kompas.index)
        df_testing.index = remove_jk(df_testing.index)

        # getting sample of n_input
        df_test = df_testing.sample(n_input).copy()

        # get mix data for dbscan detection
        df_mix = merge_df(df_test, df_kompas)    
        df_price_mix = df_price.loc[validate_index(df_price, df_mix)]

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
        if output_orientation == 'Vertical':
            # dbscan in action
            df_mix[cluster] = dbscan.fit_predict(df_mix)
            display_price(df_price_mix.copy(), df_mix, df_kompas, 'KOMPAS performance of DBSCAN cluster')
            df_mix = display_cluster(df_mix, df_test, 'DBSCAN Cluster of Input and Kompas', output_orientation)

            # kmeans in action
            df_mix[cluster] = kmeans.predict(df_mix)
            display_price(df_price_mix.copy(), df_mix, df_kompas, 'KOMPAS performance of KMEANS cluster')
            df_mix = display_cluster(df_mix, df_test, 'KMEANS Cluster', output_orientation)

            # kmeans in action
            df_mix[cluster] = kmeans.fit_predict(df_mix)
            display_price(df_price_mix.copy(), df_mix, df_mix, 'KOMPAS & INPUT performance of KMEANS cluster')
            df_mix = display_cluster(df_mix, df_test, 'KMEANS Cluster', output_orientation)

        # ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
        elif output_orientation == 'Horizontal':
            col1, col2, col3 = st.columns(3)
            with col1:
                # dbscan in action
                df_mix[cluster] = dbscan.fit_predict(df_mix)
                display_price(df_price_mix.copy(), df_mix, df_kompas, 'KOMPAS performance of DBSCAN cluster')
                df_mix = display_cluster(df_mix, df_test, 'DBSCAN Cluster of Input and Kompas', output_orientation)
            with col2:
                # kmeans in action
                df_mix[cluster] = kmeans.predict(df_mix)
                display_price(df_price_mix.copy(), df_mix, df_kompas, 'KOMPAS performance of KMEANS cluster')
                df_mix = display_cluster(df_mix, df_test, 'KMEANS Cluster', output_orientation)
            with col3:
                # kmeans in action
                df_mix[cluster] = kmeans.fit_predict(df_mix)
                display_price(df_price_mix.copy(), df_mix, df_mix, 'KOMPAS & INPUT performance of KMEANS cluster')
                df_mix = display_cluster(df_mix, df_test, 'KMEANS Cluster', output_orientation)

