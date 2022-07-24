import streamlit as st
from streamlit_option_menu import option_menu

import home, application, experiment, contact

st.set_page_config(
    page_title = 'Skripsi Web : IDX Clustering',
    page_icon = ':bar_chart:'
)

def handle_active_page(active_page):
    if active_page == 'Home':
        home.launch()
    elif active_page == 'Application':
        application.launch()
    elif active_page == 'Experiment':
        experiment.launch()
    elif active_page == 'Contact':
        contact.launch()

active_page = option_menu(
    menu_title = 'IDX Clustering',
    menu_icon = 'info-square',

    options = ['Home', 'Application', 'Experiment', 'Contact'],
    icons = ['house', 'gear-wide-connected', 'code-slash', 'globe2'],

    default_index = 1,
    orientation = 'horizontal'  
)
handle_active_page(active_page)