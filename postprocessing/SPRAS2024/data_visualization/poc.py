import streamlit as st
import sqlite3
import pandas as pd
import pymysql
import json
import os
import plotly.graph_objects as go
import plotly.express as px

from sqlalchemy import select, Table, MetaData, create_engine


# init streamlit page
st.write("# R/V SVEA Dataportal")
st.markdown(
    """
    Här visualiseras den data som samlas in av SLUs forskningsfartyg Svea i realtid. Scrolla ner för att ta del av grafer och plottar som 
    """
)
st.image(
            "https://www.eurofleets.eu/wp-content/uploads/2021/09/Svea-eurofleets-2-1-scaled.jpg",
            #width=400, # Manually Adjust the width of the image as per requirement

        )

def connect_to_db():
    """
    Function to connect to Google Db.
    """
    cconnection = create_engine("mysql+pymysql://{user}:{pw}@35.228.136.138/{db}".format(user = os.environ["DB_USER"] , pw=os.environ["DB_PASS"], db = os.environ["DB_NAME"]))
    return cconnection


def get_df(conn, table):
    query = f"SELECT * FROM {table}"  # Byt ut 'SailorTest' mot ditt faktiska tabellnamn
    df = pd.read_sql_query(query, conn)
    return df

if __name__ == "__main__":
    conn = connect_to_db()
    selected_table = 'svea'
    df = get_df(conn, selected_table)
    print(df)

    st.write("## Djupdata från expeditionen")
    
    fig = px.line(df, x="time", y="depth", title=f'Djup över expeditionen')
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.scatter_mapbox(df, lat="lat", lon="lon", hover_name="depth", hover_data=["time", "lat", "lon", "depth"],
                        color="depth", zoom=3, height=300)
    fig2.update_layout(mapbox_style="open-street-map")
    fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig2, use_container_width=True) 


    st.write("#### Välj parameter att studera")
    option1 = st.selectbox(
        'Vilken parameter vill du studera i plotten och på kartan?',
        ('wave_depth', 'depth', 'nasc0', 'fish_depth0', 'nasc1', 'fish_depth1', 'nasc2', 'fish_depth2', 'nasc3', 'fish_depth3',))
    
    fig = px.line(df, x="time", y=option1, title=f'{option1} över året')
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.scatter_mapbox(df, lat="lat", lon="lon", hover_name=option1, hover_data=["time", "lon", "lat", option1],
                        color=option1, zoom=3, height=300)
    fig2.update_layout(mapbox_style="open-street-map")
    fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig2, use_container_width=True) 


    st.write("#### Jämför två parametrar från expeditionen")
    option2 = st.selectbox(
        'Vilken parameter vill du studera på x-axeln?',
        ('time', 'lat', 'lon', 'wave_depth', 'depth', 'nasc0', 'fish_depth0', 'nasc1', 'fish_depth1', 'nasc2', 'fish_depth2', 'nasc3', 'fish_depth3',))
    
    option3 = st.selectbox(
        'Vilken parameter vill du studera på y-axeln?',
        ('time', 'lat', 'lon', 'wave_depth', 'depth', 'nasc0', 'fish_depth0', 'nasc1', 'fish_depth1', 'nasc2', 'fish_depth2', 'nasc3', 'fish_depth3',))

    fig = px.line(df, x=option2, y=option3, title=f'Linjeplott med {option2} och {option3}')
    st.plotly_chart(fig, use_container_width=True)


    st.write("#### Jämför två parametrar från expeditionen med rullande medianvärde")

    cols = df.columns.tolist()
    cols.remove("time")
    cols.remove("lat")
    cols.remove("lon")
    cols.remove("transmit_type")
    cols.remove("file")
    cols.remove("upload_time")
    df_median = df.copy()

    for col in cols:
        df_median[col] = df_median[col].fillna(0)
        df_median[col] = df[col].rolling(window=500).median()

    df_median = df_median.loc[500:-500]

    option4 = st.selectbox(
        'Vilken parameter vill du studera på x-axeln?',
        ('time', 'lat', 'lon', 'wave_depth', 'depth', 'nasc0', 'fish_depth0', 'nasc1', 'fish_depth1', 'nasc2', 'fish_depth2', 'nasc3', 'fish_depth3',), key="selectbox_x")
    
    option5 = st.selectbox(
        'Vilken parameter vill du studera på y-axeln?',
        ('time', 'lat', 'lon', 'wave_depth', 'depth', 'nasc0', 'fish_depth0', 'nasc1', 'fish_depth1', 'nasc2', 'fish_depth2', 'nasc3', 'fish_depth3',), key="selectbox:y")

    fig = px.line(df_median, x=option4, y=option5, title=f'Linjeplott med {option2} och {option3}')
    st.plotly_chart(fig, use_container_width=True)