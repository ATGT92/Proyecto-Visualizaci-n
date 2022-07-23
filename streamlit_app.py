# Libererías

import numpy as np
import pandas as pd
#from scipy import stats
#import matplotlib.pyplot as plt 
#import seaborn as sns
import glob
import datetime
import altair as alt
import streamlit as st
import time

st.title("Rankings de Artistas y Canciones en Spotify")



with st.sidebar:
    st.button("Introducción")

st.header('Introducción')
    
st.write('El objetivo principal de este proyecto es entender - por medio de visualizaciones - el comportamiento de los ranking de canciones \
          más populares en EEUU en la plataforma Spotify, entre los años 2017 y 2021. Específicamente se desea mostrar cuales son los artistas,\
          canciones y géneros musicales más populares y la relación de estas preferencias con algunas características sonoras por canción que disponibiliza\
          la aplicación a través de su API.')

with st.sidebar:
    st.button("Datos")

st.header('Datos')

st.write('En esta sección se muestran los datos con los que se trabajarán y las transformaciones para llegar al dataset final. Los datos se obtuvieron de la\
platafoma Kaggel y consisten en dato sobre los rankings semanales de canciones (charts), artistas y canciones.')

st.markdown('**- Charts:** contiene el ranking semanal de las 200 canciones más populares en Spotify')

with st.echo():
    
    df_chart = pd.read_csv('charts.csv')
    st.write(df_chart.head(5))

st.markdown('**- Artists:** contiene información sobre los artistas musicales')

with st.echo():
    
    df_artist = pd.read_csv("artists.csv")
    st.write(df_artist.head(5))

st.markdown('**- Song:** contiene información sobre las canciones expresando diferentes atributos sonoros de las mismas')

with st.echo():
    
    df_song = pd.read_csv('tracks.csv')
    st.write(df_song.head(5))
    
st.subheader('Transformación de Data')
    
st.write('A continuación se describen las transformaciones de datos realizadas para llegar al dataset final con el cual se trabajarán las visualizaciones')

st.write('Se extraen únicamente los ranking de canciones de EEUU para trabajar acotadamente.')    
with st.echo():
    chart = df_chart[df_chart['name'].isin(['United States'])]

st.write('Se reajusta el nombre de la columna **Track Name** a **Song** del dataset **chart**')
with st.echo():
    chart = chart.rename(columns={'Track Name':'Song'})

st.write('A través de la siguiente función se formatea en distintas variables de tiempo, la fecha en que estuvo presente en el ranking la canción de cierto artista')
with st.echo():
    def DateTransformation(data):
      data['Date']   = data['Date'].astype('datetime64[ns]')
      data['Ano']    = data['Date'].dt.year
      data['Mes']    = data['Date'].dt.month
      data['Dia']    = data['Date'].dt.day
      data['AnoMes'] = data['Date'].dt.strftime('%Y%m')
      data['MesDia'] = data['Date'].dt.strftime('%m%d')
  
      return data

    chart = DateTransformation(chart)
    
st.write('Se extrae el id de cada canción para poder cruzarlo posteriormente con el dataset de canciones. Para esto se toma se toma el 5to elemento de la lista que viene\
          en el campo URL') 
with st.echo():
    chart['Song_id'] = chart['URL'].str.split('/').str[4]

st.write('Se extraen solo las columnas de interés')
with st.echo():
    chart = chart[['Song_id','Song','Artist','Date','Ano','AnoMes','MesDia','Mes','Dia','Position','Streams']] 

st.write('Se eliminan posibles filas duplicadas')
with st.echo():
    chart = chart.drop_duplicates()
    
st.write('Se elimna cualquier registro que presente nulls')
with st.echo():
    chart = chart.dropna()

    
    
    
# Ajustamos los nombres de alguna columnas
artist = df_artist.rename(columns = {'id':'Artist_id','name':'Artist'})

# Generación de los géneros: 1 a 6
for i in range(1,7):
  genero = 'Genre'+str(i)
  artist[genero] = artist['genres'].str.replace("'",'').str.replace('[','').str.replace(']','').str.replace(', ',',').str.lstrip().str.rstrip().str.split(',').apply(lambda x: x[i-1] if len(x) >= i else np.nan)

# En caso que la lista sea "[" + "''" + "]" ( o [''] como lista) el primer y único elemento siempre será ''. Debemos dejarlo como nan
artist.loc[artist['Genre1'] == "",'Genre1'] = np.nan

# Extraemos columnas de interés
artist = artist[['Artist_id','Artist','Genre1','Genre2','Genre3','Genre4','Genre5','Genre6']]

# Eliminamos posibles filas duplicadas
artist = artist.drop_duplicates()

# Atributos de las canciones
columns_sound = ['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo','mode','key']

# Solo extraemos la información de importancia
song = df_song[['id','name','id_artists','artists'] + columns_sound + ['time_signature','duration_ms']]

# Extraemos el id del artista de la primera posición de la lista de artistas ("id_artists"). Asumimos que el primero es el más relevante.
song['id_artists'] = song['id_artists'].str.replace('[','').str.replace(']','').str.replace(' ','').str.split(',').apply(lambda x: x[0]).str.replace("'",'')

# Extraemos el nombre del artista de la primera posición de la lista de artistas ("artists"). Asumimos que el primero es el más relevante.
song['artists'] = song['artists'].str.replace('[','').str.replace(']','').str.replace(' ','').str.split(',').apply(lambda x: x[0]).str.replace("'",'')

# Renombramos algunas columnas
song = song.rename(columns = {'id':'Song_id','name':'Song','id_artists':'Artist_id','artists':'Artist'})

# Juntamos los datos
country_final = chart.merge(song, how = 'left', on = 'Song_id')
country_final = country_final.merge(artist, how = 'left', on = 'Artist_id')

# Extraemos las columnas relevantes
country_final = country_final[['Song_x','Artist_x','Date','Ano','AnoMes','MesDia','Mes','Dia','Position','Streams',
                               'danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence',
                               'tempo','mode','key','time_signature','duration_ms','Genre1']]
                               
# Renombramos algunas columnas
country_final = country_final.rename(columns = {'Song_x':'Song','Artist_x':'Artist'})
country_final.head()

# Eliminamos cualquier registro que presente valores nulos
country_final = country_final.dropna()

# Dimensionalidad Final
country_final.shape

country_final.head(2)

    
with st.sidebar:
    st.button("Visualizaciones")


df_artist_time = country_final.groupby(['Ano','Artist'], as_index=False).agg({'Streams':'mean', 'Position':'mean'})
print(df_artist_time.shape)
df_artist_time.head(10)

alt.Chart(df_artist_time).mark_point().encode(
    x='Position',
    y='Streams',
    color='Ano:N'
).interactive()

artist_popular = country_final.groupby(['Ano','Artist'], as_index=False).agg({'Streams':'sum'})
artist_popular = artist_popular.sort_values(['Ano', 'Streams'], ascending=False).groupby(['Ano']).head(20)
print(artist_popular.shape)
artist_popular.head(10)

#brush = alt.selection_interval()

alt.Chart(artist_popular).mark_circle().encode(
    #x='Artist:N',
    #y='Streams:Q',
    #order=alt.Order("Streams", sort="descending")
    alt.X('Artist', sort=alt.EncodingSortField(field="Streams", op="sum", order='descending')),
    y='Streams:Q',
    color = 'Ano:N'
    #color = alt.condition(brush, 'Ano:N', alt.value('lightgray'))
).interactive()#.add_selection(
 #   brush
#)

sonido = country_final[['Ano','Position','danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence']].groupby(['Ano','Position'], as_index=False).mean()
sonido.head(10)

alt.Chart(sonido).transform_fold(['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence'], as_=['key', 'value']).mark_boxplot().encode(
    x='Ano:N',
    y='value:Q',
    column = 'key:N'
).properties(
    width=180,
    height=250
)
