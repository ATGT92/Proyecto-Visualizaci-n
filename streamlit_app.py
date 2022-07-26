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
st.caption("Alonso Guzmán Toro")

pages = ["Introduccion", "Datos","Visualizacion"]
section = st.sidebar.radio('', pages)     

# hidden div with anchor
st.markdown("<div id='linkto_top'></div>", unsafe_allow_html=True)    

if section == "Introduccion":                  
    st.header('Introducción')

    st.write('El objetivo principal de este proyecto es entender - por medio de visualizaciones - el comportamiento de los ranking de canciones \
              más populares en EEUU en la plataforma Spotify, entre los años 2017 y 2021. Específicamente se desea mostrar cuales son los artistas,\
              canciones y géneros musicales más populares y la relación de estas preferencias con algunas características sonoras por canción que disponibiliza\
              la aplicación a través de su API.')
    
    
    """
    ### Propiedades Acústicas
    A continuación se describen las principales características sonoras de una canción bajo el modelo de Spotify:
    - **Danceability:** describe cuan adecuada es una canción para bailar basado en la combinación de elementos musicales como el ritmo. Un valor de 0 es poco bailable y un valor de 1 es muy bailable.
    - **Energy:** es una medida de la percepción de cuan intensa es una canción en términos de ruido, potencia, etc. Un valor de 0 está asociado a intensidad baja mientras que 1 a intensidad alta.
    - **Speechiness:** detecta la presencia de palabras habladas en un audio. Valores más cercanos a 1 denotan audios con presencia exclusiva de conversaciones.
    - **Acousticness:** medida de si una canción o audio es acústico. Valores cercanos a 1 denotan alta confianza en que el audio sea acústico mientras que valores cercano a 0 denotan su ausencia.
    - **Instrumentalness:** predice si una canción o audio tiene vocalización. Sonidos como "Ooh" o "Aah" son tratados como instrumentales. Valores cercanos a 1 denotan audios posiblemente sin la presencia de vocalización. Por ejemplo el rap debiese tener un valor de instrumentalness cercano a 0.
    - **Liveness:** detecta la presencia de una audiencia en la grabación. Valores altos de liveness representan alta probabilidad que el audio fue grabado en un concierto.
    """

    st.markdown("<a href='#linkto_top'>Link to top</a>", unsafe_allow_html=True)

if section == "Datos":
    st.header('Datos')

    st.write('En esta sección se muestran los datos con los que se trabajarán y las transformaciones para llegar al dataset final. Los datos se obtuvieron de la\
    platafoma Kaggel y consisten en dato sobre los rankings semanales de canciones (charts), artistas y canciones.')
    
    """
    ### Consideraciones
    Es importante resaltar que los tres datasets utilizados solo constituyen una muestra de los datos originales y que no se pudieron cargar completamente en github
    para su uso. Cada muestra de cada dataset se extrajo tal que después al cruzarlos no se pierde información.
    
    """

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

    st.subheader('Transformación de Data Chart')
    st.write('A continuación se describen las transformaciones de datos realizadas para al dataset **chart** llegar al dataset final con el cual se trabajarán las visualizaciones:')

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


    st.subheader('Transformación de Data Artists')

    st.write('A continuación se describen las transformaciones de datos realizadas para al dataset **artist** llegar al dataset final con el cual se trabajarán las visualizaciones:')

    st.write('Se ajusta el nombre de algunas columnas')
    with st.echo():
        artist = df_artist.rename(columns = {'id':'Artist_id','name':'Artist'})

    st.write('Se genera una columna por cada uno de los posibles 6 géneros a los que puede estar ligado un artista')
    with st.echo():
        for i in range(1,7):
            genero = 'Genre'+str(i)
            artist[genero] = artist['genres'].str.replace("'",'').str.replace('[','').str.replace(']','').str.replace(', ',',').str.lstrip().str.rstrip().str.split(',').apply(lambda x: x[i-1] if len(x) >= i else np.nan)

    st.write('En caso que la lista sea "[" + "''" + "]" ( o [''] como lista) el primer y único elemento siempre será ''. Debemos dejarlo como nan')
    with st.echo():
        artist.loc[artist['Genre1'] == "",'Genre1'] = np.nan

    st.write('Se extraen las columnas de interés')
    with st.echo():
        artist = artist[['Artist_id','Artist','Genre1','Genre2','Genre3','Genre4','Genre5','Genre6']]

    st.write('Se eliminan posibles filas duplicadas')
    with st.echo():
        artist = artist.drop_duplicates()

    st.subheader('Transformación de Data Songs')

    st.write('Se mapean las columnas de este dataframe que hacen mención a las características sonoras de las canciones, las cuales calcula Spotify en base a una\
             métrica que va en el rango [0,1]')
    with st.echo():
        columns_sound = ['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo','mode','key']

    st.write('Se extraen las columnas de interés')
    with st.echo():
        song = df_song[['id','name','id_artists','artists'] + columns_sound + ['time_signature','duration_ms']]

    st.write('Extraemos el id del artista de la primera posición de la lista de artistas ("id_artists"). Asumimos que el primero es el más relevante')
    with st.echo():
        song['id_artists'] = song['id_artists'].str.replace('[','').str.replace(']','').str.replace(' ','').str.split(',').apply(lambda x: x[0]).str.replace("'",'')

    st.write('Extraemos el nombre del artista de la primera posición de la lista de artistas ("artists"). Asumimos que el primero es el más relevante')
    with st.echo():
        song['artists'] = song['artists'].str.replace('[','').str.replace(']','').str.replace(' ','').str.split(',').apply(lambda x: x[0]).str.replace("'",'')

    st.write('Renombramos algunas columnas')
    with st.echo():
        song = song.rename(columns = {'id':'Song_id','name':'Song','id_artists':'Artist_id','artists':'Artist'})

    st.subheader('Dataset Final')

    st.write('Juntamos los datos')
    with st.echo():
        country_final = chart.merge(song, how = 'left', on = 'Song_id')
        country_final = country_final.merge(artist, how = 'left', on = 'Artist_id')

    st.write('Extraemos las columnas relevantes')
    with st.echo():
        country_final = country_final[['Song_x','Artist_x',
                                       'Date','Ano','AnoMes',
                                       'MesDia','Mes','Dia',
                                       'Position','Streams',
                                       'danceability','energy',
                                       'loudness','speechiness',
                                       'acousticness','instrumentalness',
                                       'liveness','valence',
                                       'tempo','mode','key',
                                       'time_signature','duration_ms','Genre1']]

    st.write('Eliminamos cualquier registro que presente valores nulos')
    with st.echo():
        country_final = country_final.dropna()

    st.write('Renombramos algunas columnas')
    with st.echo():
        country_final = country_final.rename(columns = {'Song_x':'Song','Artist_x':'Artist'})
        st.write(country_final.shape)
        st.write(country_final.head(10))
        
        
    if "df" not in st.session_state:
        #st.session_state["name0"] = "initial name"       
        st.session_state["df"] = country_final    

    # add the link at the bottom of each page
    st.markdown("<a href='#linkto_top'>Link to top</a>", unsafe_allow_html=True)

    
if section == "Visualizacion":                  
    st.header('Visualizaciones')

    st.write('En esta sección se muestran las visualizaciones para entender el comportamiento de los charts, artistas y canciones más populares.')
    st.subheader('Pupularidad vs Streams')
    st.write('Una pregunta natural que surge es si la posición en el chart de Spotify se correlaciona con la cantidad de streams que tiene el artista. La\
    respuesta a esto es que si como muestra el siguiente gráfico interactivo. En todos los años analizados se observa que a mejor ranking en el chart se\
    condice con una mayor cantidad promedio de streams por artista. Es un resultado esperado pero interesante de corroborar con los datos.')
    
    #with st.echo():
    df_artist_time = st.session_state.df.groupby(['Ano','Artist'], as_index=False).agg({'Streams':'mean', 'Position':'mean'})

    g1 = alt.Chart(df_artist_time).mark_point().encode(
        x='Position',
        y='Streams',
        color='Ano:N'
    ).properties(
        background = '#f9f9f9',
        title = alt.TitleParams(text = 'Popularidad vs.Streams', 
                                font = 'Ubuntu Mono', 
                                fontSize = 22, 
                                color = '#3E454F', 
                                subtitleFont = 'Ubuntu Mono',
                                subtitleFontSize = 16, 
                                subtitleColor = '#3E454F',
                                anchor = 'middle'
                               )
    ).interactive()

    st.altair_chart(g1, use_container_width=True)
    
    
    st.subheader('Artistas más escuchados')
    st.write('En esta sección se muestran los artistas más escuchados - en términos de streams - de EEUU en Spotify. Los más populares son Post Malone XXXTentation y Drake.\
    En general estos artistas tienen una cantidad de streams invariante en el tiempo, lo que quiere decir que se tienden a repetir año tras año')
  
    #with st.echo():
    artist_popular = st.session_state.df.groupby(['Ano','Artist'], as_index=False).agg({'Streams':'sum'})
    artist_popular = artist_popular.sort_values(['Ano', 'Streams'], ascending=False).groupby(['Ano']).head(20)

    g2 = alt.Chart(artist_popular).mark_circle().encode(
        alt.X('Artist', sort=alt.EncodingSortField(field="Streams", op="sum", order='descending')),
        y='Streams:Q',
        color = 'Ano:N'
    ).properties(
        background = '#f9f9f9',
        title = alt.TitleParams(text = 'Aristas más escuchados', 
        font = 'Ubuntu Mono', 
        fontSize = 22, 
        color = '#3E454F', 
        subtitleFont = 'Ubuntu Mono',
        subtitleFontSize = 16, 
        subtitleColor = '#3E454F',
        anchor = 'middle'
        )
    ).interactive()

    st.altair_chart(g2, use_container_width=True)
    
    st.subheader('Propiedades Sonoras en el tiempo')
    st.write('En esta sección se observa como varían las propiedades sonoras de las canciones más escuchadas en EEUU en el tiempo. Para esto se estudia \
    **danceability**, **energy**, **speechiness**, **acousticness**, **instrumentalness**, **liveness** y **valence**. Para esto se hace un boxplot de cada característica\
    en el tiempo sobre todas las canciones de los rankings. En general todas las métricas tienen un comportamiento constante en el tiempo salvo **danceability** Y \
    **acousticness**, donde el primero tiende a bajar en el tiempo (las canciones más populares tienden a bajar su característica de bailables) y la segunda sube en el tiempo\
    (las canciones más populares tienden a ser más acústicas).')

    #with st.echo():
    sonido = st.session_state.df[['Ano','Position','danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence']].groupby(['Ano','Position'], as_index=False).mean()

    g3 = alt.Chart(sonido).transform_fold(['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence'], as_=['key', 'value']).mark_boxplot(
    ).encode(x='Ano:N',
             y='value:Q',
             row = 'key:N'
            ).properties(
        width=250,
        height=120
    ).properties(
        background = '#f9f9f9',
        title = alt.TitleParams(text = 'Distribución de propiedades sonoras', 
                                font = 'Ubuntu Mono', 
                                fontSize = 22, 
                                color = '#3E454F', 
                                subtitleFont = 'Ubuntu Mono',
                                subtitleFontSize = 16, 
                                subtitleColor = '#3E454F',
                                anchor = 'middle'
        )
    )
    
    st.altair_chart(g3, use_container_width=True)
    
    st.subheader('Comportamiento de Streams vs. Popularidad en Spotify')
    st.write('El siguiente gráfico es interactivo. Se puede seleccionar un área del gráfico superior de Streams vs. Popularidad y ver como cambian las características\
             sonoras propmedio de los artistas pertenencientes al área de selección. Es importante notar que el comportamiento de la curva de Streams vs. Popularidad \
             se puede asociar a una ley potencia tal que muy pocos artistas condensan la gran mayoría de reproducciones. Esto no cambia en el tiempo. Al ver las\
             características (mover selección) desde los más populares a los menos populares, se puede notar que en promedio **danceability** y **energy** alcanzan valores\
             más altos a medida que nos movemos a posiciones más altas en el ranking. Esto quiere decir que podemos asociar la popularidad a artistas y canciones con un\
             mood positivo que haga sentir a la gente más contenta talvez.')
    
    #with st.echo():
    streams = st.session_state.df[['Ano','Position','Streams','danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence']].groupby(['Ano','Position']).mean().reset_index()
    streams = streams.set_index(['Ano','Position','Streams'])
    streams = streams.stack().reset_index(name = 'Valor').rename(columns={'level_3':'Variable'})

    interval = alt.selection_interval()

    scatter = alt.Chart(streams).mark_line(filled=False).encode(
        x = 'Position',
        y = 'Streams',
        color = alt.Color('Ano:N', scale=alt.Scale(range=['red','blue','lightgreen','black'])),
        size=alt.Size(scale=alt.Scale(zero=False))
    ).properties(
        selection = interval
    )

    bar = alt.Chart(streams).mark_bar().encode(
        x = alt.X('mean(Valor)', scale=alt.Scale(domain=[0, 1.0]), title = 'Valor'),
        y = alt.Y('Variable',title='Característica Sonora')
    ).transform_filter(
        interval
    )

    g4 = alt.vconcat(scatter,bar).properties(
                background = '#f9f9f9',
                title = alt.TitleParams(text = 'Comportamiento de Streams vs Posición en el Ranking Spotify por año', 
                                        font = 'Ubuntu Mono', 
                                        fontSize = 22, 
                                        color = '#3E454F', 
                                        subtitleFont = 'Ubuntu Mono',
                                        subtitleFontSize = 16, 
                                        subtitleColor = '#3E454F',
                                        anchor = 'middle'
                                        )
                )
        
    st.altair_chart(g4, use_container_width=True)
    
    st.subheader('Géneros Musicales más escuchados en Spotify en el tiempo')
    st.write('Este es un gráfico dinámico: se muestran los géneros musicales con mas reproducciones a lo largo del tiempo analizado y es posible ver las características\
    sonoras promedio de uno o un conjunto de ellos. Es interesante notar, que de la muestra tomada, los géneros más populares en EEUU son **pop**, **rap** o **hip hop**. Esto\
    hace sentido al tomar en cuenta la cultura musical de ese país, donde el desarrollo de dichos géneros ha sido muy relevante desde la década del 80. Un resultado esperado también\
    es que los atributos más importantes de estos géneros son precisamente **danceability** y **energy**, ya que están asociados a beats sonoros fuertes que tienden a generar un\
    mood positivo en la gente.')
    
    #with st.echo():
    genres = st.session_state.df[['Ano',
                            'Genre1',
                            'Streams',
                            'danceability',
                            'energy',
                            'speechiness',
                            'acousticness',
                            'instrumentalness',
                            'liveness',
                            'valence']].groupby(['Ano','Genre1']).agg({'Streams':'sum',
                                                                       'danceability':'mean',
                                                                       'energy':'mean',
                                                                       'speechiness':'mean',
                                                                       'acousticness':'mean',
                                                                       'instrumentalness':'mean',
                                                                       'liveness':'mean',
                                                                       'valence':'mean'
                                                                      }).reset_index()

    genres = genres.set_index(['Ano','Genre1','Streams'])
    genres = genres.stack().reset_index(name = 'Valor').rename(columns={'level_3':'Variable'})

    interval1 = alt.selection_interval(encodings = ['y'])
    interval2 = alt.selection_interval(encodings = ['y'])
    interval3 = alt.selection_interval(encodings = ['y'])
    interval4 = alt.selection_interval(encodings = ['y'])

    width = 200
    height = 250
    height2 = 100

    b1 = alt.Chart(genres).transform_window(
        rank='rank(sum(Streams))',
        sort=[alt.SortField('Streams', order='descending')]
    ).transform_filter(
        alt.datum.rank <= 1000
    ).mark_bar().encode(
        x = alt.X('mean(Streams):Q', scale=alt.Scale(domain=[0, 150000000]), title='Cantidad de Streams',axis=alt.Axis(labelExpr='datum.value / 1E6 + "M"')),
        y = alt.Y('Genre1:N', sort = '-x', title='Género Musical'),
        column = alt.Column('Ano:N', title = "")
    ).transform_filter(
        alt.FieldEqualPredicate(field='Ano', equal=2017)
    ).properties(
        width=width,
        height=height,
        selection = interval1
    )

    b2 = alt.Chart(genres).transform_window(
        rank='rank(sum(Streams))',
        sort=[alt.SortField('Streams', order='descending')]
    ).transform_filter(
        alt.datum.rank <= 1000
    ).mark_bar().encode(
      x = alt.X('mean(Streams):Q', scale=alt.Scale(domain=[0, 150000000]), title='Cantidad de Streams',axis=alt.Axis(labelExpr='datum.value / 1E6 + "M"')),
        y = alt.Y('Genre1:N', sort = '-x', title='Género Musical'),
        column = alt.Column('Ano:N', title = "")
    ).transform_filter(
        alt.FieldEqualPredicate(field='Ano', equal=2018)
    ).properties(
        width=width,
        height=height,
        selection = interval2
    )

    b3 = alt.Chart(genres).transform_window(
        rank='rank(sum(Streams))',
        sort=[alt.SortField('Streams', order='descending')]
    ).transform_filter(
        alt.datum.rank <= 1000
    ).mark_bar().encode(
        x = alt.X('mean(Streams):Q', scale=alt.Scale(domain=[0, 150000000]), title='Cantidad de Streams',axis=alt.Axis(labelExpr='datum.value / 1E6 + "M"')),
        y = alt.Y('Genre1:N', sort = '-x', title='Género Musical'),
        column = alt.Column('Ano:N', title = "")
    ).transform_filter(
        alt.FieldEqualPredicate(field='Ano', equal=2019)
    ).properties(
        width=width,
        height=height,
        selection = interval3
    )

    b4 = alt.Chart(genres).transform_window(
        rank='rank(sum(Streams))',
        sort=[alt.SortField('Streams', order='descending')]
    ).transform_filter(
        alt.datum.rank <= 1000
    ).mark_bar().encode(
        x = alt.X('mean(Streams):Q', scale=alt.Scale(domain=[0, 150000000]), title='Cantidad de Streams',axis=alt.Axis(labelExpr='datum.value / 1E6 + "M"')),
        y = alt.Y('Genre1:N', sort = '-x', title='Género Musical'),
        column = alt.Column('Ano:N', title = "")
    ).transform_filter(
        alt.FieldEqualPredicate(field='Ano', equal=2020)
    ).properties(
        width=width,
        height=height,
        selection = interval4
    )

    feat_b1 = alt.Chart(genres).mark_bar().encode(
        x = alt.X('average(Valor)', scale=alt.Scale(domain=[0, 1.0]), title='Valor'),
        y = alt.Y('Variable',title='Característica Sonora'),
        color = 'Variable:N'
    ).transform_filter(
        alt.FieldEqualPredicate(field='Ano', equal=2017)
    ).transform_filter(
        interval1
    ).properties(
        width=width,
        height=height2
    )

    feat_b2 = alt.Chart(genres).mark_bar().encode(
        x = alt.X('average(Valor)', scale=alt.Scale(domain=[0, 1.0]),title='Valor'),
        y = alt.Y('Variable',title='Característica Sonora'),
        color = 'Variable:N'
    ).transform_filter(
        alt.FieldEqualPredicate(field='Ano', equal=2018)
    ).transform_filter(
        interval2
    ).properties(
        width=width,
        height=height2
    )

    feat_b3 = alt.Chart(genres).mark_bar().encode(
        x = alt.X('average(Valor)', scale=alt.Scale(domain=[0, 1.0]),title='Valor'),
        y = alt.Y('Variable',title='Característica Sonora'),
        color = 'Variable:N'
    ).transform_filter(
        alt.FieldEqualPredicate(field='Ano', equal=2019)
    ).transform_filter(
        interval3
    ).properties(
        width=width,
        height=height2
    )

    feat_b4 = alt.Chart(genres).mark_bar().encode(
        x = alt.X('average(Valor)', scale=alt.Scale(domain=[0, 1.0]),title='Valor'),
        y = alt.Y('Variable',title='Característica Sonora'),
        color = alt.Color('Variable:N',
                            legend=alt.Legend(
                                orient='none',
                                legendX=450, legendY=530,
                                direction='horizontal',
                                titleAnchor='middle',
                                title = '')
        )
    ).transform_filter(
        alt.FieldEqualPredicate(field='Ano', equal=2020)
    ).transform_filter(
        interval4
    ).properties(
        width=width,
        height=height2
    )


    g5 = alt.hconcat(alt.vconcat(b1,feat_b1), alt.vconcat(b2,feat_b2), alt.vconcat(b3,feat_b3), alt.vconcat(b4,feat_b4)).properties(
        background = '#f9f9f9',
        title = alt.TitleParams(text = 'Géneros Musicales más escuchados en Spotify en el tiempo', 
                                font = 'Ubuntu Mono', 
                                fontSize = 22, 
                                color = '#3E454F', 
                                subtitleFont = 'Ubuntu Mono',
                                subtitleFontSize = 16, 
                                subtitleColor = '#3E454F',
                                anchor = 'middle'
                                )
        )
        
        
        
    st.altair_chart(g5, use_container_width=True)
        
    st.subheader('Aristas más populares de Spotify por tipo de género')
    st.write('Este también es un gráfico interactivo donde se muestra la relación entre reproducciones y posiciones en el ranking de los diferentes artistas\
    agrupados por género musical. En general no se observa una diferencia en los años y los distintos géneros no muestran un patrón que indique que el género del tipo A\
    esté mejor ubicado en el ranking que el género B. Solo se ratifica la conclusión sobre que los géneros más populares son rock, pop y hip hop. Es posible marcar cada\
    género para aislar a los artistas que lo componen. Al pasar el mouse por cada punto obtiene el nombre del arista asociado. Algunos tienden a repetirse en las mejores\
    posiciones como Mariah Carey, CardiB, Drake, Post Malone.')

    genre = st.session_state.df[['Ano','Artist','Genre1','Position','Streams']].groupby(['Ano','Artist','Genre1']).agg({'Position':'mean','Streams':'sum'}).reset_index()
    genre = genre[genre['Ano'] <= 2020]
    genre['Genre'] = genre.Genre1
    genre.loc[genre['Genre'].str.contains('rap'), 'Genre'] = 'hip hop/rap'
    genre.loc[genre['Genre'].str.contains('hip hop'), 'Genre'] = 'hip hop/rap'
    genre.loc[genre['Genre'].str.contains('rock'), 'Genre'] = 'rock'
    genre.loc[genre['Genre'].str.contains('pop'), 'Genre'] = 'pop'
    genre.loc[genre['Genre'].str.contains('electro'), 'Genre'] = 'electro/dance'
    genre.loc[genre['Genre'].str.contains('dance'), 'Genre'] = 'electro/dance'
    genre.loc[~genre['Genre'].isin(['electro/dance','hip hop/rap','rock','pop','electro','punk','funk','dance','country','soul','metal']), 'Genre'] = 'otro'
    
    selection = alt.selection_multi(fields=['Genre'])
    color = alt.condition(selection,
                          alt.Color('Genre:N', legend=None,scale=alt.Scale(scheme='category10')),
                          alt.value('lightgray')
                         )

    g6 = alt.Chart(genre).mark_point(filled = True, size = 50).encode(
        x = alt.X('Position:Q',axis=alt.Axis(tickCount=3)),
        y = alt.Y('Streams:Q', scale = alt.Scale(type = 'log'),axis=alt.Axis(tickCount=2)),
        #color = alt.Color('Genre', scale=alt.Scale(scheme='category10')),
        color = color,
        row = 'Ano:N',
        tooltip = 'Artist:N'
    ).properties(
            width=250,
            height=120
    )
    
    legend = alt.Chart(genre).mark_point().encode(
        y=alt.Y('Genre:N', axis=alt.Axis(orient='right')),
        color=color
    ).add_selection(
        selection
    )
    
    
    g6 = alt.hconcat(g6, legend).properties(
        background = '#f9f9f9',
        title = alt.TitleParams(text = 'Aristas más populares de Spotify por tipo de género', 
                                font = 'Ubuntu Mono', 
                                fontSize = 22, 
                                color = '#3E454F', 
                                subtitleFont = 'Ubuntu Mono',
                                subtitleFontSize = 16, 
                                subtitleColor = '#3E454F',
                                anchor = 'middle'
                                )
        )
    
    st.altair_chart(g6, use_container_width=True)

    st.markdown("<a href='#linkto_top'>Link to top</a>", unsafe_allow_html=True)
    
