import ast

import pandas as pd
import streamlit as st

st.set_page_config(page_title='BigDL Movie Recommendation System',
                   layout='wide')

st.image("https://bigdl-project.github.io/img/bigdl_logo.png", width=300)
df = pd.read_csv("resources/cache.csv")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.subheader('BigDL 🧱')
st.write("""
BigDL is a distributed deep learning library for Apache Spark; with BigDL, users can write their deep learning applications as standard Spark programs, which can directly run on top of existing Spark or Hadoop clusters.
""")

st.subheader('Description 🎬')
st.markdown("""
 Recommendation System is a filtration program whose prime goal is to predict the “rating” or “preference” of a user towards a domain-specific item or item. 
 In our case, this domain-specific item is a movie, therefore the main focus of our recommendation system is to filter and predict only those movies which a user would prefer given some data about the user him or herself
""")

st.subheader('Coursework  🍿')
st.markdown("""
 This project is part of Machine Learning in Production(11745) - Assignment 3 at Carnegie Mellon University
 - GiHub : [BigDL Movie Recommendation System](https://github.com/akshaybahadur21/bigDL-Movie-Rec)
 - Blog Post : [Medium | BigDL Movie Recommendation](https://github.com/akshaybahadur21/bigDL-Movie-Rec)
""")

st.markdown("""
##### Made with ❤️ and 🦙 by Akshay and Ayush
""")

if st.button('Press to generate insights from data'):
    st.markdown('##### Number of unique users')
    st.info(len(df))

    st.markdown('##### Number of unique Movies')
    movie_list = []


    def f(x):
        list_obj = ast.literal_eval(x['name'])
        movie_list.extend(list_obj)


    df.apply(lambda x: f(x), axis=1)
    st.info(len(movie_list))

    st.markdown('##### Sample Movie Names')
    st.info((movie_list)[0:10])

    st.markdown('##### Number of unique Genres')
    set_genre = set()


    def f(x):
        list_obj = ast.literal_eval(x['Genre'])
        set_genre.update(list_obj)

    df.apply(lambda x: f(x), axis=1)
    st.info(len(set_genre))
    st.markdown('##### Sample Genre Names ')
    st.info((list(set_genre))[0:10])

    st.markdown('##### Glimpse of the cached Recommendations')
    st.write(df.head())

    st.markdown('##### Layers of Neural Network')
    st.image("resources/layers.png", width=1200)

with st.sidebar.header('1. Select UserID'):
    userID = st.sidebar.slider('UserID', min_value=1, max_value=6040, step=1)

with st.sidebar.header('2. Select Multiple Genre'):
    genres = st.sidebar.multiselect(
        'What are your favorite genres',
        ['Musical', 'Romance', 'Drama', 'Animation', 'Comedy', 'Fantasy', 'Children', 'Action', 'Sci-Fi', 'Horror',
         'Adventure', 'Thriller'])

with st.sidebar.header('3. Number of movie recommendations'):
    num_rec = st.sidebar.slider('Number of Recommendations', min_value=1, max_value=20, step=1)

if st.sidebar.button('Press to generate recommendations'):
    st.subheader("Generating recommendations for userID : " + str(userID))
    filtered_df = df[df['user'] == userID]

    def f(x):
        return ast.literal_eval(x['name'])
    filtered_df['name'] = filtered_df.apply(lambda x: f(x), axis=1)


    def f(x):
        return ast.literal_eval(x['Genre'])
    filtered_df['Genre'] = filtered_df.apply(lambda x: f(x), axis=1)

    res_list = []
    def f(x):
        for i, n in enumerate(x['Genre']):
            for g in genres:
                if g in n:
                    res_list.append(x['name'][i])
    if genres != None and len(genres) > 0:
        filtered_df.apply(lambda x: f(x), axis=1)
        st.table(res_list[0:num_rec])
    else:
        st.table(filtered_df.name.tolist()[0][0:num_rec])
