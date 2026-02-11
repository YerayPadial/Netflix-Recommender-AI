import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. CONFIGURACIÓN DE LA PÁGINA

# He configurado el título de la pestaña y he activado el layout 'wide' 
# para aprovechar todo el ancho de la pantalla y dar una imagen profesional.
st.set_page_config(
    page_title="Sistema de Recomendación AI",
    layout="wide"
)


# 2. CARGA DE DATOS Y MODELO

# He utilizado el decorador @st.cache_resource. 
# Mi intención con esto es que el modelo y los datos pesados se carguen solo 
# una vez en la memoria caché, evitando que la web se vuelva lenta recargando 
# todo cada vez que pulso un botón.
@st.cache_resource
def load_data():
    """
    Función que carga el modelo entrenado y los datos procesados.
    """
    try:
        # Aquí abro el archivo .pkl que generé previamente en mi Google Colab
        with open('recommender_system.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except FileNotFoundError:
        return None

# Ejecuto la carga de mis artefactos
data = load_data()

# He añadido esta validación de seguridad para avisar si falta el archivo
if data is None:
    st.error("Error: No he encontrado el archivo 'recommender_system.pkl'. Asegúrate de que está en la misma carpeta.")
    st.stop()

# Desempaqueto los datos del diccionario para usarlos cómodamente en el resto del código
kmeans_model = data["model"]
scaler = data["scaler"]
user_profiles = data["user_profiles"]
original_movies = data["original_movies"]
original_ratings = data["original_ratings"]


# 3. Logica de recomendación

def recomendar_peliculas(user_id, top_n=5):
    """
    Lógica inteligente: Recomienda lo popular del cluster excluyendo lo que ya se ha visto.
    Ahora también devuelve el historial del usuario.
    """
    # 1. Validación de usuario
    # Primero compruebo si el ID que he metido existe en mi matriz de entrenamiento.
    if user_id not in user_profiles.index:
        return None, "Usuario Nuevo o No Encontrado", None

    # 2. Identificar el Cluster
    # Consulto a qué cluster asignó mi algoritmo K-Means a este usuario.
    user_cluster = user_profiles.loc[user_id, 'Cluster_KMeans']
    
    # 3. Identificar usuarios del mismo grupo
    # Filtro todos los usuarios que comparten el mismo "ID de Cluster".
    users_in_cluster = user_profiles[user_profiles['Cluster_KMeans'] == user_cluster].index

    # 4. Obtener lo que YA ha visto el usuario activo
    # Busco qué películas ya ha visto el usuario.
    peliculas_vistas_ids = original_ratings[original_ratings['userId'] == user_id]['movieId'].tolist()

    # Creo un DataFrame con los títulos de lo que ya ha visto para mostrarlo en la web.
    movies_watched_df = original_movies[original_movies['movieId'].isin(peliculas_vistas_ids)][['title', 'genres']]

    # 5. Analizar el Cluster
    # Extraigo todas las votaciones que han hecho los "vecinos" de mi cluster.
    cluster_ratings = original_ratings[original_ratings['userId'].isin(users_in_cluster)]

    # Calculo métricas de popularidad (Nota media y Cantidad de votos) agrupando por película.
    recs_stats = cluster_ratings.groupby('movieId').agg(
        rating_mean=('rating', 'mean'),
        rating_count=('rating', 'count')
    )

    # 6. condiciones simultáneas:
    # A) Que la película tenga al menos 5 votos en el grupo (para asegurar calidad estadística).
    # B) Que la película NO esté en mi lista de "vistas" (para asegurar novedad).
    recs_filtradas = recs_stats[
        (recs_stats['rating_count'] >= 5) & 
        (~recs_stats.index.isin(peliculas_vistas_ids)) 
    ]

    # si mis filtros son muy estrictos y no sale nada,
    # relajo la condición de votos mínimos pero mantengo que no sea una película vista.
    if recs_filtradas.empty:
        recs_filtradas = recs_stats[~recs_stats.index.isin(peliculas_vistas_ids)]

    # Ordeno los resultados por la mejor nota media y cojo solo el Top N solicitado.
    top_ids = recs_filtradas.sort_values(by='rating_mean', ascending=False).head(top_n).index

    # Cruzo los IDs con el dataframe de películas para obtener los Títulos y Géneros legibles.
    recommendations = original_movies[original_movies['movieId'].isin(top_ids)][['title', 'genres']]
    
    # Ahora devuelvo 3 cosas: Recomendaciones, Cluster ID e Historial Visto
    return recommendations, user_cluster, movies_watched_df


# 4. Frontend


st.title("Motor de Recomendación Inteligente (K-Means)")
st.markdown("""
Esta aplicación utiliza algoritmos de **Clustering** para agrupar usuarios. 
A diferencia de un sistema simple, aquí se detecta gente con gustos iguales
y se recomienda las joyas ocultas que ellos aman y tú aún no has visto.
""")

st.markdown("---")

# He dividido la pantalla en dos columnas (1 parte para controles, 3 partes para resultados)
col_control, col_display = st.columns([1, 3])

with col_control:
    st.header("Panel de Control")
    # He creado un selector con todos los usuarios disponibles en el dataset.
    user_list = user_profiles.index.tolist()
    selected_user = st.selectbox("Selecciona ID Usuario:", user_list)
    
    # Slider para que el usuario decida cuántas recomendaciones quiere ver.
    num_recs = st.slider("¿Cuántas recomendaciones?", 1, 10, 5)
    
    predict_btn = st.button("Analizar y Recomendar", use_container_width=True)

if predict_btn:
    # He añadido un spinner para mejorar la experiencia de usuario mientras se procesan los datos.
    with st.spinner('Consultando la "Mente Colmena" del Cluster...'):
        # Ahora desempaqueto 3 variables en lugar de 2
        rec_df, cluster_id, watched_df = recomendar_peliculas(selected_user, top_n=num_recs)

    if rec_df is not None and not rec_df.empty:
        with col_display:
            # Muestro el Cluster asignado para dar feedback de que la IA ha funcionado.
            st.success(f"**Usuario identificado en el Cluster #{int(cluster_id)}**")
            st.info("Estrategia: *Filtrado Colaborativo basado en Usuarios (User-Based Clustering)*")
            
            # Uso un expander para que no ocupe mucho sitio visualmente si la lista es larga
            with st.expander(f"Ver historial de películas vistas por Usuario {selected_user} ({len(watched_df)})"):
                st.dataframe(
                    watched_df, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "title": "Película",
                        "genres": "Género"
                    }
                )

            st.divider()
            
            # Utilizo st.dataframe con 'use_container_width' para que la tabla quede estética y profesional.
            st.subheader(f"Top {num_recs} Recomendaciones para ti")
            st.dataframe(
                rec_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "title": "Título de la Película",
                    "genres": "Géneros"
                }
            )

    else:
        st.warning("El usuario existe, pero no hemos encontrado recomendaciones nuevas con los filtros actuales.")