"""Application de démonstration : streamlit run app.py

Choisissez quelques films aimés (et éventuellement détestés), puis comparez les recommandations
des six méthodes. Les affiches sont affichées si la variable d'environnement TMDB_API_KEY est
définie (clé gratuite sur themoviedb.org) ; aucune clé n'est stockée dans le code.
"""

import os
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
import recommandeurs as rc  # noqa: E402

st.set_page_config(page_title="Recommandation de films", page_icon="🎬", layout="wide")


@st.cache_resource
def modeles():
    notes = rc.charger()
    return notes, {c().nom: c().ajuster(notes) for c in rc.TOUS}


@st.cache_data(show_spinner=False)
def affiche(titre_film: str):
    cle = os.environ.get("TMDB_API_KEY")
    if not cle:
        return None
    nom, annee = titre_film.rsplit(" (", 1)
    if ", The" in nom:
        nom = "The " + nom.replace(", The", "")
    try:
        r = requests.get("https://api.themoviedb.org/3/search/movie", timeout=5,
                         params={"api_key": cle, "query": nom, "year": annee.rstrip(")")}).json()
        chemin = (r.get("results") or [{}])[0].get("poster_path")
        return f"https://image.tmdb.org/t/p/w342{chemin}" if chemin else None
    except requests.RequestException:
        return None


notes, recos = modeles()
films = sorted(notes.film.unique())
genres = notes.drop_duplicates("film").set_index("film").genres

st.title("🎬 Quel film regarder ce soir ?")
st.caption(f"{notes.utilisateur.nunique()} spectateurs · {len(films)} films · {len(notes):,} notes".replace(",", " "))

with st.sidebar:
    st.header("Vos goûts")
    aimes = st.multiselect("Films que vous avez aimés", films,
                           default=["Toy Story (1995)", "Pinocchio (1940)", "Dumbo (1941)"])
    detestes = st.multiselect("Films que vous n'avez pas aimés", [f for f in films if f not in aimes],
                              default=["Heat (1995)"])
    methode = st.radio("Méthode de recommandation", list(recos), index=len(recos) - 1)
    k = st.slider("Nombre de recommandations", 3, 10, 5)

profil = {f: 5.0 for f in aimes} | {f: 1.5 for f in detestes}
if not profil:
    st.info("Choisissez au moins un film dans le panneau de gauche.")
    st.stop()

st.subheader(methode)
st.write(rc_doc if (rc_doc := type(recos[methode]).__doc__) else "")
resultat = recos[methode].recommander(profil, k)
colonnes = st.columns(len(resultat))
for col, (film, score) in zip(colonnes, resultat.items()):
    with col:
        url = affiche(film)
        if url:
            st.image(url, use_container_width=True)
        st.markdown(f"**{film}**")
        st.caption(genres.get(film, "").replace("|", " · "))
        st.progress(float(min(1.0, max(0.0, (score - resultat.min()) / (resultat.max() - resultat.min() + 1e-9)))))

with st.expander("Comparer les six méthodes"):
    for nom, r in recos.items():
        st.markdown(f"**{nom}** : " + " · ".join(r.recommander(profil, k).index))
