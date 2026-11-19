"""Tests rapides (pytest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import evaluation  # noqa: E402
import recommandeurs as rc  # noqa: E402

NOTES = rc.charger()
PROFIL = {"Toy Story (1995)": 5.0, "Pinocchio (1940)": 4.5, "Heat (1995)": 1.5}


def test_aucun_film_deja_vu_n_est_recommande():
    for c in rc.TOUS:
        reco = c().ajuster(NOTES).recommander(PROFIL, 5)
        assert len(reco) == 5
        assert not set(reco.index) & set(PROFIL)


def test_profil_a_un_seul_film_donne_des_recommandations_utiles():
    # Régression : un profil d'une seule note ne doit pas produire un ordre alphabétique
    reco = rc.VoisinsUtilisateurs().ajuster(NOTES).recommander({"Toy Story (1995)": 5.0}, 5)
    assert list(reco.index) != sorted(reco.index)


def test_contenu_recommande_des_genres_proches():
    reco = rc.Contenu().ajuster(NOTES).recommander({"Toy Story (1995)": 5.0}, 3)
    genres = NOTES.drop_duplicates("film").set_index("film").genres
    assert all("Children" in genres[f] or "Animation" in genres[f] for f in reco.index)


def test_decoupage_cache_uniquement_des_films_aimes():
    app, cache = evaluation.decouper(NOTES)
    fusion = cache.merge(NOTES, on=["utilisateur", "film"])
    assert (fusion.note >= rc.SEUIL_AIME).all()
    assert len(app) + len(cache) == len(NOTES)
