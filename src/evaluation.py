"""Évaluation hors ligne : on cache une partie des films aimés et on regarde s'ils sont recommandés."""

import numpy as np
import pandas as pd

from recommandeurs import SEUIL_AIME, TOUS


def decouper(notes, part_cachee=0.3, min_aimes=3, graine=0):
    """Pour chaque utilisateur ayant aimé au moins min_aimes films, cache 30 % de ses films aimés."""
    rng = np.random.default_rng(graine)
    caches = []
    for u, g in notes.groupby("utilisateur"):
        aimes = g[g.note >= SEUIL_AIME].film.tolist()
        if len(aimes) >= min_aimes:
            n = max(1, int(round(part_cachee * len(aimes))))
            caches += [(u, f) for f in rng.choice(aimes, n, replace=False)]
    cache = pd.DataFrame(caches, columns=["utilisateur", "film"])
    cle = notes.set_index(["utilisateur", "film"]).index
    masque = cle.isin(pd.MultiIndex.from_frame(cache))
    return notes[~masque], cache


def evaluer(notes, k=5, graines=(0, 1, 2, 3, 4)):
    lignes = []
    for g in graines:
        app, cache = decouper(notes, graine=g)
        a_trouver = cache.groupby("utilisateur").film.apply(set)
        profils = app.groupby("utilisateur").apply(lambda d: dict(zip(d.film, d.note)), include_groups=False)
        for classe in TOUS:
            r = classe().ajuster(app)
            prec, rapp, touche, recommandes = [], [], [], set()
            for u, cibles in a_trouver.items():
                reco = list(r.recommander(profils.get(u, {}), k).index)
                recommandes.update(reco)
                bons = len(set(reco) & cibles)
                prec.append(bons / k)
                rapp.append(bons / len(cibles))
                touche.append(bons > 0)
            lignes.append({"methode": r.nom, "graine": g, "precision_at_k": np.mean(prec),
                           "rappel_at_k": np.mean(rapp), "taux_reussite": np.mean(touche),
                           "couverture_catalogue": len(recommandes) / len(r.films),
                           "utilisateurs_evalues": len(a_trouver)})
    res = pd.DataFrame(lignes)
    return res.groupby("methode", sort=False).agg(
        precision_at_k=("precision_at_k", "mean"), rappel_at_k=("rappel_at_k", "mean"),
        taux_reussite=("taux_reussite", "mean"), taux_reussite_et=("taux_reussite", "std"),
        couverture_catalogue=("couverture_catalogue", "mean"),
        utilisateurs_evalues=("utilisateurs_evalues", "mean")).reset_index()
