"""Cinq façons de recommander un film, avec une même interface.

Chaque recommandeur s'ajuste sur une table de notes (utilisateur, film, note, genres) puis
renvoie, pour un profil de notes, les k films non vus au meilleur score.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEUIL_AIME = 4.0  # une note >= 4 signifie « j'ai aimé »


def charger(chemin="data/notes_films.csv"):
    d = pd.read_csv(chemin).rename(columns={"userId": "utilisateur", "title": "film", "rating": "note"})
    return d.drop_duplicates(["utilisateur", "film"], keep="last")


class Recommandeur:
    nom = "?"

    def ajuster(self, notes):
        self.notes = notes
        self.films = sorted(notes.film.unique())
        self.matrice = notes.pivot_table(index="utilisateur", columns="film", values="note").reindex(columns=self.films)
        return self

    def scores(self, profil: dict) -> pd.Series:
        raise NotImplementedError

    def centre(self, profil):
        """Centre un profil : par sa propre moyenne s'il est varié, sinon par la moyenne globale."""
        v = pd.Series(profil, dtype=float)
        return v.mean() if v.nunique() > 1 else self.notes.note.mean()

    def recommander(self, profil: dict, k=5) -> pd.Series:
        s = self.scores(profil)
        # Départage des ex aequo par le nombre de notes reçues (film plus connu d'abord)
        s = s + 1e-9 * self.notes.film.value_counts().reindex(s.index).fillna(0)
        s = s.drop(labels=list(profil), errors="ignore")
        return s.sort_values(ascending=False).head(k)


class Popularite(Recommandeur):
    """Référence : les films les mieux notés par le plus grand nombre (moyenne bayésienne)."""
    nom = "Popularité (référence)"

    def ajuster(self, notes):
        super().ajuster(notes)
        g = notes.groupby("film").note
        m, c = notes.note.mean(), g.count().median()
        self._s = (g.sum() + c * m) / (g.count() + c)
        return self

    def scores(self, profil):
        return self._s.copy()


class Contenu(Recommandeur):
    """Filtrage par contenu : films aux genres proches de ceux que l'utilisateur a aimés (TF-IDF)."""
    nom = "Contenu (genres)"

    def ajuster(self, notes):
        super().ajuster(notes)
        genres = notes.drop_duplicates("film").set_index("film").genres.reindex(self.films)
        tfidf = TfidfVectorizer(tokenizer=lambda x: x.split("|"), token_pattern=None, lowercase=False)
        self.sim = pd.DataFrame(cosine_similarity(tfidf.fit_transform(genres)), index=self.films, columns=self.films)
        return self

    def scores(self, profil):
        aimes = [f for f, n in profil.items() if n >= SEUIL_AIME and f in self.sim] or \
                [f for f in profil if f in self.sim]
        if not aimes:
            return pd.Series(0.0, index=self.films)
        poids = np.array([profil[f] for f in aimes])
        return (self.sim.loc[aimes].T @ poids) / poids.sum()


class VoisinsUtilisateurs(Recommandeur):
    """Filtrage collaboratif : moyenne des notes des 20 utilisateurs aux goûts les plus proches."""
    nom = "Collaboratif (utilisateurs voisins)"

    def __init__(self, k=20):
        self.k = k

    def ajuster(self, notes):
        super().ajuster(notes)
        self.centree = self.matrice.sub(self.matrice.mean(axis=1), axis=0).fillna(0)
        return self

    def scores(self, profil):
        v = pd.Series(profil).reindex(self.films)
        u = (v - self.centre(profil)).fillna(0).values.reshape(1, -1)
        sims = cosine_similarity(u, self.centree.values).ravel()
        idx = np.argsort(-sims)[: self.k]
        voisins = self.centree.iloc[idx]
        w = np.clip(sims[idx], 0, None)
        if w.sum() == 0:
            return pd.Series(0.0, index=self.films)
        return pd.Series((voisins.values * w[:, None]).sum(0) / w.sum(), index=self.films)


class VoisinsFilms(Recommandeur):
    """Filtrage collaboratif « item-item » : les films notés de façon similaire par les utilisateurs."""
    nom = "Collaboratif (films similaires)"

    def ajuster(self, notes):
        super().ajuster(notes)
        c = self.matrice.sub(self.matrice.mean(axis=1), axis=0).fillna(0)
        self.sim = pd.DataFrame(cosine_similarity(c.T), index=self.films, columns=self.films)
        return self

    def scores(self, profil):
        vus = [f for f in profil if f in self.sim]
        if not vus:
            return pd.Series(0.0, index=self.films)
        m = self.centre(profil)
        ecarts = np.array([profil[f] - m for f in vus])
        return self.sim.loc[vus].T @ ecarts / (np.abs(self.sim.loc[vus]).sum(0) + 1e-9)


class SVD(Recommandeur):
    """Factorisation de matrice : goûts et films projetés dans un espace de 10 facteurs latents."""
    nom = "Factorisation (SVD)"

    def __init__(self, facteurs=10):
        self.facteurs = facteurs

    def ajuster(self, notes):
        super().ajuster(notes)
        self.moy = self.matrice.mean(axis=1)
        c = self.matrice.sub(self.moy, axis=0).fillna(0)
        self.svd = TruncatedSVD(self.facteurs, random_state=42).fit(c.values)
        return self

    def scores(self, profil):
        v = pd.Series(profil).reindex(self.films)
        u = (v - self.centre(profil)).fillna(0).values.reshape(1, -1)
        return pd.Series(self.svd.inverse_transform(self.svd.transform(u)).ravel(), index=self.films)


TOUS = [Popularite, Contenu, VoisinsUtilisateurs, VoisinsFilms, SVD]


class Hybride(Recommandeur):
    """Mélange popularité + voisins : la sécurité du populaire, la personnalisation du collaboratif."""
    nom = "Hybride (popularité + voisins)"

    def __init__(self, poids_perso=0.5):
        self.w = poids_perso

    def ajuster(self, notes):
        super().ajuster(notes)
        self.pop = Popularite().ajuster(notes)
        self.vois = VoisinsUtilisateurs().ajuster(notes)
        return self

    def scores(self, profil):
        def z(s):
            return (s - s.mean()) / (s.std() + 1e-9)
        return (1 - self.w) * z(self.pop.scores(profil)) + self.w * z(self.vois.scores(profil))


TOUS = [Popularite, Contenu, VoisinsUtilisateurs, VoisinsFilms, SVD, Hybride]
