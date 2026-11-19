"""Évaluation complète et figures : python run_all.py (depuis la racine du dépôt)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np  # noqa: E402

import evaluation  # noqa: E402
import recommandeurs as rc  # noqa: E402
from style import AQUA, BLEU, BLEU_CLAIR, GRIS, INK, INK2, plt, sauver, titre  # noqa: E402

Path("docs/img").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True)

notes = rc.charger()
print(f"{len(notes)} notes · {notes.utilisateur.nunique()} utilisateurs · {notes.film.nunique()} films")

res = evaluation.evaluer(notes, k=5)
res.round(4).to_csv("results/evaluation.csv", index=False)
print(res.round(3).to_string(index=False))


def pct(v):
    return f"{v * 100:.0f} %"


# --- Image de couverture -------------------------------------------------------------
d = res.sort_values("taux_reussite")
couleurs = [GRIS if "référence" in m else BLEU if m.startswith("Hybride") else BLEU_CLAIR for m in d.methode]
fig = plt.figure(figsize=(13.5, 5.4))
a1 = fig.add_axes([0.25, 0.13, 0.38, 0.66])
a1.barh(d.methode, d.taux_reussite, xerr=d.taux_reussite_et, color=couleurs, height=0.6,
        error_kw=dict(ecolor=INK2, lw=1, capsize=3))
for i, (v, e) in enumerate(zip(d.taux_reussite, d.taux_reussite_et)):
    a1.text(v + e + 0.02, i, pct(v), va="center", fontsize=10, color=INK)
a1.set_xlim(0, 0.85)
a1.set_xticks([])
a1.spines["bottom"].set_visible(False)
a1.grid(False)
a1.tick_params(axis="y", colors=INK, labelsize=10.5)
a1.set_title("Au moins un film aimé retrouvé dans le top 5", fontsize=11)

a2 = fig.add_axes([0.70, 0.13, 0.28, 0.66])
a2.barh(d.methode, d.couverture_catalogue, color=couleurs, height=0.6)
for i, v in enumerate(d.couverture_catalogue):
    a2.text(v + 0.02, i, pct(v), va="center", fontsize=10, color=INK)
a2.set_xlim(0, 1.2)
a2.set_xticks([])
a2.set_yticks([])
a2.spines["bottom"].set_visible(False)
a2.grid(False)
a2.set_title("Part du catalogue recommandée", fontsize=11)
pop = res.set_index("methode").loc["Popularité (référence)"]
hyb = res.set_index("methode").loc["Hybride (popularité + voisins)"]
vois = res.set_index("methode").loc["Collaboratif (utilisateurs voisins)"]
titre(fig, "Recommander des films : battre la popularité est plus dur qu'il n'y paraît",
      f"{notes.utilisateur.nunique()} utilisateurs, {notes.film.nunique()} films · 30 % des films aimés cachés, 5 tirages · "
      f"les voisins font presque aussi bien ({pct(vois.taux_reussite)} contre {pct(pop.taux_reussite)}) "
      f"en recommandant {pct(vois.couverture_catalogue)} du catalogue au lieu de {pct(pop.couverture_catalogue)}")
sauver(fig, "00_couverture.png")

# --- Précision et rappel ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))
for _, r in res.iterrows():
    c = GRIS if "référence" in r.methode else BLEU if r.methode.startswith("Hybride") else AQUA
    ax.scatter(r.rappel_at_k, r.precision_at_k, s=120, color=c, edgecolor="white", linewidth=1.5, zorder=3)
    ax.annotate(r.methode, (r.rappel_at_k, r.precision_at_k), xytext=(8, -3), textcoords="offset points",
                fontsize=9, color=INK)
ax.set_xlabel("Rappel@5 : part des films cachés retrouvés")
ax.set_ylabel("Précision@5 : part du top 5 réellement aimée")
ax.set_xlim(0.2, 0.6)
titre(fig, "Précision et rappel des 6 méthodes", "Moyenne sur 5 découpages aléatoires")
fig.subplots_adjust(top=1 - 1.1 / fig.get_figheight())
sauver(fig, "01_precision_rappel.png")

# --- Exemple de recommandations pour un profil -------------------------------------------
profil = {"Toy Story (1995)": 5.0, "Pinocchio (1940)": 4.5, "Dumbo (1941)": 4.0, "Heat (1995)": 1.5}
profil = {f: n for f, n in profil.items() if f in set(notes.film)}
exemples = {}
for classe in rc.TOUS:
    r = classe().ajuster(notes)
    exemples[r.nom] = list(r.recommander(profil, 5).index)
Path("results/exemple_recommandations.json").write_text(
    json.dumps({"profil": profil, "recommandations": exemples}, indent=2, ensure_ascii=False), encoding="utf-8")
for m, l in exemples.items():
    print(f"  {m}: {l}")

synthese = res.set_index("methode").round(4).to_dict(orient="index")
Path("results/synthese.json").write_text(json.dumps(synthese, indent=2, ensure_ascii=False), encoding="utf-8")
print("Terminé : figures dans docs/img, résultats dans results/")
