# Guide pyopencrg (français)

Ce guide s'adresse aux personnes qui travaillent avec des fichiers OpenCRG (essais, simulation,
pistes scannées) sans forcément bien connaître Python. Chaque exemple peut être copié tel quel.

> pyopencrg est une implémentation **non officielle**, sans lien avec l'ASAM.

## 1. Installation

Dans un terminal (invite de commandes Windows, Anaconda Prompt, terminal Linux...) :

```bash
pip install pyopencrg
```

numpy, scipy et matplotlib s'installent automatiquement. Pour graver un logo sur une piste
(`engrave_image`), ajoutez Pillow : `pip install "pyopencrg[image]"`.

Vérification :

```bash
pyopencrg --version
```

## 2. Les notions à connaître

Un fichier CRG décrit une route le long d'une **ligne de référence** (en général l'axe de la route).

| Grandeur | Signification |
|---|---|
| `u` | distance le long de la ligne de référence (m) |
| `v` | distance latérale à la ligne de référence (m), **positive vers la gauche** |
| `x, y, z` | position dans le repère "monde" |
| `road.z` | grille des hauteurs, tableau de `nu` lignes (sections transversales) × `nv` colonnes (sections longitudinales) |
| `road.zr` | altitude de la ligne de référence (une valeur par section) |
| `road.banking` | dévers (pente transversale, m/m) |
| `road.slope` | pente longitudinale (m/m) |
| `road.phi` | cap de la ligne de référence (rad) |

La hauteur absolue d'un point vaut :

```
z = road.z (grille) + road.zr (altitude de la ligne) + road.banking × v
```

`road.uv_to_z(u, v)` donne directement cette hauteur absolue.

Les **formats** de données :

| Format | Contenu | Usage |
|---|---|---|
| `KRBI` | binaire, simple précision | défaut : petit fichier, lu par tous les outils |
| `KDBI` | binaire, double précision | précision maximale |
| `LRFI` | texte, simple précision | lisible dans un éditeur |
| `LDFI` | texte, double précision | lisible et précis (gros fichier) |

## 3. Lire et regarder une piste

```python
import pyopencrg as crg

piste = crg.read("ma_piste.crg")
print(piste)              # résumé : longueur, largeur, pas, pente, dévers, NaN...
print(piste.check())      # rapport complet du contrôle de cohérence
piste.show()              # figures de synthèse
piste.view3d()            # vue 3D interactive rapide (nécessite pip install "pyopencrg[viewer]")
```

**Contrôle à l'ouverture** : chaque fichier est vérifié à la lecture (cela prend quelques
centièmes de seconde, même pour un gros fichier).

* Un fichier tronqué, illisible ou mal défini (par exemple une section entièrement vide) est
  **refusé** avec une erreur `CrgCheckError` qui explique le problème.
* Les problèmes mineurs (NaN au milieu d'une section, pas non multiple de 1 mm...) sont signalés
  une seule fois par un avertissement.
* Pour ouvrir quand même un fichier abîmé afin de le réparer : `crg.read("piste.crg", check=False)`.
* Pour refuser aussi les fichiers avec avertissements : `crg.read("piste.crg", check="strict")`.

**Très gros fichiers** (plusieurs Go, plus gros que la mémoire) : on ne charge que ce dont on a
besoin, le reste reste sur le disque.

```python
apercu = crg.read("scan.crg", step=4)          # 1 point sur 4 dans chaque direction : 16 fois moins de mémoire
detail = crg.read("scan.crg", u=(500, 520))    # 20 m en pleine résolution
mix = crg.read("scan.crg", u=(0, 300), step=(2, 1))   # 1 section sur 2, toutes les sections longitudinales
```

En ligne de commande : `pyopencrg show scan.crg --step 4`, `pyopencrg view scan.crg --u 500 520`.

Un fichier binaire qui a été ouvert puis réenregistré par un éditeur de texte (ses données sont
alors détruites) est reconnu et refusé immédiatement, avec une explication.

Vue 3D (`view3d`) : bouton gauche = rotation, molette = zoom, bouton du milieu = déplacement,
touche `r` = vue initiale. Options utiles : `color="grid"` (couleur = texture sans pente ni
dévers), `color="normal"` (fait ressortir les défauts de scan), `z_scale=5` (exagération verticale).

Quelques figures utiles une par une :

```python
piste.plot.heightmap()          # carte des hauteurs vue de dessus (zoomer affiche tous les points)
piste.plot.normalmap()          # carte des normales (repère les défauts de scan)
piste.plot.cross_sections()     # une dizaine de profils en travers
piste.plot.long_sections()      # une dizaine de profils en long
piste.plot.show_refline_map()   # tracé, cap, courbure
import matplotlib.pyplot as plt
plt.show()
```

## 4. Interroger la piste

```python
z = piste.uv_to_z(120.0, 0.9)          # hauteur à 120 m, 0,9 m à gauche de l'axe
x, y = piste.uv_to_xy(120.0, 0.9)      # position de ce point dans le monde
u, v = piste.xy_to_uv(x, y)            # et inversement

# profil en long sous la roue gauche (v = 0,8 m), tous les 1 cm
import numpy as np
u = np.arange(0, piste.length, 0.01)
z = piste.uv_to_z(u, 0.8)

# ou plus simplement, aux points de la grille :
u, z = piste.long_section(v=0.8)
v, z = piste.cross_section(u=50.0)
```

Toutes ces fonctions acceptent un nombre ou un tableau numpy (des millions de points d'un coup).

## 5. Modifier une piste

**Règle importante** : chaque fonction de modification renvoie une **nouvelle** piste ; l'originale ne
change pas. On écrit donc :

```python
morceau = piste.crop(u=(100, 200))         # garder de u = 100 m à u = 200 m
morceau = piste.crop(v=(-1.75, 1.75))      # garder une voie de 3,5 m
raccourcie = piste.trim(before=2, after=1) # couper 2 m au début, 1 m à la fin
sans_nan = piste.trim_nan_borders()        # couper les sections avec NaN aux extrémités
avec_rampes = piste.add_ramps(2, 2)        # rampes d'accès douces de 2 m
inversee = piste.flip()                    # sens de roulage inversé
fine = piste.resample(u_inc=0.005)         # nouveau pas de 5 mm
lissee = piste.filter("gauss", size=5)     # lissage
longue = crg.concatenate([piste1, piste2]) # mise bout à bout
```

On peut aussi modifier directement la grille :

```python
piste.z += 0.002                  # tout rehausser de 2 mm
piste.z[1000:1200, :] = 0         # aplanir une zone
```

Enregistrement :

```python
morceau.save("morceau.crg")               # binaire KRBI
morceau.save("morceau.crg", fmt="LDFI")   # texte double précision
```

## 6. Pente, dévers et modificateurs

```python
p = piste.extract_slope(0.01)            # sort la pente de la grille vers la ligne de référence
p = piste.extract_banking(0.01)          # idem pour le dévers
p = piste.separate_slope_banking(20, 20) # pente et dévers filtrés sur 20 m
p = piste.set_banking(0)                 # remet le dévers dans la grille (surface inchangée)
```

Les **modificateurs** (bloc `$ROAD_CRG_MODS`) sont appliqués par les simulateurs au chargement :

```python
print(piste.mods)                        # ex. {'refpoint_x': 0.0, 'grid_nan_mode': 2.0}
vue_simulateur = piste.apply_mods()      # ce que voit le simulateur
deplacee = piste.move_to(x=0, y=0, z=0, phi=0)   # placer le début de piste à l'origine
```

Si le fichier n'a pas de bloc de modificateurs, OpenCRG applique des valeurs par défaut :
NaN → dernière valeur valide, et début de piste ramené en (0, 0, 0) avec un cap nul.

## 7. Nettoyer une piste scannée

```python
import matplotlib.pyplot as plt

piste.plot.normal_distribution()   # histogramme pour choisir la tolérance
plt.show()

propre = (piste
          .remove_artifacts(normal_tol=0.3, radius=15)  # défauts très raides -> NaN
          .fill_holes("linear"))                        # rebouche les trous localement
propre.save("piste_propre.crg")
```

Version interactive (boutons « Corriger / Passer » pour chaque défaut) :

```python
from pyopencrg.cleaner import clean_interactive
propre = clean_interactive(piste, normal_tol=0.3, radius=15)
```

## 8. Créer des pistes

```python
from pyopencrg import generate

# route plate de 100 m x 4 m
r = generate.flat_road(100, 4, u_inc=0.01)

# route définie par une formule
r = generate.from_function(lambda u, v: 0.02 * np.sin(2 * np.pi * u / 10), 100, 3, 0.01)

# ligne droite, clothoïde puis virage de rayon 50 m, avec pente et dévers
r = generate.synthetic_road(
    length=350, width=4, u_inc=0.1,
    curvature=[(120, 0, 0), (50, 0, -1/50), (180, -1/50, -1/50)],   # (longueur, début, fin)
    slope=[(170, 0, 0), (180, 0, 0.03)],
    banking=[(120, 0, 0), (50, 0, 0.02)],
)

# piste d'obstacles trapézoïdaux
u, z = generate.trapezoid_bumps(u_inc=0.005, length=20, bump_length=0.23, gap=0.30,
                                height=0.03, attack_angle=45, lead_in=10)
obstacles = generate.from_longitudinal_profile(u, z, width=2.5, v_inc=0.005, slope=-0.10)

# route qui suit des points GPS
r = generate.road_from_points(x_gps, y_gps, width=7, u_inc=0.1)
```

Ajouts sur une piste existante :

```python
# marquage de 20 cm de large, 3 mm d'épaisseur, à v = 1,75 m, sur toute la longueur
r = r.add_profile([r.u_start, r.u_end], [1, 1], [1.64, 1.65, 1.85, 1.86], [0, 0.003, 0.003, 0])
# rugosité aléatoire de 0 à 5 mm
r = r.add_random([r.u_start, r.u_end], [1, 1], [r.v_min, r.v_max], [1, 1], seed=1, amp_max=0.005)
# logo gravé (pixels blancs +1 cm)
r = r.engrave_image("logo.png", height=0.01, width=1.5, u0=20, v0=-0.75)
```

## 9. Convertir un STL en CRG

```python
from pyopencrg import importers

r = importers.from_stl("scan.stl", u_inc=0.005)                     # piste droite
r = importers.from_stl("scan.stl", u_inc=0.01, reference_line=(xc, yc))  # piste courbe : donner l'axe
r.save("scan.crg")
```

## 10. Exporter

```python
r.to_obj("piste.obj", step=2)          # maillage 3D
r.to_stl("piste.stl")                  # STL binaire
r.to_csv("piste.csv", step=10)         # tableau u, v, x, y, z pour Excel
r.to_trm("trajet.trm", step=1.0)       # trajectoire Simpack (+ fichier d'offset)
r.save_heightmap("hauteurs.png")       # image
r.save_combined_map("carte.png", n_slices=4)
r.to_html_map("carte.html")            # carte OpenStreetMap (si la piste a des données WGS84)
```

## 11. En ligne de commande (sans écrire de Python)

```bash
pyopencrg info  piste.crg
pyopencrg check piste.crg
pyopencrg show  piste.crg
pyopencrg view  piste.crg       # vue 3D
pyopencrg convert piste.crg piste_texte.crg --fmt LDFI
pyopencrg export piste.crg piste.obj --step 2
```

## 12. Obtenir de l'aide

Chaque fonction est documentée :

```python
help(crg.CrgRoad)       # toutes les méthodes
help(piste.crop)        # une méthode
```
