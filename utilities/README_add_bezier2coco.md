# `add_bezier2coco.py` — polygones COCO vers points de Bézier

Ce script prépare les JSON COCO des tampons Forbin pour l'entraînement SA-DETR (base DeepSolo). Il convertit chaque polygone de segmentation en 2 courbes de Bézier cubiques (`bezier_pts`) et encode le texte du tampon pour la CTC (`rec`).

## Utilisation

Depuis la racine du projet :

```bash
python utilities/add_bezier2coco.py \
    datasets/forbin_dataset/test_fold_0_single.json \
    output/bezier2coco_check/test_fold_0_v4.json
```

- Le second chemin est le fichier de sortie. Choisissez toujours un chemin différent de l'entrée.
- `--bezier-version v3|v4` choisit la conversion (v4 par défaut ; v3 est l'ancienne, gardée pour comparaison).
- `--keep-duplicates` conserve les annotations en double ; par défaut, elles sont supprimées.
- `--map` (par défaut `char_map/char2idx/Latin.json`) donne la table caractère → index ; `--no-indent` produit un JSON compact.

Pour comparer v3 et v4 sur un JSON et dessiner des cas précis :

```bash
python utilities/compare_bezier_versions.py datasets/forbin_dataset/test_fold_0_single.json \
    --out-dir output/bezier2coco_check --ids 733 488 53 24 713 719
```

Le script écrit un tableau par catégorie de polygone (`compare_v3_v4_<json>.md`) et une planche d'images (`compare_v3_v4_<json>.png`). Chaque cas y occupe deux panneaux (v3 puis v4), sur fond de l'image réelle du tampon, pour comparer la direction des courbes à celle du texte :

- **bleu** : segmentation ;
- **orange** : courbe du haut, fléchée de P0 à P3, départ marqué « H » ;
- **violet** : courbe du bas, fléchée de P0 à P3, départ marqué « B » ;
- **pointillé gris** : ligne médiane (la polyline que le modèle apprend), dans son sens de parcours.

`--image-root` indique le dossier des images (par défaut, celui du JSON) ; `--image-root none` désactive le fond.

## Champs ajoutés à chaque annotation

| Champ | Contenu | Utilisé par |
|---|---|---|
| `bezier_pts` | 8 points `[x, y]` : côté haut P0→P3 de gauche à droite, puis côté bas de droite à gauche, qui repart de l'extrémité où finit le haut | `load_text_json` ([text.py](../adet/data/datasets/text.py)) pour calculer `polyline`, `boundary` et `beziers` ; l'évaluateur ([text_evaluation.py](../adet/evaluation/text_evaluation.py)), qui échantillonne la courbe pour obtenir le polygone de vérité terrain |
| `rec` | 25 indices de caractères, complétés par 0 ; `"###"` si le tampon n'a pas de texte | Loss CTC. `load_text_json` écarte toute instance dont le `rec` ne contient que des 0 |
| `text` | Texte brut, ou `"###"` si `texts` est absent, vide ou blanc | Informatif |
| `language` | Toujours `"Latin"` | Tête de langue |

La `segmentation` est aussi réécrite, sans ses points dupliqués.

## Conversion v4 (`robust_polygon_to_bezier_v4`, [curves.py](curves.py))

1. **Repère** : `cv2.minAreaRect` donne le rectangle minimal. Le grand axe sert d'axe « gauche → droite », orienté selon la règle de direction ci-dessous. Le polygone est parcouru dans le sens horaire à l'écran.
2. **Coins candidats.** Plusieurs jeux de 4 coins (haut-gauche, haut-droit, bas-droit, bas-gauche) sont essayés :
   - (0) si le polygone simplifié a exactement 4 sommets, ces 4 sommets, dans la rotation dont le côté haut va le plus vers la droite (cas des parallélogrammes très penchés) ;
   - (a) les sommets du polygone **simplifié** (Douglas-Peucker, tolérance de 3 % du petit côté) les plus proches des coins du rectangle. Un point quasi aligné sur un côté ne peut donc pas devenir un coin ;
   - (b) les extrémités du grand axe (formes ovales, sans petit côté droit) ;
   - (c) les points du contour les plus proches des coins du rectangle (coins arrondis).

   Pour un tampon presque carré (petit côté / grand côté > 0,85), où le grand axe est arbitraire, les mêmes jeux sont aussi essayés sur le petit axe ; ce choix n'est retenu que s'il gagne au moins 0,01 d'IoU. Sinon, le grand axe est imposé.
3. **Coupe haut/bas** : le côté haut est la portion du contour qui va du coin haut-gauche au coin haut-droit **en suivant l'ordre des sommets** ; le côté bas va du coin bas-droit au coin bas-gauche. Les sommets des petits côtés ne sont attribués ni au haut ni au bas.
4. **Ajustement** : chaque côté est densifié (environ 150 points par longueur de grand côté). Une cubique est ajustée par moindres carrés, extrémités fixées, en 3 itérations de recalage des paramètres. La courbe suit les arêtes du polygone, pas seulement ses sommets, et ne peut donc pas partir loin.
5. **Choix** : les contours qui se croisent sont rejetés. On garde le candidat de meilleure IoU avec le polygone. Si aucun candidat n'est valide, on se replie sur le rectangle minimal à côtés droits.

## Résultats sur le fold de test 0 (910 tampons)

IoU entre le contour des Béziers (50 points par côté) et la segmentation :

| Catégorie | n | v3 IoU moy | v3 IoU min | v3 IoU<0,9 | v3 se croise | **v4 IoU moy** | **v4 IoU min** | **v4 IoU<0,9** | **v4 se croise** |
|---|---|---|---|---|---|---|---|---|---|
| Quadrilatère | 832 | 1,000 | 0,823 | 1 | 1 | 1,000 | 1,000 | 0 | 0 |
| Plus de 4 points, convexe | 75 | 0,924 | 0,557 | 17 | 4 | 0,982 | 0,918 | 0 | 0 |
| Non convexe | 3 | 0,556 | 0,325 | 3 | 2 | 0,871 | 0,787 | 2 | 0 |
| **Tous** | 910 | 0,992 | 0,325 | 21 | 7 | **0,998** | **0,787** | **2** | **0** |

Mêmes conclusions sur le train et le val du fold 0 :

| Split | n | v3 IoU moy | v3 IoU min | v3 IoU<0,9 | v3 se croise | **v4 IoU moy** | **v4 IoU min** | **v4 IoU<0,9** | **v4 se croise** |
|---|---|---|---|---|---|---|---|---|---|
| train | 2 895 | 0,993 | 0,360 | 66 | 19 | **0,999** | **0,812** | **5** | **0** |
| val | 704 | 0,992 | 0,500 | 14 | 3 | **0,998** | **0,856** | **1** | **0** |

Dans les trois splits, tous les quadrilatères restent exacts (IoU minimale de 1,000), et tous les cas encore sous 0,9 sont des polygones non convexes.

Cas signalés :

| id | 733 | 488 | 53 | 24 | 713 | 719 | 804 | 839 | 805 | 307 | 246 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 0,85 | 0,87 | 0,87 | 0,87 | 0,91 | 0,80 | 0,56 | 0,64 | 0,66 | 0,68 | 0,82 |
| v4 | 0,96 | 0,94 | 0,94 | 0,98 | 1,00 | 0,99 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |

Il reste 2 cas sous 0,9, des polygones qu'aucune paire de Béziers ne peut bien représenter : 802 (forme en « L », 0,79) et 761 (32 points ondulés, 0,86). Ce sont peut-être des erreurs d'annotation.

## Direction des courbes

En v3, le sens des courbes dépendait de l'ordre des coins renvoyé par `cv2.boxPoints` : le côté haut allait de gauche à droite pour seulement 523 tampons sur 910, et il se trouvait parfois en bas du tampon (id 24).

En v4, la direction suit une **règle géométrique fixe**, dans le repère du grand axe :

- le côté haut va de gauche à droite et le côté bas revient de droite à gauche ;
- l'axe est orienté pour que son angle (repère image, 0° = horizontal, 90° = vertical vers le bas) soit dans [−125°, 55°[ (`DIRECTION_CUT_DEG = 55` dans `curves.py`). Un tampon horizontal va donc de gauche à droite, et un tampon vertical de bas en haut, comme un tampon horizontal tourné d'un quart de tour vers la gauche.

La bascule est placée à 55° parce que c'est l'orientation la moins fréquente sur Forbin : 29 tampons sur 4 509 à ±5°. Une bascule à la verticale (90°) aurait concerné 221 tampons, dont le sens aurait changé pour une inclinaison d'un seul degré.

**Limite : cette règle ne connaît pas le sens de lecture du texte.** Les annotations ne donnent que le contour, et un contour ne dit pas où est le haut du texte. Exemples sur la planche du test :

- 719 (« V. FORBIN ») et 53 sont à l'envers : la polyline va dans le sens inverse de la lecture ;
- 505 a un texte vertical sur un tampon allongé horizontalement ;
- 321 (« DAILY MIRROR », vertical) est correct, mais seulement parce que son texte se lit de bas en haut.

Pour la **détection seule** (`POINT_TEXT_WEIGHT: 0`), une règle géométrique fixe suffit : le modèle doit seulement apprendre à reproduire un ordre de points déterministe, calculable depuis la forme du tampon. Le sens de lecture devient nécessaire si l'on veut **reconnaître le texte** ; il faudra alors l'annoter (par exemple, une rotation de 0°, 90°, 180° ou 270° par tampon) ou le déduire par OCR.

## Défauts de v3 (corrigés en v4)

Le JSON actuel a été produit avec v3. Relancer v3 sur `test_fold_0_single.json` redonne exactement les mêmes `bezier_pts`.

### 1. Rectangle avec un point en trop sur un côté (cause principale)

Exemple : l'annotation 804, segmentation `[(1528,513), (2024,513), (2028,356), (1535,356), (1532,441)]`. Le 5ᵉ point est au milieu du petit côté gauche.

1. v3 range chaque sommet en « haut » ou « bas » selon la ligne médiane du rectangle minimal. Ce point, presque sur la ligne, tombe en « bas » : ce côté a alors 3 points.
2. Il est loin du grand côté : le ratio de courbure du bas passe de 0 à 0,266, au-dessus du seuil de 0,20, et le côté est classé « courbe ».
3. L'ajustement libre doit passer par ce point. Il envoie un point de contrôle à (1495, 270), hors du tampon, et le contour se croise.

Cas similaires : 805, 839, 307, 719, 24, 782, 713, 697.

### 2. Quadrilatère penché mal coupé

Exemple : l'annotation 246, un parallélogramme fin. La ligne médiane laisse 3 sommets d'un côté et 1 de l'autre ; le côté à 1 point dégénère.

### 3. Formes courbes peu détaillées

Pour un hexagone ou un heptagone qui approxime un ovale (733, 488, 53), l'ajustement passe exactement par 1 ou 2 sommets, sans contrainte, et déborde de la forme.

## Autres corrections

- **Tampons sans texte.** Un `texts` vide était encodé en 25 zéros, et `load_text_json` écartait l'instance. Sur le fold 0, cela touchait 777 tampons sur 2 895 dans le train (27 %), et 528 images du train perdaient tous leurs tampons. Ces tampons sont maintenant encodés `"###"`. Sur le test, `load_text_json` charge désormais 908 tampons sur 908.
- **Doublons** : les annotations identiques (même image, même polygone) sont supprimées, soit 2 dans le test du fold 0 (814 et 848).
- **Erreurs** : le nombre d'annotations restées sans `bezier_pts` est signalé en fin d'exécution.
- **Import** : le script se lance sans réglage de `PYTHONPATH`.
- **Évaluateur** ([text_evaluation.py](../adet/evaluation/text_evaluation.py)) : le polygone de vérité terrain est maintenant le contour échantillonné des Béziers (`bezier_pts_to_boundary`, même calcul que `text.py`), et non plus les 8 points de contrôle. Une annotation sans `bezier_pts` est ignorée avec un avertissement ; avant, elle réutilisait le polygone de l'annotation précédente.

## Points non traités

- **Prédictions dans l'évaluateur** : elles sont réduites à un rectangle orienté (`polygon2rbox`), et celles dont le texte décodé est vide sont écartées (`if s == "": continue`). Avec `POINT_TEXT_WEIGHT: 0`, la tête de reconnaissance n'est pas entraînée ; cette règle peut donc supprimer des détections correctes.
- **Code mort dans `curves.py`** : v1, v2, `get_bezier_line`, `classify_shape`, et `compute_curvature_ratio` définie deux fois.
