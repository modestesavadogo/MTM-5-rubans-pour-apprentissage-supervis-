# Machine de Turing Multi-Rubans — Apprentissage Supervisé

Simulateur d'une **Machine de Turing Multi-Rubans (MTM) à cinq rubans** exécutant la **descente de gradient par lots** pour la régression linéaire univariée, en **arithmétique rationnelle exacte** (`Fraction`, sans aucun arrondi flottant).

Projet réalisé dans le cadre du cours d'Informatique Théorique — **INPT, Filière Ingénierie des Données & Intelligence Artificielle**, encadré par Pr. HANIN Charifa.

> Question de recherche : dans quelle mesure un algorithme d'apprentissage supervisé peut-il être formellement simulé par une MTM ? Quelles en sont les implications en termes de calculabilité et de complexité ?

## Architecture des rubans

| Ruban | Rôle | Contenu |
|-------|------|---------|
| `R0` | Dataset | paires $(x_i, y_i)$ encodées en fractions |
| `R1` | Paramètres | $w$ (poids) et $b$ (biais) |
| `R2` | Calculs | accumulateurs $\hat{y}$, erreur, gradients, perte |
| `R3` | Compteur | itération courante $t$ |
| `R4` | Sortie | modèle appris $f(x) = wx + b$ |

États : `INIT → LOAD → FORWARD → LOSS → GRAD → UPDATE → CHECK → (LOAD | HALT)`

## Contenu du dépôt

```
.
├── notebook/
│   └── turing_ml.ipynb        # simulateur exécutable (Python, sans dépendance externe)
├── report/
│   ├── turing_ml_report.tex   # rapport académique complet (source LaTeX, FR)
│   ├── turing_ml_report.pdf   # rapport compilé
│   ├── INPT_Logo.png
│   └── logo_anrt.png
├── LICENSE
└── .gitignore
```

## Pourquoi ce projet

La plupart des implémentations de descente de gradient s'arrêtent à "ça marche en Python". Ici, l'objectif est différent : montrer que l'algorithme est **formellement Turing-calculable**, en le faisant tourner à l'intérieur d'une vraie machine de Turing multi-rubans — états, fonction de transition $\delta$, têtes de lecture/écriture — plutôt que comme une simple boucle `for`.

Points clés :
- **Arithmétique exacte** : tous les symboles de ruban sont des `Fraction` Python, donc aucune perte de précision liée aux flottants.
- **Transparence totale** : chaque transition $\delta$ est une méthode dédiée (`_transition_INIT`, `_transition_FORWARD`, etc.), avec trace complète disponible.
- **Zéro dépendance** : bibliothèque standard uniquement (`fractions`, `dataclasses`, `typing`).
- **Analyse de complexité formelle** : $\mathcal{O}(T \cdot n \cdot k)$ transitions, réduction à une MTM mono-ruban, espace polynomial — tout est démontré dans le rapport.

## Lancer le notebook

Prérequis : Python 3.10+ (aucune autre dépendance).

```bash
pip install notebook
jupyter notebook notebook/turing_ml.ipynb
```

Ou directement en ligne de commande :

```bash
jupyter nbconvert --to notebook --execute notebook/turing_ml.ipynb --output executed.ipynb
```

## Aperçu d'exécution

```
======================================================================
  MTM 5-Rubans — Apprentissage Supervisé (Régression Linéaire)
  n=10 exemples | η=0.0100 | T=500 itérations
======================================================================
  δ(INIT     → LOAD    ) | step=   0 | Initialisation w=0, b=0, t=0
  δ(CHECK    → LOAD    ) | step= 600 | t=100 < T=500 → nouvelle époque
  ...
  δ(CHECK    → HALT    ) | step=3000 | t=500 ≥ T=500 → q_acc ✓ | Modèle : w=2.019862, b=0.903240
======================================================================
  HALT atteint en 3001 transitions.
  Modèle final (R4) : w = 2.019862, b = 0.903240
======================================================================

  Valeurs théoriques : w=2.0000, b=1.0000
  Valeurs apprises   : w=2.019862, b=0.903240
  MSE=0.017079 | R²=0.999487
```

Le notebook contient trois démonstrations :
1. **Démo 1** — régression $y = 2x + 1$ sur 10 points bruités → $R^2 = 0{,}9995$
2. **Démo 2** — régression $y = -0{,}5x + 10$ (pente négative) sur 8 points → $R^2 = 0{,}9507$
3. **Démo 3** — trace formelle complète, transition par transition, sur 3 exemples / 3 itérations

## Rapport

Le rapport complet (`report/turing_ml_report.pdf`) couvre :
1. Fondements théoriques (MT simple ruban, MTM, thèse de Church-Turing)
2. L'algorithme cible (régression linéaire, descente de gradient par lots)
3. Conception formelle de la MTM à 5 rubans ($Q$, $\Sigma$, $\Gamma$, $\delta_5$)
4. Implémentation Python et résultats expérimentaux
5. Analyse de complexité (transitions et espace)
6. Conclusion et perspectives (extension à la classification, réseaux de neurones)

Pour recompiler le PDF depuis la source :

```bash
cd report
pdflatex turing_ml_report.tex
pdflatex turing_ml_report.tex   # deuxième passe pour la table des matières
```

## Licence

MIT — voir [LICENSE](LICENSE).

