# PY-PT (Python Propulsion Toolbox)

Regroupe des outils Python qui servent de base à la propulsion. Les dossiers sont:

* CEA : Un wrapper pour NASA CEA, qui permet de simuler la combustion et obtenir les performances théoriques du moteur.
* DATA: Automatisation de l'analyse de données de tests de moteurs fusés hybrides.
* MOT : Simulateur de moteur fusée hybride. Génère des graphiques de performance selon les paramètres du moteur.
* NOX : Donne les propriétés de l'oxide nitreux et simule l'injection biphasique avec restriction de section.
* NOZ : Design de nozzle axisymmétrique par la méthode de Rao (approximation du contour à poussée maximale).

## global.py

Fichier qui comprend des fonctions utiles pour tous les sous-dossiers, comme la "loading bar". Pour l'inclure ajoutez:
```python
import sys
sys.path.insert(1, 'PY-PT_path')
from global import *
```
En haut de votre code pour pouvoir utiliser les fonctionnalités (remplacer PY-PT_path par le chemin menant à ce dossier).
