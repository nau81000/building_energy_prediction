# Prédiction de la consommation des bâtiments

Ce projet cherche à prédire la consommation des bâtiments en utilisant du machine learning

## Prérequis

- python3
- bentoml

## Structure principale du projet

```
├── bentoml.yaml             # Fichier de configuration de Bentoml
├── notebook.ipynb           # Notebook python contenant l'analyse exploratoire
├── results.pdf              # Résultats d'analyse
├── service.py               # Script lançant le service de prédiction
```

## Installation

1. **Cloner le dépôt**

```bash
git clone https://github.com/nau81000/building_energy_prediction.git
cd building_energy_prediction
```

2. **Installer poetry**

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. **Créer l'environnement (installer les dépendances)**

```bash
poetry install --no-root
```

4. **Utiliser le notebook python dans un navigateur pour l'analyse exploratoire**

```bash
jupyter lab notebook.ipynb
```

5. **Construction du service**

- Construction API et génération d'un id: ```bentoml build -f bentoml.yaml```
- Construction image Docker: ```bentoml containerize --opt platform=linux/amd64 projet6-ml-service:<id_bentoml>```
- Test image Docker: ```docker run --rm -p 8080:8080 projet6-ml-service::<id_bentoml>```

6. **Exemple de déploiement sur BentoCloud**

- ```bentoml deploy projet6-ml-service:<id_bentoml> -n ${DEPLOYMENT_NAME}```
