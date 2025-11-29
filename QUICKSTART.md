# Guide de Démarrage Rapide - IDS ML

**Auteurs**: Rana Romdhane & Oulimata Sall

Ce guide vous permet de démarrer rapidement avec le système IDS ML.

## 🚀 Démarrage en 5 Minutes

### Option 1: Avec Docker (Recommandé)

```bash
# 1. Cloner le projet
git clone https://github.com/RanaRomdhane/intelligent-ids.git
cd intelligent-ids

# 2. Lancer avec Docker Compose
docker-compose up -d

# 3. Accéder à l'application
# Ouvrir http://localhost:5000 dans votre navigateur
```

### Option 2: Installation Locale

```bash
# 1. Cloner le projet
git clone https://github.com/RanaRomdhane/intelligent-ids.git
cd intelligent-ids

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Créer des données d'exemple et entraîner les modèles
python example_usage.py
python scripts/train.py

# 5. Lancer l'application
python app.py

# 6. Accéder à l'application
# Ouvrir http://localhost:5000 dans votre navigateur
```

## 📊 Utiliser vos Propres Données

### 1. Préparer votre Dataset

Placez votre fichier CSV dans `data/raw/` :

```bash
cp votre_dataset.csv data/raw/
```

Format attendu :
```csv
duration,protocol_type,service,src_bytes,dst_bytes,count,label
0,tcp,http,1000,2000,5,normal
1,udp,ftp,500,1500,3,dos
```

### 2. Entraîner les Modèles

```bash
python scripts/train.py --data data/raw/votre_dataset.csv
```

Cela va :
- Nettoyer et préparer les données
- Extraire les features
- Entraîner Random Forest, SVM et Neural Network
- Sauvegarder les modèles dans `models/`
- Générer un rapport de comparaison

### 3. Faire des Prédictions

```bash
python scripts/predict.py \
  --input data/raw/nouveau_trafic.csv \
  --model random_forest \
  --output predictions.csv
```

## 🌐 Utiliser l'Interface Web

### 1. Lancer l'Application

```bash
python app.py
```

### 2. Accéder aux Pages

- **Accueil**: http://localhost:5000/
- **Démo**: http://localhost:5000/demo.html
- **Documentation**: http://localhost:5000/documentation.html

### 3. Utiliser la Démo

1. Cliquer sur "Démo" dans le menu
2. Uploader un fichier CSV
3. Sélectionner un modèle (Random Forest recommandé)
4. Cliquer sur "Analyser"
5. Voir les résultats en temps réel

## 📓 Utiliser les Notebooks Jupyter

```bash
# 1. Lancer Jupyter
jupyter notebook

# 2. Ouvrir les notebooks dans l'ordre :
# - notebooks/data_exploration.ipynb
# - notebooks/model_training.ipynb
# - notebooks/evaluation.ipynb
```

## 🐳 Commandes Docker Utiles

```bash
# Construire l'image
docker-compose build

# Démarrer les services
docker-compose up -d

# Voir les logs
docker-compose logs -f ids-ml

# Arrêter les services
docker-compose down

# Redémarrer
docker-compose restart

# Accéder au conteneur
docker-compose exec ids-ml bash
```

## 🔧 Commandes Utiles

### Entraînement Personnalisé

```bash
# Avec paramètres personnalisés
python scripts/train.py \
  --data data/raw/dataset.csv \
  --test-size 0.3 \
  --random-state 42
```

### Prédiction avec Différents Modèles

```bash
# Random Forest
python scripts/predict.py --input test.csv --model random_forest

# SVM
python scripts/predict.py --input test.csv --model svm

# Neural Network
python scripts/predict.py --input test.csv --model neural_network
```

### API REST

```bash
# Lister les modèles disponibles
curl http://localhost:5000/api/models

# Obtenir les statistiques
curl http://localhost:5000/api/stats

# Faire une prédiction (avec fichier)
curl -X POST http://localhost:5000/api/predict \
  -F "file=@test.csv" \
  -F "model=random_forest"
```

## 📦 Datasets Recommandés

### NSL-KDD
```bash
wget http://example.com/nsl-kdd.csv -O data/raw/nsl-kdd.csv
python scripts/train.py --data data/raw/nsl-kdd.csv
```

### UNSW-NB15
```bash
wget http://example.com/unsw-nb15.csv -O data/raw/unsw-nb15.csv
python scripts/train.py --data data/raw/unsw-nb15.csv
```

### CICIDS2017
```bash
wget http://example.com/cicids2017.csv -O data/raw/cicids2017.csv
python scripts/train.py --data data/raw/cicids2017.csv
```

## 🐛 Dépannage

### Erreur d'Import
```bash
# Réinstaller les dépendances
pip install --force-reinstall -r requirements.txt
```

### Erreur TensorFlow
```bash
# Utiliser la version CPU
pip install tensorflow-cpu==2.16.2
```

### Erreur de Mémoire
```bash
# Réduire la taille du dataset ou utiliser un échantillon
python scripts/train.py --data data/raw/dataset.csv --sample 10000
```

### Port Déjà Utilisé
```bash
# Changer le port dans app.py
# ou arrêter le processus utilisant le port 5000
lsof -ti:5000 | xargs kill -9  # Mac/Linux
# ou
netstat -ano | findstr :5000  # Windows
```

## 📚 Documentation Complète

Pour plus d'informations, consultez :
- [README.md](README.md) - Documentation complète
- [Documentation Web](http://localhost:5000/documentation.html) - Une fois l'app lancée
- [Notebooks Jupyter](notebooks/) - Exemples détaillés

## 🆘 Support

Pour toute question ou problème :
1. Vérifier la [documentation](README.md)
2. Consulter les [issues GitHub](https://github.com/RanaRomdhane/intelligent-ids/issues)
3. Contacter les auteurs

---

**Bon démarrage avec IDS ML ! 🚀**

Rana Romdhane & Oulimata Sall