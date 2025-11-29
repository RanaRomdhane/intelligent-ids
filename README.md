# 🛡️ IDS ML - Système de Détection d'Intrusions Intelligent

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Academic-purple.svg)](LICENSE)

Système intelligent de détection d'intrusions réseau basé sur le Machine Learning, développé dans le cadre d'un projet académique en cybersécurité.

**Développé par :** Rana Romdhane & Oulimata Sall  
**Année :** 2025  
**Objectif :** Projet Académique - Cybersécurité

---

## 📋 Table des Matières

- [Caractéristiques](#-caractéristiques)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Utilisation](#-utilisation)
- [API Documentation](#-api-documentation)
- [Modèles ML](#-modèles-ml)
- [Intégration ELK](#-intégration-elk)
- [Surveillance Temps Réel](#-surveillance-temps-réel)
- [Structure du Projet](#-structure-du-projet)
- [Contribuer](#-contribuer)
- [License](#-license)

---

## ✨ Caractéristiques

### 🎯 Fonctionnalités Principales

- **Détection Multi-Modèles** : Random Forest, SVM, et Réseaux de Neurones
- **Analyse Temps Réel** : Monitoring continu du trafic réseau
- **Système d'Alertes** : Notifications automatiques avec niveaux de sévérité
- **Intégration SIEM** : Support ELK Stack (Elasticsearch, Logstash, Kibana)
- **Dashboard Interactif** : Visualisation en temps réel avec WebSockets
- **API REST** : Endpoints pour intégration externe
- **Métriques Détaillées** : Accuracy, Precision, Recall, F1-Score, ROC/AUC

### 🎯 Types d'Attaques Détectées

- DoS/DDoS (Denial of Service)
- Port Scan & Reconnaissance (Probe)
- Brute Force
- SQL Injection
- Remote to Local (R2L)
- User to Root (U2R)
- Botnet Activity
- Data Exfiltration

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Interface Web                        │
│  (Dashboard Temps Réel | Démo | Documentation)          │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              API Flask + WebSocket                       │
│  (REST Endpoints | Real-time Communication)             │
└──────────┬────────────────┬────────────┬────────────────┘
           │                │            │
     ┌─────▼─────┐    ┌────▼────┐  ┌───▼────┐
     │  ML       │    │ Alert   │  │  ELK   │
     │  Models   │    │ Manager │  │  Stack │
     └───────────┘    └─────────┘  └────────┘
           │
     ┌─────▼──────────────────┐
     │  Data Preprocessing     │
     │  (Feature Engineering)  │
     └────────────────────────┘
```

---

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- (Optionnel) ELK Stack pour intégration SIEM
- (Optionnel) Docker pour conteneurisation

### Installation Rapide

```bash
# 1. Cloner le repository
git clone https://github.com/RanaRomdhane/intelligent-ids.git
cd intelligent-ids

# 2. Créer un environnement virtuel
python -m venv venv

# 3. Activer l'environnement virtuel
# Sur Linux/Mac:
source venv/bin/activate
# Sur Windows:
venv\Scripts\activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Créer les répertoires nécessaires
python -c "from config import Config; Config.init_directories()"

# 6. Entraîner les modèles (première utilisation)
python scripts/train.py

# 7. Démarrer le système
python start.py
```

### Installation avec Docker (Recommandé pour Production)

```bash
# Build l'image Docker
docker build -t ids-ml:latest .

# Lancer le conteneur
docker run -p 5000:5000 -v $(pwd)/data:/app/data ids-ml:latest
```

---

## ⚙️ Configuration

### Configuration de Base

Créer un fichier `.env` à la racine du projet :

```env
# Application
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
DEBUG=True
PORT=5000

# Alert Manager
EMAIL_ENABLED=false
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_FROM=ids@example.com
SMTP_TO=admin@example.com
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# ELK Stack
ELASTICSEARCH_HOSTS=localhost:9200
ELASTICSEARCH_USERNAME=elastic
ELASTICSEARCH_PASSWORD=your-password
SIEM_ENABLED=false

# Monitoring
NETWORK_INTERFACE=eth0
LOG_LEVEL=INFO
```

### Configuration des Modèles

Les paramètres des modèles ML peuvent être ajustés dans `config.py` :

```python
ML_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None
    },
    # ... autres paramètres
}
```

---

## 💻 Utilisation

### 1. Démarrage Rapide

```bash
# Démarrage avec script automatique
python start.py

# Ou démarrage manuel
python app.py
```

Le système sera accessible à :
- **Interface Web** : http://localhost:5000
- **Dashboard** : http://localhost:5000/dashboard
- **Démo** : http://localhost:5000/demo.html

### 2. Entraînement des Modèles

```bash
# Entraîner tous les modèles
python scripts/train.py

# Entraîner un modèle spécifique
python scripts/train.py --model random_forest

# Avec dataset personnalisé
python scripts/train.py --data data/raw/your_dataset.csv
```

### 3. Prédictions sur Nouvelles Données

```bash
# Via ligne de commande
python scripts/predict.py --input data/raw/test_data.csv --model random_forest

# Via l'interface web (méthode recommandée)
# Accéder à http://localhost:5000/demo.html
```

### 4. Surveillance Temps Réel

```python
# Démarrer via API
curl -X POST http://localhost:5000/api/monitoring/start

# Ou via le dashboard web
# Accéder à http://localhost:5000/dashboard
```

---

## 📚 API Documentation

### Endpoints Principaux

#### Prédiction

```http
POST /api/predict
Content-Type: multipart/form-data

Parameters:
- file: CSV file containing network traffic data
- model: Model name (random_forest, svm, neural_network)

Response:
{
  "success": true,
  "total_samples": 1000,
  "predictions": {"normal": 800, "dos": 150, "probe": 50},
  "accuracy": 95.5,
  "alerts_count": 200
}
```

#### Statistiques Système

```http
GET /api/stats

Response:
{
  "models_loaded": 3,
  "available_models": ["random_forest", "svm", "neural_network"],
  "preprocessor_loaded": true,
  "alert_manager_active": true,
  "elk_connected": true,
  "monitoring_active": false
}
```

#### Gestion du Monitoring

```http
# Démarrer
POST /api/monitoring/start

# Arrêter
POST /api/monitoring/stop

# Obtenir les stats
GET /api/monitoring/stats
```

#### Gestion des Alertes

```http
# Lister les alertes
GET /api/alerts?limit=50&severity=high

# Mettre à jour le statut
PUT /api/alerts/{alert_id}/status
{
  "status": "acknowledged"
}
```

### WebSocket Events

```javascript
// Connexion
socket = io('http://localhost:5000');

// Recevoir les mises à jour de stats
socket.on('stats_update', function(data) {
    console.log('Stats:', data);
});

// Recevoir les nouvelles alertes
socket.on('new_alert', function(alert) {
    console.log('Alert:', alert);
});
```

---

## 🤖 Modèles ML

### Random Forest

**Caractéristiques** :
- Ensemble de 100 arbres de décision
- Excellente performance sur données déséquilibrées
- Résistant au surapprentissage

**Utilisation** :
```python
from src.models import RandomForestIDS

model = RandomForestIDS(n_estimators=100, random_state=42)
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

### SVM (Support Vector Machine)

**Caractéristiques** :
- Kernel RBF pour classification non-linéaire
- Bon pour données haute dimension
- Nécessite normalisation des features

### Neural Network

**Architecture** :
- Couches cachées : [128, 64]
- Dropout : 0.3
- Activation : ReLU
- Optimiseur : Adam

---

## 📊 Intégration ELK

### Installation d'Elasticsearch

```bash
# Avec Docker
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.0

# Vérifier la connexion
curl http://localhost:9200
```

### Configuration dans IDS ML

```python
# Le système se connecte automatiquement si ELK est disponible
# Configuration dans config.py ou .env
ELASTICSEARCH_HOSTS=localhost:9200
SIEM_ENABLED=true
```

### Visualisation dans Kibana

1. Accéder à Kibana : http://localhost:5601
2. Créer un index pattern : `ids-*`
3. Les dashboards sont automatiquement peuplés

---

## 📡 Surveillance Temps Réel

### Architecture

```
Capture Réseau → Extraction Features → ML Model → Alert Manager → Dashboard
     (Scapy)         (Pipeline)        (Predict)     (Notify)      (WebSocket)
```

### Configuration

```python
# Dans config.py
MONITORING_INTERFACE = 'eth0'  # Interface réseau à surveiller
MONITORING_UPDATE_INTERVAL = 2  # Secondes entre mises à jour
```

### Utilisation

```bash
# Démarrer le monitoring
python start.py

# Dans le dashboard web
# Cliquer sur "Démarrer Surveillance"
```

---

## 📁 Structure du Projet

```
ids-ml/
├── app.py                      # Application Flask principale
├── config.py                   # Configuration centralisée
├── start.py                    # Script de démarrage
├── requirements.txt            # Dépendances Python
├── README.md                   # Ce fichier
│
├── src/                        # Code source
│   ├── __init__.py
│   ├── preprocessing.py        # Prétraitement des données
│   ├── models.py              # Modèles ML
│   ├── evaluation.py          # Métriques et évaluation
│   ├── visualization.py       # Visualisations
│   ├── alert_system.py        # Système d'alertes
│   ├── realtime_monitor.py    # Monitoring temps réel
│   ├── elk_integration.py     # Intégration ELK
│   └── feature_extraction.py  # Extraction de features
│
├── scripts/                    # Scripts utilitaires
│   ├── train.py               # Entraînement des modèles
│   └── predict.py             # Prédictions
│
├── web/                        # Interface web
│   ├── index.html             # Page d'accueil
│   ├── dashboard.html         # Dashboard temps réel
│   ├── demo.html              # Démo interactive
│   ├── about.html             # À propos
│   ├── css/                   # Styles CSS
│   └── js/                    # Scripts JavaScript
│
├── notebooks/                  # Jupyter Notebooks
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
│
├── data/                       # Données
│   ├── raw/                   # Données brutes
│   └── processed/             # Données traitées
│
├── models/                     # Modèles entraînés
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── neural_network_model.h5
│   └── preprocessor.pkl
│
└── logs/                       # Fichiers de log
    ├── app.log
    ├── alerts.log
    └── realtime_monitor.log
```

---

## 🤝 Contribuer

Ce projet est développé dans un cadre académique. Les contributions sont les bienvenues !

### Comment Contribuer

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit les changements (`git commit -m 'Ajout fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

---

## 📄 License

Ce projet est développé à des fins académiques et éducatives.

**© 2025 Rana Romdhane & Oulimata Sall**

Tous droits réservés. Voir [LICENSE](LICENSE) pour plus de détails.

---

## 📞 Contact

- **Rana Romdhane** - Développement & ML
- **Oulimata Sall** - Développement & Tests

**Email** : rana.romdhane@enicar.ucar.tn

---

## 🙏 Remerciements

- Communauté Open Source pour les outils et bibliothèques
- Datasets publics : CICIDS2017, UNSW-NB15
- TensorFlow, scikit-learn, Flask communities

---

## 📚 Références

1. [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
2. [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
3. [MITRE ATT&CK Framework](https://attack.mitre.org/)
4. Documentation TensorFlow & scikit-learn

---

**Projet Académique 2025 - Cybersécurité**  
*Système de Détection d'Intrusions avec Machine Learning*