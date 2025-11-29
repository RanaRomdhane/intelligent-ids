#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de démarrage du système IDS ML.
Initialise tous les composants et démarre le serveur web.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def print_banner():
    """Affiche la bannière de démarrage."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║       IDS ML - Intelligent Intrusion Detection System        ║
    ║                                                               ║
    ║       Système de Détection d'Intrusions avec ML              ║
    ║       Projet Académique 2025                                  ║
    ║                                                               ║
    ║       Développé par:                                          ║
    ║       • Rana Romdhane                                         ║
    ║       • Oulimata Sall                                         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_requirements():
    """Vérifie que toutes les dépendances sont installées."""
    print("\n📦 Vérification des dépendances...")
    
    required_packages = [
        'flask', 'numpy', 'pandas', 'sklearn', 
        'tensorflow', 'flask_socketio', 'elasticsearch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Special case for scikit-learn
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MANQUANT")
            missing_packages.append(package if package != 'sklearn' else 'scikit-learn')
    
    if missing_packages:
        print(f"\n⚠️  Packages manquants: {', '.join(missing_packages)}")
        print("   Exécutez: pip install -r requirements.txt")
        return False
    
    print("✓ Toutes les dépendances sont installées\n")
    return True


def check_directories():
    """Vérifie et crée les répertoires nécessaires."""
    print("📁 Vérification des répertoires...")
    
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'logs',
        'uploads',
        'web/css',
        'web/js'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("✓ Tous les répertoires sont prêts\n")


def check_models():
    """Vérifie si les modèles sont entraînés."""
    print("🤖 Vérification des modèles ML...")
    
    model_files = [
        'models/preprocessor.pkl',
        'models/random_forest_model.pkl',
        'models/svm_model.pkl',
        'models/neural_network_model.h5'
    ]
    
    models_exist = all(os.path.exists(f) for f in model_files)
    
    if models_exist:
        print("✓ Modèles ML trouvés\n")
        return True
    else:
        print("⚠️  Modèles ML non trouvés")
        print("   Les modèles doivent être entraînés avant utilisation")
        print("   Exécutez: python scripts/train.py\n")
        return False


def check_elk_stack():
    """Vérifie la connexion à ELK Stack."""
    print("🔍 Vérification de ELK Stack...")
    
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(['localhost:9200'], request_timeout=2)
        
        if es.ping():
            print("✓ ELK Stack connecté\n")
            return True
        else:
            print("⚠️  ELK Stack non accessible")
            print("   Le système fonctionnera sans intégration SIEM\n")
            return False
    except Exception as e:
        print("⚠️  ELK Stack non disponible")
        print("   Le système fonctionnera sans intégration SIEM\n")
        return False


def train_models():
    """Lance l'entraînement des modèles."""
    print("\n🎓 Entraînement des modèles ML...")
    print("   Cela peut prendre plusieurs minutes...\n")
    
    try:
        subprocess.run([sys.executable, 'scripts/train.py'], check=True)
        print("\n✓ Modèles entraînés avec succès\n")
        return True
    except subprocess.CalledProcessError:
        print("\n✗ Erreur lors de l'entraînement des modèles\n")
        return False


def start_server(port=5000, debug=True):
    """Démarre le serveur Flask."""
    print(f"\n🚀 Démarrage du serveur sur le port {port}...")
    print(f"\n📍 Accès au système:")
    print(f"   • Interface Web: http://localhost:{port}")
    print(f"   • Dashboard: http://localhost:{port}/dashboard")
    print(f"   • Démo: http://localhost:{port}/demo.html")
    print(f"\n⌨️  Appuyez sur Ctrl+C pour arrêter le serveur\n")
    print("="*70 + "\n")
    
    try:
        # Import et lancement de l'application
        from app import app, socketio
        socketio.run(app, debug=debug, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Arrêt du serveur...")
    except Exception as e:
        print(f"\n✗ Erreur lors du démarrage: {e}")
        sys.exit(1)


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description='Démarrer le système IDS ML'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port du serveur web (défaut: 5000)'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Entraîner les modèles avant de démarrer'
    )
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Sauter les vérifications initiales'
    )
    parser.add_argument(
        '--no-debug',
        action='store_true',
        help='Désactiver le mode debug'
    )
    
    args = parser.parse_args()
    
    # Afficher la bannière
    print_banner()
    
    if not args.skip_checks:
        # Vérifications
        if not check_requirements():
            print("❌ Veuillez installer les dépendances manquantes")
            print("\nCommande d'installation:")
            print("pip install -r requirements.txt")
            sys.exit(1)
        
        check_directories()
        
        models_ready = check_models()
        
        if args.train or not models_ready:
            response = input("Voulez-vous entraîner les modèles maintenant? (o/N): ")
            if response.lower() in ['o', 'oui', 'y', 'yes']:
                if not train_models():
                    sys.exit(1)
            elif not models_ready:
                print("\n⚠️  Attention: Le système démarrera sans modèles entraînés")
                print("   Certaines fonctionnalités ne seront pas disponibles\n")
                input("Appuyez sur Entrée pour continuer...")
        
        check_elk_stack()
    
    # Démarrer le serveur
    start_server(port=args.port, debug=not args.no_debug)


if __name__ == '__main__':
    main()