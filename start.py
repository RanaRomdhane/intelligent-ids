#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de démarrage du système IDS ML.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def print_banner():
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
    print("\n📦 Vérification des dépendances...")
    required = ['flask', 'numpy', 'pandas', 'sklearn', 'tensorflow', 'flask_socketio', 'elasticsearch']
    missing = []
    for pkg in required:
        try:
            __import__('sklearn' if pkg == 'sklearn' else pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} - MANQUANT")
            missing.append(pkg if pkg != 'sklearn' else 'scikit-learn')
    
    if missing:
        print(f"\n⚠️  Packages manquants: {', '.join(missing)}")
        return False
    print("✓ Toutes les dépendances sont installées\n")
    return True


def check_directories():
    print("📁 Vérification des répertoires...")
    dirs = ['data/raw', 'data/processed', 'models', 'logs', 'uploads', 'web/css', 'web/js']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")
    print("✓ Tous les répertoires sont prêts\n")


def check_models():
    print("🤖 Vérification des modèles ML...")
    files = ['models/preprocessor.pkl', 'models/random_forest_model.pkl', 
             'models/svm_model.pkl', 'models/neural_network_model.h5']
    if all(os.path.exists(f) for f in files):
        print("✓ Modèles ML trouvés\n")
        return True
    print("⚠️  Modèles ML non trouvés\n")
    return False


def check_elk_stack():
    print("🔍 Vérification de ELK Stack...")
    try:
        from elasticsearch import Elasticsearch
        # Essayer plusieurs hôtes
        hosts = ['http://localhost:9200', 'http://127.0.0.1:9200']
        for host in hosts:
            try:
                es = Elasticsearch([host], request_timeout=2, max_retries=1)
                if es.ping():
                    print(f"✓ ELK Stack connecté ({host})\n")
                    return True
            except:
                continue
        print("⚠️  ELK Stack non disponible")
        print("   Le système fonctionnera sans intégration SIEM\n")
        return False
    except Exception:
        print("⚠️  ELK Stack non disponible\n")
        return False


def start_server(port=5000, debug=True):
    print(f"\n🚀 Démarrage du serveur sur le port {port}...")
    print(f"\n📍 Accès au système:")
    print(f"   • Interface Web: http://localhost:{port}")
    print(f"   • Dashboard: http://localhost:{port}/dashboard")
    print(f"   • Démo: http://localhost:{port}/demo.html")
    print(f"   • Kibana: http://localhost:5601")
    print(f"\n⌨️  Appuyez sur Ctrl+C pour arrêter le serveur\n")
    print("="*70 + "\n")
    
    try:
        from app import app, socketio
        socketio.run(app, debug=debug, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Arrêt du serveur...")
    except Exception as e:
        print(f"\n✗ Erreur: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Démarrer le système IDS ML')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--skip-checks', action='store_true')
    parser.add_argument('--no-debug', action='store_true')
    args = parser.parse_args()
    
    print_banner()
    
    if not args.skip_checks:
        if not check_requirements():
            sys.exit(1)
        check_directories()
        models_ready = check_models()
        
        if args.train or not models_ready:
            response = input("Voulez-vous entraîner les modèles? (o/N): ")
            if response.lower() in ['o', 'oui', 'y', 'yes']:
                subprocess.run([sys.executable, 'scripts/train.py'], check=True)
        
        check_elk_stack()
    
    start_server(port=args.port, debug=not args.no_debug)


if __name__ == '__main__':
    main()