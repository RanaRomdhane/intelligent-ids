"""
Module d'intégration avec ELK Stack (Elasticsearch, Logstash, Kibana).
Permet l'envoi et la visualisation des alertes et logs IDS.
"""

from datetime import datetime
import json
import logging
import os

logger = logging.getLogger(__name__)


class ELKIntegration:
    """Intégration avec Elasticsearch pour SIEM."""
    
    def __init__(self, hosts=None, username=None, password=None):
        """
        Initialise la connexion Elasticsearch.
        Essaie plusieurs hôtes possibles.
        """
        self.logger = logging.getLogger(__name__)
        self.es = None
        self.connected = False
        
        # Liste des hôtes à essayer (Docker et local)
        if hosts is None:
            hosts = [
                'http://localhost:9200',      # Local
                'http://elasticsearch:9200',   # Docker Compose
                'http://host.docker.internal:9200',  # Docker vers host
                'http://127.0.0.1:9200'       # Alternative local
            ]
        
        # Formater les hôtes
        formatted_hosts = []
        for host in hosts:
            if not host.startswith('http'):
                formatted_hosts.append(f"http://{host}")
            else:
                formatted_hosts.append(host)

        try:
            from elasticsearch import Elasticsearch
            
            # Essayer chaque hôte
            for host in formatted_hosts:
                try:
                    if username and password:
                        es_client = Elasticsearch(
                            [host],
                            basic_auth=(username, password),
                            verify_certs=False,
                            request_timeout=3,
                            max_retries=1,
                            retry_on_timeout=False
                        )
                    else:
                        es_client = Elasticsearch(
                            [host],
                            verify_certs=False,
                            request_timeout=3,
                            max_retries=1,
                            retry_on_timeout=False
                        )
                    
                    # Tester la connexion
                    if es_client.ping():
                        self.es = es_client
                        self.connected = True
                        self.logger.info(f"✓ Connexion Elasticsearch établie sur {host}")
                        break
                        
                except Exception:
                    continue
            
            if not self.connected:
                self.logger.warning("⚠️ Aucun serveur Elasticsearch disponible")
                    
        except ImportError:
            self.logger.error("Package 'elasticsearch' non installé")
            self.es = None
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur connexion Elasticsearch: {e}")
            self.es = None
    
    def create_indices(self):
        """Crée les indices nécessaires dans Elasticsearch."""
        if not self.es or not self.connected:
            return False
        
        indices = {
            'ids-alerts': {
                'mappings': {
                    'properties': {
                        'alert_id': {'type': 'keyword'},
                        'timestamp': {'type': 'date'},
                        'alert_type': {'type': 'keyword'},
                        'attack_category': {'type': 'keyword'},
                        'severity': {'type': 'keyword'},
                        'source_ip': {'type': 'keyword'},
                        'destination_ip': {'type': 'keyword'},
                        'description': {'type': 'text'},
                        'confidence_score': {'type': 'float'},
                        'status': {'type': 'keyword'}
                    }
                }
            },
            'ids-predictions': {
                'mappings': {
                    'properties': {
                        'timestamp': {'type': 'date'},
                        'model': {'type': 'keyword'},
                        'total_samples': {'type': 'integer'},
                        'predictions': {'type': 'object'},
                        'attack_categories': {'type': 'keyword'}
                    }
                }
            }
        }
        
        for index_name, index_body in indices.items():
            try:
                if not self.es.indices.exists(index=index_name):
                    self.es.indices.create(index=index_name, body=index_body)
                    self.logger.info(f"✓ Index '{index_name}' créé")
            except Exception as e:
                self.logger.warning(f"Index {index_name}: {e}")
        
        return True
    
    def index_alert(self, alert):
        """Indexe une alerte dans Elasticsearch."""
        if not self.es or not self.connected:
            return False
        
        try:
            # Convertir en dict si nécessaire
            if hasattr(alert, 'to_dict'):
                alert_dict = alert.to_dict()
            else:
                alert_dict = dict(alert)
            
            # S'assurer que timestamp est au bon format
            if 'timestamp' in alert_dict:
                if hasattr(alert_dict['timestamp'], 'isoformat'):
                    alert_dict['timestamp'] = alert_dict['timestamp'].isoformat()
            else:
                alert_dict['timestamp'] = datetime.now().isoformat()
            
            self.es.index(index='ids-alerts', document=alert_dict)
            return True
            
        except Exception as e:
            self.logger.debug(f"Erreur indexation: {e}")
            return False
    
    def index_prediction_result(self, prediction_data):
        """Indexe les résultats d'une prédiction."""
        if not self.es or not self.connected:
            return False
        
        try:
            doc = {
                'timestamp': datetime.now().isoformat(),
                'model': prediction_data.get('model', 'unknown'),
                'total_samples': prediction_data.get('total_samples', 0),
                'predictions': prediction_data.get('predictions', {}),
                'alerts_count': prediction_data.get('alerts_count', 0)
            }
            
            self.es.index(index='ids-predictions', document=doc)
            return True
        except Exception:
            return False
    
    def search_alerts(self, query=None, severity=None, size=100):
        """Recherche des alertes."""
        if not self.es or not self.connected:
            return []
        
        try:
            search_body = {
                'query': {'match_all': {}},
                'sort': [{'timestamp': {'order': 'desc'}}],
                'size': size
            }
            
            if severity:
                search_body['query'] = {'term': {'severity': severity}}
            
            response = self.es.search(index='ids-alerts', body=search_body)
            return [hit['_source'] for hit in response['hits']['hits']]
        except Exception:
            return []
    
    def get_statistics(self):
        """Retourne les statistiques des alertes."""
        if not self.es or not self.connected:
            return {'total': 0, 'by_severity': {}, 'by_type': {}}
        
        try:
            agg_body = {
                'size': 0,
                'aggs': {
                    'by_severity': {'terms': {'field': 'severity'}},
                    'by_type': {'terms': {'field': 'alert_type'}}
                }
            }
            
            response = self.es.search(index='ids-alerts', body=agg_body)
            
            return {
                'total': response['hits']['total']['value'],
                'by_severity': {
                    b['key']: b['doc_count'] 
                    for b in response['aggregations']['by_severity']['buckets']
                },
                'by_type': {
                    b['key']: b['doc_count'] 
                    for b in response['aggregations']['by_type']['buckets']
                }
            }
        except Exception:
            return {'total': 0, 'by_severity': {}, 'by_type': {}}


def setup_elk_integration():
    """Configure l'intégration ELK."""
    elk = ELKIntegration()
    
    if elk.connected:
        print("✓ Connexion à Elasticsearch établie")
        elk.create_indices()
        print("✓ Indices créés avec succès")
        
        print("\n📊 Dashboards Kibana suggérés:")
        print("  - IDS Overview Dashboard")
        
        return elk
    else:
        print("✗ Échec de connexion à Elasticsearch")
        return elk  # Retourne quand même l'objet, mais non connecté


if __name__ == "__main__":
    elk = setup_elk_integration()
    
    if elk.connected:
        # Test d'indexation
        test_alert = {
            'alert_id': 'TEST-001',
            'timestamp': datetime.now().isoformat(),
            'alert_type': 'DoS',
            'severity': 'high',
            'source_ip': '192.168.1.100',
            'destination_ip': '10.0.0.1',
            'description': 'Test alert',
            'confidence_score': 0.95,
            'status': 'new'
        }
        
        if elk.index_alert(test_alert):
            print("✓ Test d'indexation réussi")
        else:
            print("✗ Échec du test d'indexation")