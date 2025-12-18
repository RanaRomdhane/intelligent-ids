"""
Module d'intégration avec ELK Stack (Elasticsearch, Logstash, Kibana).
Permet l'envoi et la visualisation des alertes et logs IDS.
"""

from elasticsearch import Elasticsearch, helpers
from datetime import datetime
import json
import logging


class ELKIntegration:
    """Intégration avec Elasticsearch pour SIEM."""
    
    def __init__(self, hosts=['http://localhost:9200'], username=None, password=None):
        """
        Initialise la connexion Elasticsearch.
        """
        self.logger = logging.getLogger(__name__)
        
        # CORRECTION : On s'assure que chaque hôte commence par http://
        formatted_hosts = []
        for host in hosts:
            if not host.startswith('http'):
                formatted_hosts.append(f"http://{host}")
            else:
                formatted_hosts.append(host)

        try:
            if username and password:
                self.es = Elasticsearch(
                    formatted_hosts,
                    basic_auth=(username, password),
                    verify_certs=False
                )
            else:
                self.es = Elasticsearch(
                    formatted_hosts,
                    verify_certs=False
                )
            
            # Vérifier la connexion
            if self.es.ping():
                self.logger.info(f"✓ Connexion à Elasticsearch établie sur {formatted_hosts}")
            else:
                self.logger.error("✗ Impossible de se connecter à Elasticsearch (Ping échoué)")
                
        except Exception as e:
            self.logger.error(f"Erreur de connexion Elasticsearch: {e}")
            self.es = None
    
    def create_indices(self):
        """Crée les indices nécessaires dans Elasticsearch."""
        if not self.es:
            return False
        
        indices = {
            'ids-alerts': {
                'mappings': {
                    'properties': {
                        'alert_id': {'type': 'keyword'},
                        'timestamp': {'type': 'date'},
                        'alert_type': {'type': 'keyword'},
                        'severity': {'type': 'keyword'},
                        'source_ip': {'type': 'ip'},
                        'destination_ip': {'type': 'ip'},
                        'description': {'type': 'text'},
                        'confidence_score': {'type': 'float'},
                        'status': {'type': 'keyword'}
                    }
                }
            },
            'ids-traffic': {
                'mappings': {
                    'properties': {
                        'timestamp': {'type': 'date'},
                        'source_ip': {'type': 'ip'},
                        'destination_ip': {'type': 'ip'},
                        'source_port': {'type': 'integer'},
                        'destination_port': {'type': 'integer'},
                        'protocol': {'type': 'keyword'},
                        'packet_size': {'type': 'integer'},
                        'prediction': {'type': 'keyword'},
                        'confidence': {'type': 'float'}
                    }
                }
            },
            'ids-statistics': {
                'mappings': {
                    'properties': {
                        'timestamp': {'type': 'date'},
                        'total_packets': {'type': 'long'},
                        'packets_per_second': {'type': 'float'},
                        'attacks_detected': {'type': 'integer'},
                        'protocol_distribution': {'type': 'object'}
                    }
                }
            }
        }
        
        for index_name, index_body in indices.items():
            try:
                if not self.es.indices.exists(index=index_name):
                    self.es.indices.create(index=index_name, body=index_body)
                    self.logger.info(f"Index '{index_name}' créé")
                else:
                    self.logger.info(f"Index '{index_name}' existe déjà")
            except Exception as e:
                self.logger.error(f"Erreur création index {index_name}: {e}")
                return False
        
        return True
    
    def index_alert(self, alert):
        """
        Indexe une alerte dans Elasticsearch.
        
        Args:
            alert: Objet Alert à indexer
        
        Returns:
            bool: True si succès
        """
        if not self.es:
            return False
        
        try:
            alert_dict = alert.to_dict()
            
            # Convertir timestamp en format ISO
            if isinstance(alert_dict['timestamp'], str):
                alert_dict['timestamp'] = alert_dict['timestamp']
            else:
                alert_dict['timestamp'] = alert_dict['timestamp'].isoformat()
            
            # Indexer
            response = self.es.index(
                index='ids-alerts',
                document=alert_dict
            )
            
            self.logger.info(f"Alerte {alert.alert_id} indexée dans Elasticsearch")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur indexation alerte: {e}")
            return False
    
    def index_traffic(self, packet_data):
        """
        Indexe des données de trafic réseau.
        
        Args:
            packet_data: Dictionnaire contenant les données du paquet
        
        Returns:
            bool: True si succès
        """
        if not self.es:
            return False
        
        try:
            # Ajouter timestamp si absent
            if 'timestamp' not in packet_data:
                packet_data['timestamp'] = datetime.now().isoformat()
            
            self.es.index(
                index='ids-traffic',
                document=packet_data
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur indexation trafic: {e}")
            return False
    
    def bulk_index_traffic(self, traffic_data_list):
        """
        Indexe en masse des données de trafic.
        
        Args:
            traffic_data_list: Liste de dictionnaires de données de trafic
        
        Returns:
            int: Nombre de documents indexés
        """
        if not self.es or not traffic_data_list:
            return 0
        
        try:
            actions = []
            for traffic_data in traffic_data_list:
                if 'timestamp' not in traffic_data:
                    traffic_data['timestamp'] = datetime.now().isoformat()
                
                actions.append({
                    '_index': 'ids-traffic',
                    '_source': traffic_data
                })
            
            success, failed = helpers.bulk(self.es, actions)
            self.logger.info(f"{success} documents de trafic indexés")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Erreur bulk indexation: {e}")
            return 0
    
    def index_statistics(self, stats):
        """
        Indexe des statistiques système.
        
        Args:
            stats: Dictionnaire de statistiques
        
        Returns:
            bool: True si succès
        """
        if not self.es:
            return False
        
        try:
            stats['timestamp'] = datetime.now().isoformat()
            
            self.es.index(
                index='ids-statistics',
                document=stats
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur indexation statistiques: {e}")
            return False
    
    def search_alerts(self, query=None, severity=None, start_time=None, 
                      end_time=None, size=100):
        """
        Recherche des alertes avec filtres.
        
        Args:
            query: Requête de recherche
            severity: Filtrer par sévérité
            start_time: Date de début
            end_time: Date de fin
            size: Nombre max de résultats
        
        Returns:
            list: Liste des alertes trouvées
        """
        if not self.es:
            return []
        
        try:
            # Construire la requête
            must_clauses = []
            
            if query:
                must_clauses.append({
                    'multi_match': {
                        'query': query,
                        'fields': ['description', 'alert_type']
                    }
                })
            
            if severity:
                must_clauses.append({'term': {'severity': severity}})
            
            if start_time or end_time:
                range_clause = {'range': {'timestamp': {}}}
                if start_time:
                    range_clause['range']['timestamp']['gte'] = start_time
                if end_time:
                    range_clause['range']['timestamp']['lte'] = end_time
                must_clauses.append(range_clause)
            
            search_body = {
                'query': {
                    'bool': {
                        'must': must_clauses if must_clauses else [{'match_all': {}}]
                    }
                },
                'sort': [{'timestamp': {'order': 'desc'}}],
                'size': size
            }
            
            response = self.es.search(
                index='ids-alerts',
                body=search_body
            )
            
            return [hit['_source'] for hit in response['hits']['hits']]
            
        except Exception as e:
            self.logger.error(f"Erreur recherche alertes: {e}")
            return []
    
    def get_alert_statistics(self, start_time=None, end_time=None):
        """
        Obtient des statistiques sur les alertes.
        
        Args:
            start_time: Date de début
            end_time: Date de fin
        
        Returns:
            dict: Statistiques d'alertes
        """
        if not self.es:
            return {}
        
        try:
            # Construire le filtre de temps
            time_filter = []
            if start_time or end_time:
                range_filter = {'range': {'timestamp': {}}}
                if start_time:
                    range_filter['range']['timestamp']['gte'] = start_time
                if end_time:
                    range_filter['range']['timestamp']['lte'] = end_time
                time_filter.append(range_filter)
            
            # Agrégations
            agg_body = {
                'query': {
                    'bool': {
                        'filter': time_filter if time_filter else []
                    }
                },
                'aggs': {
                    'by_severity': {
                        'terms': {'field': 'severity'}
                    },
                    'by_type': {
                        'terms': {'field': 'alert_type'}
                    },
                    'by_status': {
                        'terms': {'field': 'status'}
                    },
                    'timeline': {
                        'date_histogram': {
                            'field': 'timestamp',
                            'calendar_interval': 'hour'
                        }
                    }
                },
                'size': 0
            }
            
            response = self.es.search(
                index='ids-alerts',
                body=agg_body
            )
            
            return {
                'total': response['hits']['total']['value'],
                'by_severity': {
                    bucket['key']: bucket['doc_count'] 
                    for bucket in response['aggregations']['by_severity']['buckets']
                },
                'by_type': {
                    bucket['key']: bucket['doc_count']
                    for bucket in response['aggregations']['by_type']['buckets']
                },
                'by_status': {
                    bucket['key']: bucket['doc_count']
                    for bucket in response['aggregations']['by_status']['buckets']
                },
                'timeline': [
                    {
                        'timestamp': bucket['key_as_string'],
                        'count': bucket['doc_count']
                    }
                    for bucket in response['aggregations']['timeline']['buckets']
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Erreur statistiques alertes: {e}")
            return {}
    
    def create_kibana_dashboards(self):
        """Crée des dashboards Kibana prédéfinis."""
        # Configuration des dashboards Kibana
        dashboards = {
            'ids-overview': {
                'title': 'IDS Overview Dashboard',
                'description': 'Vue d\'ensemble du système IDS',
                'visualizations': [
                    'alerts-by-severity',
                    'alerts-timeline',
                    'top-attack-types',
                    'network-traffic-volume'
                ]
            }
        }
        
        self.logger.info("Configuration des dashboards Kibana à effectuer manuellement")
        return dashboards


def setup_elk_integration():
    """Configure l'intégration ELK complète."""
    elk = ELKIntegration(
        # CORRECTION ICI : Ajout de http://
        hosts=['http://localhost:9200'],
        # username='elastic',
        # password='your_password'
    )
    
    if elk.es and elk.es.ping():
        print("✓ Connexion à Elasticsearch établie")
        
        # Créer les indices
        if elk.create_indices():
            print("✓ Indices créés avec succès")
        
        # Afficher la configuration Kibana
        dashboards = elk.create_kibana_dashboards()
        print("\n📊 Dashboards Kibana suggérés:")
        for name, config in dashboards.items():
            print(f"  - {config['title']}")
        
        return elk
    else:
        print("✗ Échec de connexion à Elasticsearch")
        return None


if __name__ == "__main__":
    # Test de l'intégration
    elk = setup_elk_integration()
    
    if elk:
        # Tester l'indexation
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
        
        print("\n🧪 Test d'indexation...")
        # elk.index_alert(test_alert)
        
        print("✓ Intégration ELK configurée")