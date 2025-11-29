"""
Module de surveillance en temps réel pour IDS.
Capture et analyse le trafic réseau en temps réel.
"""

import threading
import time
import queue
from datetime import datetime
import pandas as pd
import numpy as np
from scapy.all import sniff, IP, TCP, UDP
import logging


class RealTimeMonitor:
    """Moniteur de trafic réseau en temps réel."""
    
    def __init__(self, model, preprocessor, alert_manager, interface='eth0'):
        """
        Initialise le moniteur temps réel.
        
        Args:
            model: Modèle ML entraîné
            preprocessor: Préprocesseur de données
            alert_manager: Gestionnaire d'alertes
            interface: Interface réseau à surveiller
        """
        self.model = model
        self.preprocessor = preprocessor
        self.alert_manager = alert_manager
        self.interface = interface
        
        self.packet_queue = queue.Queue(maxsize=1000)
        self.running = False
        self.packet_count = 0
        self.attack_count = 0
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Statistiques en temps réel
        self.stats = {
            'total_packets': 0,
            'packets_per_second': 0,
            'attacks_detected': 0,
            'protocols': {},
            'top_sources': {},
            'top_destinations': {},
            'start_time': None
        }
    
    def _setup_logging(self):
        """Configure le système de logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/realtime_monitor.log'),
                logging.StreamHandler()
            ]
        )
    
    def start(self):
        """Démarre la surveillance en temps réel."""
        if self.running:
            self.logger.warning("Le moniteur est déjà en cours d'exécution")
            return
        
        self.running = True
        self.stats['start_time'] = datetime.now()
        
        # Démarrer les threads
        capture_thread = threading.Thread(target=self._capture_packets, daemon=True)
        analysis_thread = threading.Thread(target=self._analyze_packets, daemon=True)
        stats_thread = threading.Thread(target=self._update_stats, daemon=True)
        
        capture_thread.start()
        analysis_thread.start()
        stats_thread.start()
        
        self.logger.info(f"Surveillance temps réel démarrée sur {self.interface}")
    
    def stop(self):
        """Arrête la surveillance."""
        self.running = False
        self.logger.info("Surveillance temps réel arrêtée")
    
    def _capture_packets(self):
        """Capture les paquets réseau."""
        def packet_handler(packet):
            if not self.running:
                return
            
            try:
                if IP in packet:
                    self.packet_queue.put(packet, timeout=1)
                    self.packet_count += 1
            except queue.Full:
                self.logger.warning("Queue de paquets pleine, paquet ignoré")
        
        try:
            sniff(
                iface=self.interface,
                prn=packet_handler,
                store=False,
                stop_filter=lambda x: not self.running
            )
        except Exception as e:
            self.logger.error(f"Erreur de capture: {e}")
    
    def _analyze_packets(self):
        """Analyse les paquets capturés."""
        batch_size = 10
        batch = []
        
        while self.running:
            try:
                packet = self.packet_queue.get(timeout=1)
                batch.append(packet)
                
                if len(batch) >= batch_size:
                    self._process_batch(batch)
                    batch = []
                    
            except queue.Empty:
                if batch:
                    self._process_batch(batch)
                    batch = []
                continue
            except Exception as e:
                self.logger.error(f"Erreur d'analyse: {e}")
    
    def _process_batch(self, packets):
        """Traite un batch de paquets."""
        try:
            # Extraire les features des paquets
            features_list = []
            packet_data_list = []
            
            for packet in packets:
                features, packet_data = self._extract_packet_features(packet)
                if features:
                    features_list.append(features)
                    packet_data_list.append(packet_data)
            
            if not features_list:
                return
            
            # Créer un DataFrame
            df = pd.DataFrame(features_list)
            
            # Prétraiter
            df_processed = self.preprocessor.extract_features(df)
            X = df_processed[self.preprocessor.feature_columns]
            
            # Prédire
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # Traiter les prédictions
            for i, (pred, proba, packet_data) in enumerate(zip(predictions, probabilities, packet_data_list)):
                if pred != 0:  # Si ce n'est pas normal (0 = normal)
                    confidence = np.max(proba)
                    
                    # Décoder la prédiction
                    if hasattr(self.preprocessor.label_encoder, 'classes_'):
                        attack_type = self.preprocessor.label_encoder.inverse_transform([pred])[0]
                    else:
                        attack_type = str(pred)
                    
                    # Créer une alerte
                    alert = self.alert_manager.create_alert(
                        prediction=attack_type,
                        flow_data=packet_data,
                        confidence_score=confidence
                    )
                    
                    self.attack_count += 1
                    self.stats['attacks_detected'] += 1
                    
                    self.logger.warning(
                        f"⚠️ Attaque détectée: {attack_type} "
                        f"(confiance: {confidence*100:.1f}%) "
                        f"de {packet_data['src_ip']} vers {packet_data['dst_ip']}"
                    )
        
        except Exception as e:
            self.logger.error(f"Erreur de traitement du batch: {e}")
    
    def _extract_packet_features(self, packet):
        """Extrait les features d'un paquet."""
        try:
            if not IP in packet:
                return None, None
            
            features = {}
            packet_data = {}
            
            # IP features
            ip_layer = packet[IP]
            features['src_ip'] = ip_layer.src
            features['dst_ip'] = ip_layer.dst
            features['ttl'] = ip_layer.ttl
            features['ip_len'] = ip_layer.len
            
            packet_data['src_ip'] = ip_layer.src
            packet_data['dst_ip'] = ip_layer.dst
            
            # Protocol
            if TCP in packet:
                tcp_layer = packet[TCP]
                features['protocol'] = 'tcp'
                features['src_port'] = tcp_layer.sport
                features['dst_port'] = tcp_layer.dport
                features['flags'] = int(tcp_layer.flags)
                
                packet_data['protocol'] = 'tcp'
                packet_data['src_port'] = tcp_layer.sport
                packet_data['dst_port'] = tcp_layer.dport
                
                # Update stats
                self.stats['protocols']['tcp'] = self.stats['protocols'].get('tcp', 0) + 1
                
            elif UDP in packet:
                udp_layer = packet[UDP]
                features['protocol'] = 'udp'
                features['src_port'] = udp_layer.sport
                features['dst_port'] = udp_layer.dport
                features['flags'] = 0
                
                packet_data['protocol'] = 'udp'
                packet_data['src_port'] = udp_layer.sport
                packet_data['dst_port'] = udp_layer.dport
                
                # Update stats
                self.stats['protocols']['udp'] = self.stats['protocols'].get('udp', 0) + 1
            else:
                features['protocol'] = 'other'
                features['src_port'] = 0
                features['dst_port'] = 0
                features['flags'] = 0
                
                packet_data['protocol'] = 'other'
                
                # Update stats
                self.stats['protocols']['other'] = self.stats['protocols'].get('other', 0) + 1
            
            # Packet size
            features['packet_size'] = len(packet)
            
            # Timestamp
            features['timestamp'] = datetime.now()
            
            # Update IP stats
            self._update_ip_stats(features['src_ip'], features['dst_ip'])
            
            return features, packet_data
            
        except Exception as e:
            self.logger.error(f"Erreur d'extraction de features: {e}")
            return None, None
    
    def _update_ip_stats(self, src_ip, dst_ip):
        """Met à jour les statistiques d'IP."""
        self.stats['top_sources'][src_ip] = self.stats['top_sources'].get(src_ip, 0) + 1
        self.stats['top_destinations'][dst_ip] = self.stats['top_destinations'].get(dst_ip, 0) + 1
    
    def _update_stats(self):
        """Met à jour les statistiques périodiquement."""
        while self.running:
            time.sleep(1)
            
            # Calculer packets par seconde
            if self.stats['start_time']:
                elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
                if elapsed > 0:
                    self.stats['packets_per_second'] = self.packet_count / elapsed
            
            self.stats['total_packets'] = self.packet_count
    
    def get_stats(self):
        """Retourne les statistiques actuelles."""
        # Top 5 sources et destinations
        top_sources = sorted(
            self.stats['top_sources'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        top_destinations = sorted(
            self.stats['top_destinations'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'total_packets': self.stats['total_packets'],
            'packets_per_second': round(self.stats['packets_per_second'], 2),
            'attacks_detected': self.stats['attacks_detected'],
            'protocols': self.stats['protocols'],
            'top_sources': dict(top_sources),
            'top_destinations': dict(top_destinations),
            'uptime': str(datetime.now() - self.stats['start_time']) if self.stats['start_time'] else '0:00:00'
        }
    
    def get_dashboard_data(self):
        """Retourne les données pour le dashboard."""
        stats = self.get_stats()
        alerts = self.alert_manager.get_statistics()
        
        return {
            'monitoring': stats,
            'alerts': alerts,
            'recent_alerts': [a.to_dict() for a in self.alert_manager.get_alerts(limit=10)]
        }


class NetworkFlowAggregator:
    """Agrège les paquets en flux réseau pour une meilleure détection."""
    
    def __init__(self, timeout=60):
        """
        Initialise l'agrégateur de flux.
        
        Args:
            timeout: Durée en secondes avant expiration d'un flux
        """
        self.flows = {}
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    def add_packet(self, packet_features):
        """Ajoute un paquet à un flux existant ou crée un nouveau flux."""
        flow_key = self._get_flow_key(packet_features)
        current_time = datetime.now()
        
        if flow_key in self.flows:
            flow = self.flows[flow_key]
            flow['packet_count'] += 1
            flow['total_bytes'] += packet_features.get('packet_size', 0)
            flow['last_seen'] = current_time
            flow['packets'].append(packet_features)
        else:
            self.flows[flow_key] = {
                'src_ip': packet_features['src_ip'],
                'dst_ip': packet_features['dst_ip'],
                'src_port': packet_features.get('src_port', 0),
                'dst_port': packet_features.get('dst_port', 0),
                'protocol': packet_features.get('protocol', 'unknown'),
                'packet_count': 1,
                'total_bytes': packet_features.get('packet_size', 0),
                'first_seen': current_time,
                'last_seen': current_time,
                'packets': [packet_features]
            }
        
        # Nettoyer les flux expirés
        self._cleanup_expired_flows(current_time)
    
    def _get_flow_key(self, packet_features):
        """Génère une clé unique pour un flux."""
        return (
            packet_features['src_ip'],
            packet_features['dst_ip'],
            packet_features.get('src_port', 0),
            packet_features.get('dst_port', 0),
            packet_features.get('protocol', 'unknown')
        )
    
    def _cleanup_expired_flows(self, current_time):
        """Supprime les flux expirés."""
        expired_keys = []
        for key, flow in self.flows.items():
            if (current_time - flow['last_seen']).total_seconds() > self.timeout:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.flows[key]
    
    def get_flow_features(self, flow_key):
        """Extrait les features d'un flux pour la détection."""
        if flow_key not in self.flows:
            return None
        
        flow = self.flows[flow_key]
        duration = (flow['last_seen'] - flow['first_seen']).total_seconds()
        
        return {
            'duration': duration,
            'packet_count': flow['packet_count'],
            'total_bytes': flow['total_bytes'],
            'packets_per_second': flow['packet_count'] / max(duration, 1),
            'bytes_per_second': flow['total_bytes'] / max(duration, 1),
            'avg_packet_size': flow['total_bytes'] / flow['packet_count'],
            'protocol': flow['protocol'],
            'src_ip': flow['src_ip'],
            'dst_ip': flow['dst_ip'],
            'src_port': flow['src_port'],
            'dst_port': flow['dst_port']
        }