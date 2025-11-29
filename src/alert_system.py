"""
Module de système d'alertes pour IDS.
Gère la détection, la génération et l'envoi d'alertes en temps réel.
"""

import json
import logging
from datetime import datetime
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests


class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types d'attaques détectées."""
    DOS = "DoS/DDoS"
    PROBE = "Port Scan/Probe"
    R2L = "Remote to Local"
    U2R = "User to Root"
    BOTNET = "Botnet Activity"
    BRUTE_FORCE = "Brute Force"
    EXFILTRATION = "Data Exfiltration"
    INJECTION = "Injection Attack"
    UNKNOWN = "Unknown Attack"


class Alert:
    """Classe représentant une alerte de sécurité."""
    
    def __init__(self, alert_type, severity, source_ip, destination_ip, 
                 description, confidence_score, timestamp=None):
        self.alert_id = self._generate_alert_id()
        self.alert_type = alert_type
        self.severity = severity
        self.source_ip = source_ip
        self.destination_ip = destination_ip
        self.description = description
        self.confidence_score = confidence_score
        self.timestamp = timestamp or datetime.now()
        self.status = "new"
        
    def _generate_alert_id(self):
        """Génère un ID unique pour l'alerte."""
        return f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    
    def to_dict(self):
        """Convertit l'alerte en dictionnaire."""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value if isinstance(self.alert_type, AlertType) else self.alert_type,
            'severity': self.severity.value if isinstance(self.severity, AlertSeverity) else self.severity,
            'source_ip': self.source_ip,
            'destination_ip': self.destination_ip,
            'description': self.description,
            'confidence_score': self.confidence_score,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status
        }
    
    def to_json(self):
        """Convertit l'alerte en JSON."""
        return json.dumps(self.to_dict(), indent=2)


class AlertManager:
    """Gestionnaire d'alertes pour l'IDS."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.alerts = []
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure le système de logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/alerts.log'),
                logging.StreamHandler()
            ]
        )
    
    def create_alert(self, prediction, flow_data, confidence_score):
        """
        Crée une alerte basée sur une prédiction d'attaque.
        
        Args:
            prediction: Type d'attaque prédit
            flow_data: Données du flux réseau
            confidence_score: Score de confiance de la prédiction
        
        Returns:
            Alert: Objet alerte créé
        """
        # Mapper les prédictions aux types d'alertes
        alert_type_map = {
            'dos': AlertType.DOS,
            'ddos': AlertType.DOS,
            'probe': AlertType.PROBE,
            'scan': AlertType.PROBE,
            'r2l': AlertType.R2L,
            'u2r': AlertType.U2R,
            'botnet': AlertType.BOTNET,
            'brute_force': AlertType.BRUTE_FORCE,
            'exfiltration': AlertType.EXFILTRATION,
            'injection': AlertType.INJECTION
        }
        
        alert_type = alert_type_map.get(prediction.lower(), AlertType.UNKNOWN)
        
        # Déterminer la sévérité basée sur le type et le score de confiance
        severity = self._determine_severity(alert_type, confidence_score)
        
        # Extraire les informations du flux
        source_ip = flow_data.get('src_ip', 'Unknown')
        dest_ip = flow_data.get('dst_ip', 'Unknown')
        
        description = self._generate_description(alert_type, flow_data, confidence_score)
        
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            source_ip=source_ip,
            destination_ip=dest_ip,
            description=description,
            confidence_score=confidence_score
        )
        
        self.alerts.append(alert)
        self.logger.info(f"Alerte créée: {alert.alert_id} - {alert.alert_type.value}")
        
        # Envoyer les notifications
        self._send_notifications(alert)
        
        return alert
    
    def _determine_severity(self, alert_type, confidence_score):
        """Détermine la sévérité de l'alerte."""
        critical_types = [AlertType.DOS, AlertType.U2R, AlertType.EXFILTRATION]
        high_types = [AlertType.BRUTE_FORCE, AlertType.BOTNET, AlertType.INJECTION]
        
        if alert_type in critical_types and confidence_score > 0.9:
            return AlertSeverity.CRITICAL
        elif alert_type in critical_types or (alert_type in high_types and confidence_score > 0.85):
            return AlertSeverity.HIGH
        elif confidence_score > 0.7:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _generate_description(self, alert_type, flow_data, confidence_score):
        """Génère une description détaillée de l'alerte."""
        descriptions = {
            AlertType.DOS: f"Attaque DoS/DDoS détectée avec {confidence_score*100:.1f}% de confiance",
            AlertType.PROBE: f"Scan de port ou activité de reconnaissance détectée ({confidence_score*100:.1f}% confiance)",
            AlertType.R2L: f"Tentative d'accès non autorisé Remote-to-Local détectée ({confidence_score*100:.1f}% confiance)",
            AlertType.U2R: f"Tentative d'élévation de privilèges User-to-Root détectée ({confidence_score*100:.1f}% confiance)",
            AlertType.BOTNET: f"Activité de botnet détectée ({confidence_score*100:.1f}% confiance)",
            AlertType.BRUTE_FORCE: f"Attaque par force brute détectée ({confidence_score*100:.1f}% confiance)",
            AlertType.EXFILTRATION: f"Tentative d'exfiltration de données détectée ({confidence_score*100:.1f}% confiance)",
            AlertType.INJECTION: f"Attaque par injection détectée ({confidence_score*100:.1f}% confiance)"
        }
        
        base_description = descriptions.get(alert_type, f"Activité suspecte détectée ({confidence_score*100:.1f}% confiance)")
        
        # Ajouter des détails supplémentaires si disponibles
        details = []
        if 'src_port' in flow_data:
            details.append(f"Port source: {flow_data['src_port']}")
        if 'dst_port' in flow_data:
            details.append(f"Port destination: {flow_data['dst_port']}")
        if 'protocol' in flow_data:
            details.append(f"Protocole: {flow_data['protocol']}")
        
        if details:
            base_description += " | " + " | ".join(details)
        
        return base_description
    
    def _send_notifications(self, alert):
        """Envoie des notifications pour l'alerte."""
        if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            # Envoyer email
            if self.config.get('email_enabled'):
                self._send_email_notification(alert)
            
            # Envoyer vers SIEM/ELK
            if self.config.get('siem_enabled'):
                self._send_to_siem(alert)
            
            # Webhook
            if self.config.get('webhook_url'):
                self._send_webhook(alert)
    
    def _send_email_notification(self, alert):
        """Envoie une notification par email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.get('smtp_from')
            msg['To'] = self.config.get('smtp_to')
            msg['Subject'] = f"[IDS ALERT - {alert.severity.value.upper()}] {alert.alert_type.value}"
            
            body = f"""
            Alerte de Sécurité IDS
            =====================
            
            ID Alerte: {alert.alert_id}
            Type: {alert.alert_type.value}
            Sévérité: {alert.severity.value.upper()}
            Timestamp: {alert.timestamp}
            
            Source IP: {alert.source_ip}
            Destination IP: {alert.destination_ip}
            
            Description: {alert.description}
            Score de confiance: {alert.confidence_score*100:.2f}%
            
            Action requise: Veuillez investiguer immédiatement.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.get('smtp_server'), self.config.get('smtp_port', 587))
            server.starttls()
            server.login(self.config.get('smtp_username'), self.config.get('smtp_password'))
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification envoyée pour {alert.alert_id}")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi d'email: {e}")
    
    def _send_to_siem(self, alert):
        """Envoie l'alerte vers un SIEM (ELK Stack)."""
        try:
            siem_url = self.config.get('siem_url')
            if siem_url:
                response = requests.post(
                    siem_url,
                    json=alert.to_dict(),
                    headers={'Content-Type': 'application/json'}
                )
                if response.status_code == 200:
                    self.logger.info(f"Alerte {alert.alert_id} envoyée au SIEM")
                else:
                    self.logger.error(f"Erreur SIEM: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi au SIEM: {e}")
    
    def _send_webhook(self, alert):
        """Envoie l'alerte via webhook."""
        try:
            webhook_url = self.config.get('webhook_url')
            if webhook_url:
                response = requests.post(
                    webhook_url,
                    json=alert.to_dict(),
                    headers={'Content-Type': 'application/json'}
                )
                if response.status_code == 200:
                    self.logger.info(f"Webhook envoyé pour {alert.alert_id}")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi du webhook: {e}")
    
    def get_alerts(self, severity=None, status=None, limit=100):
        """
        Récupère les alertes filtrées.
        
        Args:
            severity: Filtrer par sévérité
            status: Filtrer par statut
            limit: Nombre maximum d'alertes à retourner
        
        Returns:
            list: Liste des alertes
        """
        filtered_alerts = self.alerts
        
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
        
        if status:
            filtered_alerts = [a for a in filtered_alerts if a.status == status]
        
        return filtered_alerts[-limit:]
    
    def update_alert_status(self, alert_id, new_status):
        """Met à jour le statut d'une alerte."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.status = new_status
                self.logger.info(f"Alerte {alert_id} mise à jour: {new_status}")
                return True
        return False
    
    def get_statistics(self):
        """Retourne des statistiques sur les alertes."""
        total = len(self.alerts)
        
        by_severity = {}
        for severity in AlertSeverity:
            by_severity[severity.value] = len([a for a in self.alerts if a.severity == severity])
        
        by_type = {}
        for alert_type in AlertType:
            by_type[alert_type.value] = len([a for a in self.alerts if a.alert_type == alert_type])
        
        return {
            'total_alerts': total,
            'by_severity': by_severity,
            'by_type': by_type,
            'critical_active': len([a for a in self.alerts if a.severity == AlertSeverity.CRITICAL and a.status == 'new'])
        }


# Configuration d'exemple
def create_alert_manager():
    """Crée un gestionnaire d'alertes avec configuration."""
    config = {
        'email_enabled': False,  # À activer en production
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'smtp_from': 'ids@example.com',
        'smtp_to': 'soc@example.com',
        'smtp_username': 'your_email',
        'smtp_password': 'your_password',
        'siem_enabled': True,  # À activer si ELK est configuré
        'siem_url': 'http://localhost:9200/ids-alerts/_doc',
        'webhook_url': None
    }
    
    return AlertManager(config)