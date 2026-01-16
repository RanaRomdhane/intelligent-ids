"""
Module de système d'alertes pour IDS - Version Finale.
"""

import json
import logging
import os
from datetime import datetime
from enum import Enum

os.makedirs('logs', exist_ok=True)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    DOS = "DoS/DDoS"
    PROBE = "Port Scan/Probe"
    R2L = "Remote to Local"
    U2R = "User to Root"
    BOTNET = "Botnet Activity"
    BRUTE_FORCE = "Brute Force"
    EXFILTRATION = "Data Exfiltration"
    INJECTION = "Injection Attack"
    EXPLOIT = "Exploit"
    BACKDOOR = "Backdoor"
    SHELLCODE = "Shellcode"
    WORMS = "Worms"
    GENERIC = "Generic Attack"
    FUZZERS = "Fuzzers"
    RECONNAISSANCE = "Reconnaissance"
    ANALYSIS = "Analysis"
    UNKNOWN = "Unknown Attack"


class Alert:
    def __init__(self, alert_type, severity, source_ip, destination_ip, 
                 description, confidence_score, attack_category=None, timestamp=None):
        self.alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.alert_type = alert_type
        self.attack_category = attack_category or str(alert_type.value if isinstance(alert_type, AlertType) else alert_type)
        self.severity = severity
        self.source_ip = source_ip
        self.destination_ip = destination_ip
        self.description = description
        self.confidence_score = confidence_score
        self.timestamp = timestamp or datetime.now()
        self.status = "new"
        
    def to_dict(self):
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value if isinstance(self.alert_type, AlertType) else str(self.alert_type),
            'attack_category': self.attack_category,
            'severity': self.severity.value if isinstance(self.severity, AlertSeverity) else str(self.severity),
            'source_ip': str(self.source_ip),
            'destination_ip': str(self.destination_ip),
            'description': str(self.description),
            'confidence_score': float(self.confidence_score),
            'timestamp': self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp),
            'status': self.status
        }


class AlertManager:
    def __init__(self, config=None):
        self.config = config or {}
        self.alerts = []
        self.logger = logging.getLogger(__name__)
        self.elk_integration = None
        self._init_elk()
        
    def _init_elk(self):
        """Initialise la connexion ELK une seule fois."""
        try:
            from src.elk_integration import ELKIntegration
            self.elk_integration = ELKIntegration()
            if self.elk_integration.connected:
                self.logger.info("✓ AlertManager connecté à ELK")
        except Exception as e:
            self.logger.debug(f"ELK non disponible: {e}")
            self.elk_integration = None
    
    def create_alert(self, prediction, flow_data, confidence_score):
        """Crée une alerte basée sur une prédiction."""
        
        alert_type_map = {
            'dos': AlertType.DOS, 'ddos': AlertType.DOS,
            'probe': AlertType.PROBE, 'scan': AlertType.PROBE,
            'r2l': AlertType.R2L, 'u2r': AlertType.U2R,
            'botnet': AlertType.BOTNET, 'brute_force': AlertType.BRUTE_FORCE,
            'exfiltration': AlertType.EXFILTRATION, 'injection': AlertType.INJECTION,
            'exploit': AlertType.EXPLOIT, 'exploits': AlertType.EXPLOIT,
            'backdoor': AlertType.BACKDOOR, 'backdoors': AlertType.BACKDOOR,
            'shellcode': AlertType.SHELLCODE, 'worms': AlertType.WORMS,
            'generic': AlertType.GENERIC, 'fuzzers': AlertType.FUZZERS,
            'reconnaissance': AlertType.RECONNAISSANCE, 'analysis': AlertType.ANALYSIS
        }
        
        pred_lower = str(prediction).lower().strip()
        alert_type = AlertType.UNKNOWN
        
        if pred_lower in alert_type_map:
            alert_type = alert_type_map[pred_lower]
        else:
            for key, val in alert_type_map.items():
                if key in pred_lower or pred_lower in key:
                    alert_type = val
                    break
        
        severity = self._determine_severity(alert_type, confidence_score)
        
        source_ip = flow_data.get('src_ip', flow_data.get('srcip', 'Unknown'))
        dest_ip = flow_data.get('dst_ip', flow_data.get('dstip', 'Unknown'))
        
        description = f"{prediction} détectée avec {confidence_score*100:.1f}% de confiance"
        
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            source_ip=source_ip,
            destination_ip=dest_ip,
            description=description,
            confidence_score=confidence_score,
            attack_category=str(prediction)
        )
        
        self.alerts.append(alert)
        
        # Envoyer à ELK si disponible
        if self.elk_integration and self.elk_integration.connected:
            self.elk_integration.index_alert(alert)
        
        return alert
    
    def _determine_severity(self, alert_type, confidence_score):
        critical_types = [AlertType.DOS, AlertType.U2R, AlertType.EXFILTRATION, AlertType.EXPLOIT, AlertType.BACKDOOR]
        high_types = [AlertType.BRUTE_FORCE, AlertType.BOTNET, AlertType.INJECTION, AlertType.SHELLCODE, AlertType.WORMS]
        
        if alert_type in critical_types and confidence_score > 0.85:
            return AlertSeverity.CRITICAL
        elif alert_type in critical_types or (alert_type in high_types and confidence_score > 0.8):
            return AlertSeverity.HIGH
        elif confidence_score > 0.7:
            return AlertSeverity.MEDIUM
        return AlertSeverity.LOW
    
    def get_alerts(self, severity=None, status=None, limit=100):
        filtered = self.alerts
        if severity:
            if isinstance(severity, str):
                filtered = [a for a in filtered if a.severity.value == severity]
            else:
                filtered = [a for a in filtered if a.severity == severity]
        if status:
            filtered = [a for a in filtered if a.status == status]
        return filtered[-limit:]
    
    def update_alert_status(self, alert_id, new_status):
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.status = new_status
                return True
        return False
    
    def get_statistics(self):
        total = len(self.alerts)
        by_severity = {s.value: len([a for a in self.alerts if a.severity == s]) for s in AlertSeverity}
        by_type = {}
        for a in self.alerts:
            cat = a.attack_category
            by_type[cat] = by_type.get(cat, 0) + 1
        
        return {
            'total_alerts': total,
            'by_severity': by_severity,
            'by_type': by_type,
            'elk_connected': self.elk_integration.connected if self.elk_integration else False
        }


def create_alert_manager():
    """Crée un gestionnaire d'alertes."""
    return AlertManager()