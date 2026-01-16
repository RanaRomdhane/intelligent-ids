"""
Flask web application for IDS ML project - Version Finale Corrigée.
Provides web interface and API endpoints for intrusion detection with real-time monitoring.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import sys
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import threading
import time
from datetime import datetime
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import DataPreprocessor
from src.models import RandomForestIDS, SVMIDS, NeuralNetworkIDS
from src.evaluation import ModelEvaluator
from src.alert_system import AlertManager, create_alert_manager
from src.elk_integration import ELKIntegration, setup_elk_integration

# Setup logging with UTF-8 encoding
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        if stream is None:
            stream = sys.stderr
        self.stream = stream
        
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Encode to UTF-8 for Windows console
            if hasattr(stream, 'buffer'):
                stream.buffer.write((msg + self.terminator).encode('utf-8'))
                stream.flush()
            else:
                stream.write(msg + self.terminator)
                stream.flush()
        except Exception:
            self.handleError(record)

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler with UTF-8 encoding
file_handler = logging.FileHandler('logs/app.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler with UTF-8 support
console_handler = UTF8StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

app = Flask(__name__, static_folder='web', static_url_path='')
app.config['SECRET_KEY'] = 'ids-ml-secret-key-2025'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=100*1024*1024)

# Configuration - CRITICAL: Augmenter TOUTES les limites
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# IMPORTANT: Configurer les limites Werkzeug
import werkzeug
werkzeug.utils.MAX_CONTENT_LENGTH = 100 * 1024 * 1024

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global variables
preprocessor = None
models = {}
alert_manager = None
elk_integration = None
monitoring_active = False
monitoring_thread = None
stats_cache = {
    'total_packets': 0,
    'packets_per_second': 0,
    'attacks_detected': 0,
    'critical_alerts': 0,
    'protocols': {'tcp': 0, 'udp': 0, 'other': 0},
    'start_time': None
}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_models():
    """Load trained models if they exist."""
    global preprocessor, models, alert_manager, elk_integration
    
    logger.info("Loading models and initializing systems...")
    
    # Load preprocessor
    preprocessor_path = 'models/preprocessor.pkl'
    if os.path.exists(preprocessor_path):
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessor(preprocessor_path)
        logger.info("Preprocessor loaded")
    else:
        logger.warning("Preprocessor not found - run train.py first")
    
    # Load ML models
    model_paths = {
        'random_forest': 'models/random_forest_model.pkl',
        'svm': 'models/svm_model.pkl',
        'neural_network': 'models/neural_network_model.h5'
    }
    
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            try:
                if model_name == 'random_forest':
                    model = RandomForestIDS()
                    model.load(model_path)
                    models[model_name] = model
                elif model_name == 'svm':
                    model = SVMIDS()
                    model.load(model_path)
                    models[model_name] = model
                elif model_name == 'neural_network':
                    model = NeuralNetworkIDS()
                    model.load(model_path)
                    models[model_name] = model
                logger.info(f"{model_name} model loaded")
            except Exception as e:
                logger.error(f"Error loading {model_name}: {e}")
    
    # Initialize Alert Manager
    alert_manager = create_alert_manager()
    logger.info("Alert Manager initialized")
    
    # Initialize ELK Integration (optional)
    try:
        elk_integration = setup_elk_integration()
        if elk_integration and elk_integration.es:
            logger.info("ELK Stack connected")
        else:
            logger.warning("ELK Stack not available - running without SIEM")
    except Exception as e:
        logger.warning(f"ELK Stack initialization failed: {e}")
        elk_integration = None


def simulate_realtime_monitoring():
    """Simulate real-time monitoring for demo purposes."""
    global monitoring_active, stats_cache
    
    logger.info("Real-time monitoring started")
    
    while monitoring_active:
        try:
            # Simulate packet capture and analysis
            packets_this_cycle = np.random.randint(30, 100)
            stats_cache['total_packets'] += packets_this_cycle
            stats_cache['packets_per_second'] = packets_this_cycle / 2
            
            # Update protocols
            stats_cache['protocols']['tcp'] += np.random.randint(20, 60)
            stats_cache['protocols']['udp'] += np.random.randint(10, 30)
            stats_cache['protocols']['other'] += np.random.randint(0, 10)
            
            # Simulate attack detection (10% probability)
            if np.random.random() < 0.1:
                attack_type = np.random.choice(['DoS', 'Port Scan', 'Brute Force', 'SQL Injection'])
                severity = np.random.choice(['critical', 'high', 'medium', 'low'])
                
                # Create mock alert
                alert_data = {
                    'type': attack_type,
                    'severity': severity,
                    'source_ip': f'192.168.1.{np.random.randint(1, 255)}',
                    'destination_ip': f'10.0.0.{np.random.randint(1, 255)}',
                    'confidence': np.random.uniform(0.7, 0.99),
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }
                
                stats_cache['attacks_detected'] += 1
                if severity == 'critical':
                    stats_cache['critical_alerts'] += 1
                
                # Emit alert via WebSocket
                socketio.emit('new_alert', alert_data)
                
                logger.info(f"Alert: {attack_type} ({severity}) from {alert_data['source_ip']}")
                
                # Index in ELK if available
                if elk_integration and elk_integration.es:
                    try:
                        elk_integration.index_alert({
                            'alert_id': f"ALERT-{int(time.time())}",
                            'timestamp': datetime.now().isoformat(),
                            'alert_type': attack_type,
                            'severity': severity,
                            'source_ip': alert_data['source_ip'],
                            'destination_ip': alert_data['destination_ip'],
                            'confidence_score': alert_data['confidence'],
                            'description': f"{attack_type} detected",
                            'status': 'new'
                        })
                    except Exception as e:
                        logger.error(f"ELK indexation error: {e}")
            
            # Emit stats update
            socketio.emit('stats_update', {
                'total_packets': stats_cache['total_packets'],
                'packets_per_second': round(stats_cache['packets_per_second'], 2),
                'attacks_detected': stats_cache['attacks_detected'],
                'critical_alerts': stats_cache['critical_alerts'],
                'protocols': stats_cache['protocols']
            })
            
            time.sleep(2)  # Update every 2 seconds
            
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            time.sleep(2)
    
    logger.info("Real-time monitoring stopped")


# Load models on startup
load_models()


# ============================================================================
# ROUTES - Web Pages
# ============================================================================

@app.route('/')
def index():
    """Home page."""
    return app.send_static_file('index.html')


@app.route('/dashboard')
def dashboard():
    """Real-time dashboard."""
    return app.send_static_file('dashboard.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    try:
        return app.send_static_file(filename)
    except:
        if not filename.endswith('.html'):
            return app.send_static_file(filename + '.html')
        return "File not found", 404


# ============================================================================
# API ENDPOINTS - Prediction
# ============================================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions."""
    try:
        # Vérifier la taille du fichier
        if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({
                'error': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] / (1024*1024):.0f}MB'
            }), 413

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        model_name = request.form.get('model', 'random_forest')

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        logger.info(f"Processing file: {filename} with model: {model_name}")

        if preprocessor is None:
            os.remove(filepath)
            return jsonify({'error': 'Preprocessor not loaded. Please train models first.'}), 500

        df = preprocessor.load_data(filepath)
        if df is None:
            os.remove(filepath)
            return jsonify({'error': 'Could not load data file'}), 400

        # Garder juste quelques colonnes utiles pour les alertes (évite un énorme .copy())
        cols_for_flow = []
        for c in ['srcip', 'dstip', 'proto', 'src_ip', 'dst_ip', 'protocol']:
            if c in df.columns:
                cols_for_flow.append(c)
        original_df = df[cols_for_flow].copy() if cols_for_flow else df.head(0).copy()

        df_cleaned = preprocessor.clean_data(df)
        df_features = preprocessor.extract_features(df_cleaned)

        # Assurer la compatibilité des features
        df_features = preprocessor.ensure_feature_compatibility(df_features)
        X = df_features[preprocessor.feature_columns].copy()

        # Scale features
        X_scaled = preprocessor.scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=preprocessor.feature_columns)

        # Get model
        if model_name not in models:
            os.remove(filepath)
            return jsonify({'error': f'Model {model_name} not available.'}), 404

        model = models[model_name]

        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        # ========== Décodage des labels ==========
        if hasattr(preprocessor.label_encoder, 'classes_') and len(preprocessor.label_encoder.classes_) > 0:
            class_names = preprocessor.label_encoder.classes_
            logger.info(f"Classes disponibles: {list(class_names)}")

            predicted_labels = []
            for pred in predictions:
                # pred peut être numpy int
                pred_int = int(pred) if isinstance(pred, (np.integer, int)) else pred
                if isinstance(pred_int, int) and 0 <= pred_int < len(class_names):
                    predicted_labels.append(str(class_names[pred_int]))
                else:
                    predicted_labels.append(str(pred))
        else:
            predicted_labels = [str(p) for p in predictions]

        # Compter les prédictions
        prediction_counts = {}
        for label in predicted_labels:
            prediction_counts[label] = prediction_counts.get(label, 0) + 1

        logger.info(f"Prediction counts: {prediction_counts}")
        # =======================================

        # ========== IMPORTANT : limiter la génération d'alertes ==========
        MAX_ALERTS_TO_GENERATE = 50          # pour éviter blocage
        MAX_ALERTS_RETURNED = 10             # pour JSON
        ALERT_CONFIDENCE_THRESHOLD = 0.80    # optionnel (évite les alertes faibles)

        alerts_generated = []

        # indices des attaques
        attack_indices = []
        for i, label in enumerate(predicted_labels):
            lbl = str(label).lower().strip()
            if lbl not in ['normal', 'benign']:
                attack_indices.append(i)

        # Générer des alertes seulement pour un échantillon limité
        for i in attack_indices:
            if len(alerts_generated) >= MAX_ALERTS_TO_GENERATE:
                break

            # confiance
            if probabilities is not None:
                conf = float(np.max(probabilities[i]))
            else:
                conf = 1.0

            if conf < ALERT_CONFIDENCE_THRESHOLD:
                continue

            if alert_manager:
                flow_data = {}
                if i < len(original_df) and not original_df.empty:
                    row = original_df.iloc[i]
                    flow_data['src_ip'] = str(row.get('srcip', row.get('src_ip', 'Unknown')))
                    flow_data['dst_ip'] = str(row.get('dstip', row.get('dst_ip', 'Unknown')))
                    flow_data['protocol'] = str(row.get('proto', row.get('protocol', 'Unknown')))

                alert = alert_manager.create_alert(
                    prediction=str(predicted_labels[i]),
                    flow_data=flow_data,
                    confidence_score=conf
                )
                alerts_generated.append(alert.to_dict())
        # ===============================================================

        # Accuracy (si labels dispo)
        accuracy = None
        target_cols = ['attack_cat', 'Attack_Cat', 'label', 'Label', 'class', 'Class']
        target_col = next((c for c in target_cols if c in df.columns), None)

        if target_col:
            try:
                y_true = df[target_col].values
                if hasattr(preprocessor.label_encoder, 'classes_'):
                    y_true_encoded = preprocessor.label_encoder.transform(y_true)
                    accuracy = float(np.mean(predictions == y_true_encoded) * 100)
            except Exception as e:
                logger.warning(f"Could not calculate accuracy: {e}")

        # Clean up uploaded file
        os.remove(filepath)

        logger.info(f"Analysis complete: {len(predictions)} samples, {len(alerts_generated)} alerts generated")

        return jsonify({
            'success': True,
            'total_samples': int(len(predictions)),
            'predictions': {str(k): int(v) for k, v in prediction_counts.items()},
            'accuracy': accuracy,
            'model': model_name,
            'alerts_count': int(len(alerts_generated)),
            'alerts': alerts_generated[:MAX_ALERTS_RETURNED]
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - System Management
# ============================================================================

@app.route('/api/models', methods=['GET'])
def list_models():
    """List available models."""
    available_models = list(models.keys())
    return jsonify({
        'available_models': available_models,
        'total': len(available_models)
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    stats = {
        'models_loaded': len(models),
        'available_models': list(models.keys()),
        'preprocessor_loaded': preprocessor is not None,
        'alert_manager_active': alert_manager is not None,
        'elk_connected': elk_integration is not None and elk_integration.es is not None,
        'monitoring_active': monitoring_active
    }
    
    if alert_manager:
        stats['alert_statistics'] = alert_manager.get_statistics()
    
    return jsonify(stats)


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts."""
    if not alert_manager:
        return jsonify({'error': 'Alert manager not initialized'}), 500
    
    limit = request.args.get('limit', 50, type=int)
    severity = request.args.get('severity', None)
    
    alerts = alert_manager.get_alerts(severity=severity, limit=limit)
    
    return jsonify({
        'alerts': [alert.to_dict() for alert in alerts],
        'total': len(alerts)
    })


@app.route('/api/alerts/<alert_id>/status', methods=['PUT'])
def update_alert_status(alert_id):
    """Update alert status."""
    if not alert_manager:
        return jsonify({'error': 'Alert manager not initialized'}), 500
    
    data = request.json
    new_status = data.get('status')
    
    if not new_status:
        return jsonify({'error': 'Status is required'}), 400
    
    success = alert_manager.update_alert_status(alert_id, new_status)
    
    if success:
        return jsonify({'success': True, 'message': f'Alert {alert_id} updated'})
    else:
        return jsonify({'error': 'Alert not found'}), 404


# ============================================================================
# API ENDPOINTS - Real-time Monitoring
# ============================================================================

@app.route('/api/monitoring/start', methods=['POST'])
def start_monitoring():
    """Start real-time monitoring."""
    global monitoring_active, monitoring_thread, stats_cache
    
    if monitoring_active:
        return jsonify({'error': 'Monitoring already active'}), 400
    
    # Reset stats
    stats_cache = {
        'total_packets': 0,
        'packets_per_second': 0,
        'attacks_detected': 0,
        'critical_alerts': 0,
        'protocols': {'tcp': 0, 'udp': 0, 'other': 0},
        'start_time': datetime.now()
    }
    
    monitoring_active = True
    monitoring_thread = threading.Thread(target=simulate_realtime_monitoring, daemon=True)
    monitoring_thread.start()
    
    logger.info("Monitoring started via API")
    
    return jsonify({
        'success': True,
        'message': 'Real-time monitoring started',
        'start_time': stats_cache['start_time'].isoformat()
    })


@app.route('/api/monitoring/stop', methods=['POST'])
def stop_monitoring():
    """Stop real-time monitoring."""
    global monitoring_active
    
    if not monitoring_active:
        return jsonify({'error': 'Monitoring not active'}), 400
    
    monitoring_active = False
    logger.info("Monitoring stopped via API")
    
    return jsonify({
        'success': True,
        'message': 'Real-time monitoring stopped'
    })


@app.route('/api/monitoring/stats', methods=['GET'])
def get_monitoring_stats():
    """Get current monitoring statistics."""
    if not monitoring_active:
        return jsonify({'error': 'Monitoring not active'}), 400
    
    # Calculate uptime
    uptime_seconds = 0
    if stats_cache['start_time']:
        uptime_seconds = (datetime.now() - stats_cache['start_time']).total_seconds()
    
    return jsonify({
        'stats': stats_cache,
        'uptime_seconds': uptime_seconds,
        'monitoring_active': monitoring_active
    })


# ============================================================================
# WebSocket Events
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    emit('connection_status', {'status': 'connected', 'monitoring_active': monitoring_active})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('request_stats')
def handle_stats_request():
    """Handle stats request from client."""
    if monitoring_active:
        emit('stats_update', {
            'total_packets': stats_cache['total_packets'],
            'packets_per_second': round(stats_cache['packets_per_second'], 2),
            'attacks_detected': stats_cache['attacks_detected'],
            'critical_alerts': stats_cache['critical_alerts'],
            'protocols': stats_cache['protocols']
        })


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("IDS ML - Intelligent Intrusion Detection System")
    print("="*70)
    print(f"\nFlask Server: http://localhost:5000")
    print(f"Dashboard: http://localhost:5000/dashboard")
    print(f"\nSystem Status:")
    print(f"  - Models loaded: {len(models)}")
    print(f"  - Available models: {', '.join(models.keys()) if models else 'None'}")
    print(f"  - Preprocessor: {'Ready' if preprocessor else 'Not loaded'}")
    print(f"  - Alert Manager: {'Active' if alert_manager else 'Not initialized'}")
    print(f"  - ELK Stack: {'Connected' if (elk_integration and elk_integration.es) else 'Not available'}")
    print(f"  - Max upload size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")
    print(f"\nStarting server...\n")
    print("="*70)
    
    # Run with SocketIO
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)