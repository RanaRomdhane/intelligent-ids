// Demo Page JavaScript - Version Corrigée

document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const resultsContent = document.getElementById('resultsContent');
    const modelSelect = document.getElementById('modelSelect');
    let selectedFile = null;

    uploadArea.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', function(e) {
        handleFile(e.target.files[0]);
    });

    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function() {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleFile(e.dataTransfer.files[0]);
    });

    function handleFile(file) {
        if (file && (file.type === 'text/csv' || file.name.endsWith('.csv'))) {
            selectedFile = file;
            uploadArea.innerHTML = `
                <i class="fas fa-check-circle" style="font-size: 48px; color: var(--success-color); margin-bottom: 15px;"></i>
                <p><strong>${file.name}</strong></p>
                <p style="color: var(--text-light); font-size: 14px;">${(file.size / 1024).toFixed(2)} KB</p>
                <button class="btn btn-secondary" style="margin-top: 10px;" onclick="event.stopPropagation(); location.reload();">Change File</button>
            `;
            analyzeBtn.disabled = false;
        } else {
            showNotification('Please upload a CSV file.', 'error');
        }
    }

    analyzeBtn.addEventListener('click', async function() {
        if (!selectedFile) {
            showNotification('Please select a file first.', 'error');
            return;
        }

        loading.classList.add('show');
        results.classList.remove('show');
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('model', modelSelect.value);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                console.log('API Response:', data); // Debug
                displayResults(data);
            } else {
                const errorData = await response.json();
                showNotification(errorData.error || 'Analysis failed', 'error');
            }
        } catch (error) {
            console.error('Error:', error);
            showNotification('Server connection error', 'error');
        } finally {
            loading.classList.remove('show');
            analyzeBtn.disabled = false;
        }
    });

    function displayResults(data) {
        const predictions = data.predictions || {};
        
        console.log('Predictions received:', predictions); // Debug
        
        // Identifier le trafic normal (plusieurs variantes possibles)
        let normalCount = 0;
        const normalKeys = ['Normal', 'normal', 'NORMAL', 'Benign', 'benign', 'BENIGN'];
        
        for (const key of normalKeys) {
            if (predictions[key]) {
                normalCount += predictions[key];
            }
        }
        
        // Filtrer les attaques (tout ce qui n'est pas normal)
        const attacks = Object.entries(predictions)
            .filter(([key, value]) => {
                const keyLower = key.toLowerCase().trim();
                // Exclure normal, benign, et les clés numériques qui pourraient être "0"
                const isNormal = keyLower === 'normal' || keyLower === 'benign' || key === '0';
                return !isNormal && value > 0;
            })
            .map(([key, value]) => {
                // Formater le nom de la catégorie
                return [formatCategoryName(key), value];
            });

        const totalAttacks = attacks.reduce((sum, [_, count]) => sum + count, 0);

        let html = `
            <div class="result-card">
                <h4><i class="fas fa-chart-pie"></i> Analysis Summary</h4>
                <p><strong>Total Samples:</strong> ${data.total_samples || 0}</p>
                <p><strong>Model Used:</strong> ${formatModelName(data.model)}</p>
                ${data.accuracy !== null && data.accuracy !== undefined ? 
                    `<p><strong>Accuracy:</strong> ${data.accuracy.toFixed(2)}%</p>` : ''}
                <p><strong>Normal Traffic:</strong> <span style="color: #27ae60; font-weight: bold;">${normalCount}</span> samples</p>
                <p><strong>Attack Traffic:</strong> <span style="color: #e74c3c; font-weight: bold;">${totalAttacks}</span> samples</p>
            </div>
        `;

        if (attacks.length > 0) {
            html += `
                <div class="result-card">
                    <h4><i class="fas fa-exclamation-triangle" style="color: #e74c3c;"></i> Attack Categories Detected</h4>
                    <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                        <thead>
                            <tr style="background: #f8f9fa;">
                                <th style="padding: 12px; text-align: left; border-bottom: 2px solid #dee2e6;">Category</th>
                                <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">Count</th>
                                <th style="padding: 12px; text-align: center; border-bottom: 2px solid #dee2e6;">Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${attacks.sort((a, b) => b[1] - a[1]).map(([type, count]) => {
                                const percentage = ((count / data.total_samples) * 100).toFixed(1);
                                const severityClass = getSeverityClass(type);
                                return `
                                    <tr>
                                        <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                                            <span class="prediction-badge ${severityClass}">${type}</span>
                                        </td>
                                        <td style="padding: 12px; text-align: center; border-bottom: 1px solid #dee2e6;">
                                            <strong>${count}</strong>
                                        </td>
                                        <td style="padding: 12px; text-align: center; border-bottom: 1px solid #dee2e6;">
                                            ${percentage}%
                                        </td>
                                    </tr>
                                `;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            `;

            // Section des alertes
            if (data.alerts_count > 0) {
                html += `
                    <div class="result-card">
                        <h4><i class="fas fa-bell" style="color: #f39c12;"></i> Alerts Generated: ${data.alerts_count}</h4>
                        <p>Security alerts have been created for detected attacks.</p>
                        ${data.alerts && data.alerts.length > 0 ? `
                            <details style="margin-top: 10px;">
                                <summary style="cursor: pointer; color: #3498db;">View recent alerts</summary>
                                <ul style="margin-top: 10px; padding-left: 20px;">
                                    ${data.alerts.slice(0, 5).map(alert => `
                                        <li style="margin: 5px 0;">
                                            <strong>${alert.alert_type || 'Unknown'}</strong> - 
                                            Severity: ${alert.severity || 'N/A'} - 
                                            Confidence: ${((alert.confidence_score || 0) * 100).toFixed(1)}%
                                        </li>
                                    `).join('')}
                                </ul>
                            </details>
                        ` : ''}
                    </div>
                `;
            }
        } else {
            html += `
                <div class="result-card" style="background: #d4edda; border: 1px solid #c3e6cb;">
                    <h4><i class="fas fa-check-circle" style="color: #28a745;"></i> No Attacks Detected</h4>
                    <p style="color: #155724;">All ${data.total_samples} samples appear to be normal traffic. Your network looks secure!</p>
                </div>
            `;
        }

        resultsContent.innerHTML = html;
        results.classList.add('show');
        
        // Scroll vers les résultats
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function formatCategoryName(name) {
        // Formater les noms de catégories pour l'affichage
        const categoryMap = {
            'dos': 'DoS Attack',
            'ddos': 'DDoS Attack',
            'probe': 'Probe/Scan',
            'r2l': 'Remote to Local',
            'u2r': 'User to Root',
            'analysis': 'Analysis',
            'backdoor': 'Backdoor',
            'backdoors': 'Backdoor',
            'exploits': 'Exploits',
            'fuzzers': 'Fuzzers',
            'generic': 'Generic Attack',
            'reconnaissance': 'Reconnaissance',
            'shellcode': 'Shellcode',
            'worms': 'Worms',
            'normal': 'Normal',
            'benign': 'Benign'
        };
        
        const lowerName = name.toLowerCase().trim();
        
        // Vérifier le mapping exact
        if (categoryMap[lowerName]) {
            return categoryMap[lowerName];
        }
        
        // Vérifier les correspondances partielles
        for (const [key, value] of Object.entries(categoryMap)) {
            if (lowerName.includes(key)) {
                return value;
            }
        }
        
        // Si pas de correspondance, capitaliser la première lettre
        return name.charAt(0).toUpperCase() + name.slice(1);
    }

    function formatModelName(model) {
        const modelNames = {
            'random_forest': 'Random Forest',
            'svm': 'Support Vector Machine (SVM)',
            'neural_network': 'Neural Network'
        };
        return modelNames[model] || model;
    }

    function getSeverityClass(attackType) {
        const type = attackType.toLowerCase();
        
        // Attaques critiques
        if (type.includes('dos') || type.includes('ddos') || type.includes('exploit')) {
            return 'prediction-critical';
        }
        
        // Attaques élevées
        if (type.includes('backdoor') || type.includes('shellcode') || type.includes('worm')) {
            return 'prediction-high';
        }
        
        // Attaques moyennes
        if (type.includes('probe') || type.includes('reconnaissance') || 
            type.includes('analysis') || type.includes('fuzzer')) {
            return 'prediction-medium';
        }
        
        // Par défaut
        return 'prediction-attack';
    }

    function showNotification(message, type = 'success') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#27ae60' : '#e74c3c'};
            color: white;
            padding: 15px 20px;
            border-radius: 5px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
            z-index: 10000;
            animation: slideIn 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
});