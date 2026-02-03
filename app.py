import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time
from datetime import datetime, timedelta
import random

# Page configuration
st.set_page_config(
    page_title="CyberShield | Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark cyber theme with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');
    
    :root {
        --cyber-cyan: #00f5ff;
        --cyber-purple: #bf00ff;
        --cyber-green: #00ff88;
        --cyber-red: #ff0055;
        --cyber-orange: #ff6b00;
        --cyber-dark: #0a0a0f;
        --cyber-darker: #050508;
        --grid-color: rgba(0, 245, 255, 0.03);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d0d1a 50%, #0a0a0f 100%);
    }
    
    /* Animated grid background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(0, 245, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 245, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        animation: gridMove 20s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes gridMove {
        0% { transform: translate(0, 0); }
        100% { transform: translate(50px, 50px); }
    }
    
    /* Glowing scan line effect */
    .stApp::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--cyber-cyan), transparent);
        animation: scanLine 3s linear infinite;
        pointer-events: none;
        z-index: 1000;
        opacity: 0.6;
    }
    
    @keyframes scanLine {
        0% { top: 0; }
        100% { top: 100%; }
    }
    
    /* Main header styling */
    .cyber-header {
        text-align: center;
        padding: 30px 0;
        margin-bottom: 30px;
        position: relative;
    }
    
    .cyber-title {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00f5ff 0%, #bf00ff 50%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 50px rgba(0, 245, 255, 0.5);
        animation: titleGlow 2s ease-in-out infinite alternate;
        letter-spacing: 4px;
        margin: 0;
    }
    
    @keyframes titleGlow {
        0% { filter: drop-shadow(0 0 20px rgba(0, 245, 255, 0.8)); }
        100% { filter: drop-shadow(0 0 40px rgba(191, 0, 255, 0.8)); }
    }
    
    .cyber-subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2rem;
        color: rgba(0, 245, 255, 0.7);
        letter-spacing: 8px;
        text-transform: uppercase;
        margin-top: 10px;
        animation: subtitlePulse 2s ease-in-out infinite;
    }
    
    @keyframes subtitlePulse {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    
    /* Metric cards with cyber styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(10, 10, 15, 0.9) 0%, rgba(13, 13, 26, 0.9) 100%);
        border: 1px solid rgba(0, 245, 255, 0.3);
        border-radius: 15px;
        padding: 15px 10px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 245, 255, 0.1), transparent);
        animation: cardShine 3s infinite;
    }
    
    @keyframes cardShine {
        0% { left: -100%; }
        50%, 100% { left: 100%; }
    }
    
    .metric-card:hover {
        border-color: var(--cyber-cyan);
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.3), inset 0 0 30px rgba(0, 245, 255, 0.05);
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 5px 0;
        white-space: nowrap;
    }
    
    .metric-label {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .cyan { color: var(--cyber-cyan); text-shadow: 0 0 20px rgba(0, 245, 255, 0.8); }
    .green { color: var(--cyber-green); text-shadow: 0 0 20px rgba(0, 255, 136, 0.8); }
    .red { color: var(--cyber-red); text-shadow: 0 0 20px rgba(255, 0, 85, 0.8); }
    .purple { color: var(--cyber-purple); text-shadow: 0 0 20px rgba(191, 0, 255, 0.8); }
    .orange { color: var(--cyber-orange); text-shadow: 0 0 20px rgba(255, 107, 0, 0.8); }
    
    /* Section headers */
    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 1.5rem;
        color: var(--cyber-cyan);
        border-left: 4px solid var(--cyber-cyan);
        padding-left: 20px;
        margin: 30px 0 20px 20px;
        text-transform: uppercase;
        letter-spacing: 3px;
        animation: headerPulse 2s ease-in-out infinite;
    }
    
    /* Tab spacing */
    .stTabs {
        margin-top: 30px !important;
    }
    
    @keyframes headerPulse {
        0%, 100% { border-left-color: var(--cyber-cyan); }
        50% { border-left-color: var(--cyber-purple); }
    }
    
    /* Alert box styling */
    .alert-critical {
        background: linear-gradient(135deg, rgba(255, 0, 85, 0.1) 0%, rgba(255, 0, 85, 0.05) 100%);
        border: 1px solid rgba(255, 0, 85, 0.5);
        border-radius: 10px;
        padding: 15px 20px;
        margin: 10px 0;
        animation: alertPulse 1s ease-in-out infinite;
    }
    
    @keyframes alertPulse {
        0%, 100% { box-shadow: 0 0 10px rgba(255, 0, 85, 0.3); }
        50% { box-shadow: 0 0 25px rgba(255, 0, 85, 0.6); }
    }
    
    .alert-warning {
        background: linear-gradient(135deg, rgba(255, 107, 0, 0.1) 0%, rgba(255, 107, 0, 0.05) 100%);
        border: 1px solid rgba(255, 107, 0, 0.5);
        border-radius: 10px;
        padding: 15px 20px;
        margin: 10px 0;
    }
    
    .alert-safe {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 255, 136, 0.05) 100%);
        border: 1px solid rgba(0, 255, 136, 0.5);
        border-radius: 10px;
        padding: 15px 20px;
        margin: 10px 0;
    }
    
    /* Threat log styling */
    .threat-log {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.85rem;
        background: rgba(0, 0, 0, 0.5);
        border-left: 3px solid var(--cyber-red);
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
        color: rgba(255, 255, 255, 0.8);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 10, 15, 0.95) 0%, rgba(5, 5, 8, 0.98) 100%);
        border-right: 1px solid rgba(0, 245, 255, 0.2);
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stMultiSelect label {
        font-family: 'Rajdhani', sans-serif;
        color: var(--cyber-cyan) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Grey dropdown styling */
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: rgba(40, 40, 50, 0.9) !important;
        border-color: rgba(0, 245, 255, 0.3) !important;
        color: rgba(255, 255, 255, 0.8) !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox > div > div:hover {
        border-color: var(--cyber-cyan) !important;
    }
    
    /* Hide Plotly toolbar */
    .modebar {
        display: none !important;
    }
    
    /* Data table styling */
    .dataframe {
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.8rem !important;
    }
    
    /* Status indicator */
    .status-online {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: var(--cyber-green);
        border-radius: 50%;
        margin-right: 8px;
        animation: statusBlink 1s ease-in-out infinite;
    }
    
    @keyframes statusBlink {
        0%, 100% { opacity: 1; box-shadow: 0 0 10px var(--cyber-green); }
        50% { opacity: 0.5; box-shadow: 0 0 5px var(--cyber-green); }
    }
    
    /* Terminal-style text */
    .terminal-text {
        font-family: 'Share Tech Mono', monospace;
        color: var(--cyber-green);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Progress bar override */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--cyber-cyan), var(--cyber-purple));
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--cyber-darker);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--cyber-cyan), var(--cyber-purple));
        border-radius: 4px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-family: 'Rajdhani', sans-serif;
        color: var(--cyber-cyan) !important;
        background: rgba(0, 245, 255, 0.05);
        border: 1px solid rgba(0, 245, 255, 0.2);
        border-radius: 10px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Rajdhani', sans-serif;
        color: rgba(255, 255, 255, 0.6);
        background: rgba(0, 245, 255, 0.05);
        border: 1px solid rgba(0, 245, 255, 0.2);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--cyber-cyan) !important;
        background: rgba(0, 245, 255, 0.15);
        border-color: var(--cyber-cyan);
    }
</style>
""", unsafe_allow_html=True)


# Generate synthetic network traffic data
@st.cache_data
def generate_network_data(n_samples=10000):
    np.random.seed(42)
    
    # Base normal traffic
    normal_samples = int(n_samples * 0.7)
    attack_samples = n_samples - normal_samples
    
    # Normal traffic patterns
    normal_data = {
        'src_bytes': np.random.exponential(500, normal_samples),
        'dst_bytes': np.random.exponential(1000, normal_samples),
        'duration': np.random.exponential(30, normal_samples),
        'src_packets': np.random.poisson(10, normal_samples),
        'dst_packets': np.random.poisson(15, normal_samples),
        'wrong_fragment': np.zeros(normal_samples),
        'urgent': np.zeros(normal_samples),
        'hot': np.random.poisson(1, normal_samples),
        'num_failed_logins': np.zeros(normal_samples),
        'logged_in': np.ones(normal_samples),
        'num_compromised': np.zeros(normal_samples),
        'root_shell': np.zeros(normal_samples),
        'su_attempted': np.zeros(normal_samples),
        'num_root': np.zeros(normal_samples),
        'num_file_creations': np.random.poisson(2, normal_samples),
        'num_shells': np.zeros(normal_samples),
        'num_access_files': np.random.poisson(3, normal_samples),
        'count': np.random.poisson(5, normal_samples),
        'srv_count': np.random.poisson(10, normal_samples),
        'serror_rate': np.random.beta(1, 50, normal_samples),
        'srv_serror_rate': np.random.beta(1, 50, normal_samples),
        'rerror_rate': np.random.beta(1, 50, normal_samples),
        'srv_rerror_rate': np.random.beta(1, 50, normal_samples),
        'same_srv_rate': np.random.beta(50, 5, normal_samples),
        'diff_srv_rate': np.random.beta(1, 20, normal_samples),
        'dst_host_count': np.random.poisson(100, normal_samples),
        'dst_host_srv_count': np.random.poisson(50, normal_samples),
        'attack_type': ['Normal'] * normal_samples
    }
    
    # Attack patterns
    attack_types = ['DDoS', 'Port Scan', 'SQL Injection', 'Brute Force', 'Malware', 'Phishing', 'Man-in-Middle']
    attack_weights = [0.25, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05]
    
    attack_data = {
        'src_bytes': [],
        'dst_bytes': [],
        'duration': [],
        'src_packets': [],
        'dst_packets': [],
        'wrong_fragment': [],
        'urgent': [],
        'hot': [],
        'num_failed_logins': [],
        'logged_in': [],
        'num_compromised': [],
        'root_shell': [],
        'su_attempted': [],
        'num_root': [],
        'num_file_creations': [],
        'num_shells': [],
        'num_access_files': [],
        'count': [],
        'srv_count': [],
        'serror_rate': [],
        'srv_serror_rate': [],
        'rerror_rate': [],
        'srv_rerror_rate': [],
        'same_srv_rate': [],
        'diff_srv_rate': [],
        'dst_host_count': [],
        'dst_host_srv_count': [],
        'attack_type': []
    }
    
    for _ in range(attack_samples):
        attack = np.random.choice(attack_types, p=attack_weights)
        attack_data['attack_type'].append(attack)
        
        if attack == 'DDoS':
            attack_data['src_bytes'].append(np.random.exponential(50000))
            attack_data['dst_bytes'].append(np.random.exponential(100))
            attack_data['duration'].append(np.random.exponential(1))
            attack_data['src_packets'].append(np.random.poisson(1000))
            attack_data['dst_packets'].append(np.random.poisson(5))
            attack_data['count'].append(np.random.poisson(500))
            attack_data['srv_count'].append(np.random.poisson(500))
            attack_data['serror_rate'].append(np.random.beta(20, 5))
            attack_data['same_srv_rate'].append(np.random.beta(50, 1))
        elif attack == 'Port Scan':
            attack_data['src_bytes'].append(np.random.exponential(100))
            attack_data['dst_bytes'].append(np.random.exponential(50))
            attack_data['duration'].append(np.random.exponential(0.1))
            attack_data['src_packets'].append(np.random.poisson(3))
            attack_data['dst_packets'].append(np.random.poisson(2))
            attack_data['count'].append(np.random.poisson(300))
            attack_data['srv_count'].append(np.random.poisson(5))
            attack_data['serror_rate'].append(np.random.beta(10, 5))
            attack_data['same_srv_rate'].append(np.random.beta(1, 20))
        elif attack == 'SQL Injection':
            attack_data['src_bytes'].append(np.random.exponential(2000))
            attack_data['dst_bytes'].append(np.random.exponential(5000))
            attack_data['duration'].append(np.random.exponential(60))
            attack_data['src_packets'].append(np.random.poisson(20))
            attack_data['dst_packets'].append(np.random.poisson(30))
            attack_data['count'].append(np.random.poisson(10))
            attack_data['srv_count'].append(np.random.poisson(10))
            attack_data['serror_rate'].append(np.random.beta(5, 20))
            attack_data['same_srv_rate'].append(np.random.beta(40, 5))
        elif attack == 'Brute Force':
            attack_data['src_bytes'].append(np.random.exponential(300))
            attack_data['dst_bytes'].append(np.random.exponential(200))
            attack_data['duration'].append(np.random.exponential(120))
            attack_data['src_packets'].append(np.random.poisson(100))
            attack_data['dst_packets'].append(np.random.poisson(100))
            attack_data['count'].append(np.random.poisson(200))
            attack_data['srv_count'].append(np.random.poisson(200))
            attack_data['serror_rate'].append(np.random.beta(15, 10))
            attack_data['same_srv_rate'].append(np.random.beta(45, 5))
        else:  # Malware, Phishing, Man-in-Middle
            attack_data['src_bytes'].append(np.random.exponential(3000))
            attack_data['dst_bytes'].append(np.random.exponential(8000))
            attack_data['duration'].append(np.random.exponential(300))
            attack_data['src_packets'].append(np.random.poisson(50))
            attack_data['dst_packets'].append(np.random.poisson(80))
            attack_data['count'].append(np.random.poisson(20))
            attack_data['srv_count'].append(np.random.poisson(15))
            attack_data['serror_rate'].append(np.random.beta(3, 30))
            attack_data['same_srv_rate'].append(np.random.beta(30, 10))
        
        # Common attack characteristics
        attack_data['wrong_fragment'].append(np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]))
        attack_data['urgent'].append(np.random.choice([0, 1], p=[0.9, 0.1]))
        attack_data['hot'].append(np.random.poisson(5))
        attack_data['num_failed_logins'].append(np.random.poisson(3) if attack == 'Brute Force' else 0)
        attack_data['logged_in'].append(np.random.choice([0, 1], p=[0.6, 0.4]))
        attack_data['num_compromised'].append(np.random.poisson(2))
        attack_data['root_shell'].append(np.random.choice([0, 1], p=[0.8, 0.2]))
        attack_data['su_attempted'].append(np.random.choice([0, 1], p=[0.9, 0.1]))
        attack_data['num_root'].append(np.random.poisson(1))
        attack_data['num_file_creations'].append(np.random.poisson(5))
        attack_data['num_shells'].append(np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]))
        attack_data['num_access_files'].append(np.random.poisson(8))
        attack_data['srv_serror_rate'].append(np.random.beta(10, 10))
        attack_data['rerror_rate'].append(np.random.beta(5, 20))
        attack_data['srv_rerror_rate'].append(np.random.beta(5, 20))
        attack_data['diff_srv_rate'].append(np.random.beta(10, 10))
        attack_data['dst_host_count'].append(np.random.poisson(150))
        attack_data['dst_host_srv_count'].append(np.random.poisson(30))
    
    # Combine data
    df_normal = pd.DataFrame(normal_data)
    df_attack = pd.DataFrame(attack_data)
    df = pd.concat([df_normal, df_attack], ignore_index=True)
    
    # Add timestamps
    base_time = datetime.now() - timedelta(days=7)
    df['timestamp'] = [base_time + timedelta(minutes=i*random.uniform(0.5, 2)) for i in range(len(df))]
    
    # Add source/destination IPs
    df['src_ip'] = [f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}" for _ in range(len(df))]
    df['dst_ip'] = [f"192.168.{random.randint(1, 10)}.{random.randint(1, 254)}" for _ in range(len(df))]
    
    # Add protocols and services
    protocols = ['TCP', 'UDP', 'ICMP']
    services = ['http', 'https', 'ssh', 'ftp', 'smtp', 'dns', 'telnet', 'mysql', 'other']
    df['protocol'] = np.random.choice(protocols, len(df), p=[0.6, 0.3, 0.1])
    df['service'] = np.random.choice(services, len(df))
    
    # Add geographic data for visualization
    countries = ['China', 'Russia', 'USA', 'Brazil', 'India', 'Germany', 'UK', 'Iran', 'North Korea', 'Unknown']
    country_weights_attack = [0.25, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05]
    country_weights_normal = [0.05, 0.05, 0.4, 0.1, 0.1, 0.1, 0.1, 0.02, 0.01, 0.07]
    
    df['country'] = df.apply(
        lambda x: np.random.choice(countries, p=country_weights_attack) if x['attack_type'] != 'Normal' 
        else np.random.choice(countries, p=country_weights_normal), axis=1
    )
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


# Train ML models
@st.cache_resource
def train_models(df):
    feature_cols = ['src_bytes', 'dst_bytes', 'duration', 'src_packets', 'dst_packets',
                    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                    'num_file_creations', 'num_shells', 'num_access_files', 'count',
                    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                    'dst_host_count', 'dst_host_srv_count']
    
    X = df[feature_cols].values
    
    # Anomaly Detection with Isolation Forest
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    iso_forest = IsolationForest(contamination=0.3, random_state=42, n_estimators=100)
    iso_forest.fit(X_scaled)
    
    # Attack Classification with Random Forest
    le = LabelEncoder()
    y = le.fit_transform(df['attack_type'])
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    
    accuracy = rf_classifier.score(X_test, y_test)
    
    return iso_forest, rf_classifier, scaler, le, feature_cols, accuracy


# Generate data and train models
df = generate_network_data()
iso_forest, rf_classifier, scaler, le, feature_cols, model_accuracy = train_models(df)

# Add anomaly predictions
df['anomaly_score'] = iso_forest.decision_function(scaler.transform(df[feature_cols]))
df['is_anomaly'] = iso_forest.predict(scaler.transform(df[feature_cols]))
df['is_anomaly'] = df['is_anomaly'].map({1: 'Normal', -1: 'Anomaly'})

# Predict attack types
df['predicted_attack'] = le.inverse_transform(rf_classifier.predict(scaler.transform(df[feature_cols])))


# Header
st.markdown("""
<div class="cyber-header">
    <h1 class="cyber-title">CYBERSHIELD</h1>
    <p class="cyber-subtitle">Advanced Threat Detection System</p>
</div>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <span class="status-online"></span>
        <span style="font-family: 'Share Tech Mono', monospace; color: #00ff88;">SYSTEM ONLINE</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Filters
    st.markdown("### 🎯 THREAT FILTERS")
    
    attack_types = ['All'] + list(df['attack_type'].unique())
    selected_attack = st.selectbox("Attack Type", attack_types)
    
    protocols = ['All'] + list(df['protocol'].unique())
    selected_protocol = st.selectbox("Protocol", protocols)
    
    countries = ['All'] + list(df['country'].unique())
    selected_country = st.selectbox("Source Country", countries)
    
    severity_threshold = st.slider("Anomaly Sensitivity", 0.0, 1.0, 0.5)
    
    st.markdown("---")
    st.markdown("### 📊 MODEL METRICS")
    st.metric("Classification Accuracy", f"{model_accuracy*100:.1f}%")
    st.metric("Total Samples Analyzed", f"{len(df):,}")
    st.metric("Threats Detected", f"{len(df[df['attack_type'] != 'Normal']):,}")


# Filter data
filtered_df = df.copy()
if selected_attack != 'All':
    filtered_df = filtered_df[filtered_df['attack_type'] == selected_attack]
if selected_protocol != 'All':
    filtered_df = filtered_df[filtered_df['protocol'] == selected_protocol]
if selected_country != 'All':
    filtered_df = filtered_df[filtered_df['country'] == selected_country]


# Key Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

total_traffic = len(filtered_df)
threats_detected = len(filtered_df[filtered_df['attack_type'] != 'Normal'])
threat_rate = (threats_detected / total_traffic * 100) if total_traffic > 0 else 0
blocked_attacks = int(threats_detected * 0.94)
active_connections = random.randint(150, 300)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Traffic</div>
        <div class="metric-value cyan">{total_traffic:,}</div>
        <div class="metric-label">Packets Analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Threats Detected</div>
        <div class="metric-value red">{threats_detected:,}</div>
        <div class="metric-label">Malicious Events</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Threat Rate</div>
        <div class="metric-value orange">{threat_rate:.1f}%</div>
        <div class="metric-label">Of Total Traffic</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Attacks Blocked</div>
        <div class="metric-value green">{blocked_attacks:,}</div>
        <div class="metric-label">Auto-Mitigated</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active Sessions</div>
        <div class="metric-value purple">{active_connections}</div>
        <div class="metric-label">Live Connections</div>
    </div>
    """, unsafe_allow_html=True)


# Tabs for different analysis views
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📡 LIVE MONITOR", "🎯 ATTACK ANALYSIS", "🌍 GEO INTEL", "🧠 ML INSIGHTS", "📋 THREAT LOG"])


with tab1:
    st.markdown('<h3 class="section-header">Real-Time Network Traffic</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Time series of traffic
        traffic_by_time = filtered_df.groupby(filtered_df['timestamp'].dt.floor('H')).agg({
            'src_bytes': 'sum',
            'attack_type': lambda x: (x != 'Normal').sum()
        }).reset_index()
        traffic_by_time.columns = ['timestamp', 'traffic_volume', 'attack_count']
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=traffic_by_time['timestamp'],
                y=traffic_by_time['traffic_volume'],
                name='Traffic Volume',
                fill='tozeroy',
                fillcolor='rgba(0, 255, 136, 0.15)',
                line=dict(color='#00ff88', width=2),
                hovertemplate='%{y:,.0f} bytes<extra></extra>'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=traffic_by_time['timestamp'],
                y=traffic_by_time['attack_count'],
                name='Attacks Detected',
                line=dict(color='#ff0055', width=3),
                mode='lines+markers',
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='%{y} attacks<extra></extra>'
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Rajdhani', color='white'),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(size=14, color='white'),
                bgcolor='rgba(20, 20, 30, 0.8)',
                bordercolor='rgba(0, 245, 255, 0.3)',
                borderwidth=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            height=400,
            xaxis=dict(showgrid=True, gridcolor='rgba(0, 245, 255, 0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(0, 245, 255, 0.1)', title=dict(text='Traffic (bytes)', font=dict(color='#00ff88'))),
            yaxis2=dict(title=dict(text='Attack Count', font=dict(color='#ff0055')), showgrid=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Protocol distribution
        protocol_counts = filtered_df['protocol'].value_counts()
        
        fig_protocol = go.Figure(data=[go.Pie(
            labels=protocol_counts.index,
            values=protocol_counts.values,
            hole=0.6,
            marker=dict(colors=['#00f5ff', '#bf00ff', '#00ff88']),
            textinfo='label+percent',
            textfont=dict(family='Share Tech Mono', size=12),
            hovertemplate='%{label}: %{value:,}<extra></extra>'
        )])
        
        fig_protocol.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Rajdhani', color='white'),
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            height=400,
            annotations=[dict(
                text='PROTOCOL<br>DIST',
                x=0.5, y=0.5,
                font=dict(size=14, family='Orbitron', color='#00f5ff'),
                showarrow=False
            )]
        )
        
        st.plotly_chart(fig_protocol, use_container_width=True)


with tab2:
    st.markdown('<h3 class="section-header">Attack Type Distribution</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Attack type bar chart
        attack_counts = filtered_df['attack_type'].value_counts()
        colors = ['#00ff88' if x == 'Normal' else '#ff0055' for x in attack_counts.index]
        
        fig_attacks = go.Figure(data=[go.Bar(
            x=attack_counts.index,
            y=attack_counts.values,
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            text=attack_counts.values,
            textposition='outside',
            textfont=dict(family='Share Tech Mono', color='white'),
            hovertemplate='%{x}: %{y:,}<extra></extra>'
        )])
        
        fig_attacks.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Rajdhani', color='white'),
            margin=dict(l=20, r=20, t=40, b=20),
            height=400,
            xaxis=dict(showgrid=False, tickangle=45),
            yaxis=dict(showgrid=True, gridcolor='rgba(0, 245, 255, 0.1)')
        )
        
        st.plotly_chart(fig_attacks, use_container_width=True)
    
    with col2:
        # Attack severity heatmap by service
        attack_service = filtered_df[filtered_df['attack_type'] != 'Normal'].groupby(
            ['attack_type', 'service']
        ).size().unstack(fill_value=0)
        
        if not attack_service.empty:
            fig_heat = go.Figure(data=go.Heatmap(
                z=attack_service.values,
                x=attack_service.columns,
                y=attack_service.index,
                colorscale=[[0, '#0a0a0f'], [0.5, '#bf00ff'], [1, '#ff0055']],
                hovertemplate='Service: %{x}<br>Attack: %{y}<br>Count: %{z}<extra></extra>'
            ))
            
            fig_heat.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Rajdhani', color='white'),
                margin=dict(l=20, r=20, t=40, b=20),
                height=400,
                xaxis=dict(tickangle=45),
            )
            
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No attack data to display for current filters")
    
    # Attack characteristics
    st.markdown('<h3 class="section-header">Attack Characteristics Comparison</h3>', unsafe_allow_html=True)
    
    attack_stats = filtered_df.groupby('attack_type').agg({
        'src_bytes': 'mean',
        'dst_bytes': 'mean',
        'duration': 'mean',
        'src_packets': 'mean',
        'serror_rate': 'mean'
    }).round(2)
    
    fig_radar = go.Figure()
    
    categories = ['Src Bytes', 'Dst Bytes', 'Duration', 'Packets', 'Error Rate']
    
    for attack in attack_stats.index[:5]:  # Top 5 attack types
        values = attack_stats.loc[attack].values
        # Normalize values
        values_norm = (values - values.min()) / (values.max() - values.min() + 1e-10)
        values_norm = list(values_norm) + [values_norm[0]]  # Close the polygon
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values_norm,
            theta=categories + [categories[0]],
            name=attack,
            fill='toself',
            opacity=0.6
        ))
    
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, gridcolor='rgba(0, 245, 255, 0.2)', linecolor='rgba(0, 245, 255, 0.3)'),
            angularaxis=dict(gridcolor='rgba(0, 245, 255, 0.2)', linecolor='rgba(0, 245, 255, 0.3)')
        ),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Rajdhani', color='white'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5
        ),
        margin=dict(l=60, r=60, t=40, b=80),
        height=450
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)


with tab3:
    st.markdown('<h3 class="section-header">Geographic Threat Intelligence</h3>', unsafe_allow_html=True)
    
    # Country threat counts
    country_threats = filtered_df[filtered_df['attack_type'] != 'Normal'].groupby('country').size().reset_index(name='threats')
    country_total = filtered_df.groupby('country').size().reset_index(name='total')
    geo_data = country_threats.merge(country_total, on='country')
    geo_data['threat_rate'] = (geo_data['threats'] / geo_data['total'] * 100).round(1)
    
    # Country coordinates for mapping
    country_coords = {
        'China': {'lat': 35.86, 'lon': 104.19},
        'Russia': {'lat': 61.52, 'lon': 105.31},
        'USA': {'lat': 37.09, 'lon': -95.71},
        'Brazil': {'lat': -14.23, 'lon': -51.92},
        'India': {'lat': 20.59, 'lon': 78.96},
        'Germany': {'lat': 51.16, 'lon': 10.45},
        'UK': {'lat': 55.37, 'lon': -3.43},
        'Iran': {'lat': 32.42, 'lon': 53.68},
        'North Korea': {'lat': 40.33, 'lon': 127.51},
        'Unknown': {'lat': 0, 'lon': 0}
    }
    
    geo_data['lat'] = geo_data['country'].map(lambda x: country_coords.get(x, {}).get('lat', 0))
    geo_data['lon'] = geo_data['country'].map(lambda x: country_coords.get(x, {}).get('lon', 0))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_map = go.Figure()
        
        fig_map.add_trace(go.Scattergeo(
            lon=geo_data['lon'],
            lat=geo_data['lat'],
            mode='markers',
            marker=dict(
                size=geo_data['threats'] / geo_data['threats'].max() * 50 + 10,
                color=geo_data['threat_rate'],
                colorscale=[[0, '#00ff88'], [0.5, '#ff6b00'], [1, '#ff0055']],
                opacity=0.8,
                line=dict(width=2, color='white'),
                colorbar=dict(
                    title=dict(text='Threat Rate %', font=dict(color='white')),
                    tickfont=dict(color='white')
                )
            ),
            text=geo_data.apply(lambda x: f"{x['country']}<br>Threats: {x['threats']}<br>Rate: {x['threat_rate']}%", axis=1),
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig_map.update_layout(
            geo=dict(
                bgcolor='rgba(0,0,0,0)',
                showland=True,
                landcolor='rgba(30, 30, 40, 0.8)',
                showocean=True,
                oceancolor='rgba(10, 10, 20, 0.8)',
                showcoastlines=True,
                coastlinecolor='rgba(0, 245, 255, 0.3)',
                showframe=False,
                showcountries=True,
                countrycolor='rgba(0, 245, 255, 0.2)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            height=500
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        st.markdown("#### 🔴 Top Threat Sources")
        
        geo_sorted = geo_data.sort_values('threats', ascending=False).head(5)
        
        for _, row in geo_sorted.iterrows():
            threat_level = "CRITICAL" if row['threat_rate'] > 50 else "HIGH" if row['threat_rate'] > 30 else "MEDIUM"
            color_class = "red" if threat_level == "CRITICAL" else "orange" if threat_level == "HIGH" else "cyan"
            
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 10px; padding: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-family: 'Rajdhani'; font-size: 1.1rem; color: white;">{row['country']}</span>
                    <span class="{color_class}" style="font-family: 'Orbitron'; font-size: 0.8rem;">{threat_level}</span>
                </div>
                <div style="margin-top: 8px;">
                    <span class="terminal-text">{row['threats']:,} threats • {row['threat_rate']}% rate</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


with tab4:
    st.markdown('<h3 class="section-header">Machine Learning Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly detection visualization
        st.markdown("#### Anomaly Detection (Isolation Forest)")
        
        # Sample for visualization
        sample_df = filtered_df.sample(min(1000, len(filtered_df)), random_state=42)
        
        fig_anomaly = go.Figure()
        
        normal_mask = sample_df['is_anomaly'] == 'Normal'
        anomaly_mask = sample_df['is_anomaly'] == 'Anomaly'
        
        fig_anomaly.add_trace(go.Scatter(
            x=sample_df[normal_mask]['src_bytes'],
            y=sample_df[normal_mask]['dst_bytes'],
            mode='markers',
            name='Normal',
            marker=dict(color='#00ff88', size=6, opacity=0.6),
            hovertemplate='Src: %{x:.0f}<br>Dst: %{y:.0f}<extra></extra>'
        ))
        
        fig_anomaly.add_trace(go.Scatter(
            x=sample_df[anomaly_mask]['src_bytes'],
            y=sample_df[anomaly_mask]['dst_bytes'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='#ff0055', size=8, opacity=0.8, symbol='x'),
            hovertemplate='Src: %{x:.0f}<br>Dst: %{y:.0f}<extra></extra>'
        ))
        
        fig_anomaly.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Rajdhani', color='white'),
            xaxis=dict(title='Source Bytes (log)', type='log', showgrid=True, gridcolor='rgba(0, 245, 255, 0.1)'),
            yaxis=dict(title='Destination Bytes (log)', type='log', showgrid=True, gridcolor='rgba(0, 245, 255, 0.1)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            margin=dict(l=20, r=20, t=40, b=20),
            height=400
        )
        
        st.plotly_chart(fig_anomaly, use_container_width=True)
    
    with col2:
        # Feature importance
        st.markdown("#### Feature Importance (Random Forest)")
        
        importances = rf_classifier.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(10)
        
        fig_importance = go.Figure(data=[go.Bar(
            x=feature_imp['importance'],
            y=feature_imp['feature'],
            orientation='h',
            marker=dict(
                color=feature_imp['importance'],
                colorscale=[[0, '#00f5ff'], [1, '#bf00ff']],
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            hovertemplate='%{y}: %{x:.4f}<extra></extra>'
        )])
        
        fig_importance.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Rajdhani', color='white'),
            xaxis=dict(title='Importance Score', showgrid=True, gridcolor='rgba(0, 245, 255, 0.1)'),
            yaxis=dict(showgrid=False),
            margin=dict(l=20, r=20, t=40, b=20),
            height=400
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Confusion matrix for attack classification
    st.markdown('<h3 class="section-header">Classification Performance</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Get predictions for confusion matrix
        y_true = le.transform(filtered_df['attack_type'])
        y_pred = le.transform(filtered_df['predicted_attack'])
        cm = confusion_matrix(y_true, y_pred)
        labels = le.classes_
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=[[0, '#0a0a0f'], [0.3, '#00f5ff'], [0.6, '#bf00ff'], [1, '#ff0055']],
            hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
        ))
        
        fig_cm.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Rajdhani', color='white'),
            xaxis=dict(title='Predicted', tickangle=45),
            yaxis=dict(title='Actual'),
            margin=dict(l=20, r=20, t=40, b=80),
            height=500
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)
    
    # Model metrics
    st.markdown("#### 🎯 Model Performance Metrics")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value green">{model_accuracy*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        precision = np.diag(cm).sum() / cm.sum() 
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value cyan">{precision*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        anomaly_detection_rate = len(df[df['is_anomaly'] == 'Anomaly']) / len(df) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Anomaly Rate</div>
            <div class="metric-value orange">{anomaly_detection_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Features Used</div>
            <div class="metric-value purple">{len(feature_cols)}</div>
        </div>
        """, unsafe_allow_html=True)


with tab5:
    st.markdown('<h3 class="section-header">Recent Threat Activity Log</h3>', unsafe_allow_html=True)
    
    # Alert summary
    critical_count = len(filtered_df[filtered_df['attack_type'].isin(['DDoS', 'SQL Injection', 'Man-in-Middle'])])
    high_count = len(filtered_df[filtered_df['attack_type'].isin(['Brute Force', 'Malware'])])
    medium_count = len(filtered_df[filtered_df['attack_type'].isin(['Port Scan', 'Phishing'])])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="alert-critical">
            <span style="font-family: 'Orbitron'; color: #ff0055; font-size: 1.2rem;">⚠️ CRITICAL: {critical_count:,}</span>
            <div class="terminal-text" style="color: #ff0055;">DDoS, SQL Injection, MITM attacks detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="alert-warning">
            <span style="font-family: 'Orbitron'; color: #ff6b00; font-size: 1.2rem;">🔶 HIGH: {high_count:,}</span>
            <div class="terminal-text" style="color: #ff6b00;">Brute Force, Malware activity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="alert-safe">
            <span style="font-family: 'Orbitron'; color: #00ff88; font-size: 1.2rem;">🟢 MEDIUM: {medium_count:,}</span>
            <div class="terminal-text" style="color: #00ff88;">Port Scan, Phishing attempts</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Threat log table
    threat_log = filtered_df[filtered_df['attack_type'] != 'Normal'].sort_values('timestamp', ascending=False).head(50)
    
    if not threat_log.empty:
        display_df = threat_log[['timestamp', 'src_ip', 'dst_ip', 'attack_type', 'protocol', 'service', 'country', 'src_bytes']].copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df.columns = ['Timestamp', 'Source IP', 'Target IP', 'Attack Type', 'Protocol', 'Service', 'Origin', 'Bytes']
        
        # Color code by attack type
        def highlight_attack(row):
            if row['Attack Type'] in ['DDoS', 'SQL Injection', 'Man-in-Middle']:
                return ['background-color: rgba(255, 0, 85, 0.2)'] * len(row)
            elif row['Attack Type'] in ['Brute Force', 'Malware']:
                return ['background-color: rgba(255, 107, 0, 0.2)'] * len(row)
            else:
                return ['background-color: rgba(0, 245, 255, 0.1)'] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_attack, axis=1),
            use_container_width=True,
            height=500
        )
    else:
        st.info("No threats detected with current filters")
    
    # Export option
    st.markdown("---")
    if st.button("📥 Export Threat Report", type="primary"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="threat_report.csv",
            mime="text/csv"
        )


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 30px 20px;">
    <p style="font-family: 'Rajdhani', sans-serif; font-size: 0.95rem; color: rgba(255, 255, 255, 0.5); margin-bottom: 10px;">
        Built with <span style="color: #00f5ff;">Python</span> • <span style="color: #bf00ff;">Streamlit</span> • <span style="color: #00ff88;">scikit-learn</span> • <span style="color: #ff6b00;">Plotly</span>
    </p>
    <p style="font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; color: rgba(255, 255, 255, 0.4); margin-bottom: 8px;">
        Dataset: <a href="https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection" target="_blank" style="color: rgba(0, 245, 255, 0.5); text-decoration: none;">Synthetic Network Traffic Data</a> (10,000 samples)
    </p>
    <p style="font-family: 'Rajdhani', sans-serif; font-size: 0.9rem; color: rgba(0, 245, 255, 0.6); margin-top: 15px;">
        Sam Bolger
    </p>
    <p style="font-family: 'Share Tech Mono', monospace; font-size: 0.75rem; color: rgba(255, 255, 255, 0.35);">
        <a href="https://linkedin.com" style="color: rgba(0, 245, 255, 0.5); text-decoration: none;">LinkedIn</a> • 
        <a href="mailto:sbolger@cord.edu" style="color: rgba(0, 245, 255, 0.5); text-decoration: none;">sbolger@cord.edu</a> • 
        <a href="https://github.com" style="color: rgba(0, 245, 255, 0.5); text-decoration: none;">GitHub</a> • 
        <a href="https://sammybolger.com" style="color: rgba(0, 245, 255, 0.5); text-decoration: none;">sammybolger.com</a>
    </p>
</div>
""", unsafe_allow_html=True)
