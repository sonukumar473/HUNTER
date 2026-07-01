import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack, csr_matrix
import base64

# ---------------------------
# 1. PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="HUNTER - Fake Job Detector", page_icon="🔍", layout="wide")

# ---------------------------
# 2. MATRIX RAIN BACKGROUND (CSS + HTML Animation)
# ---------------------------
st.markdown("""
<style>
    /* Reset and base */
    .stApp {
        background: #0a0a0a !important;
    }

    /* Matrix Rain Canvas - Full Screen Background */
    #matrix-canvas {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1;
        opacity: 0.15;
        pointer-events: none;
    }

    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    
    * {
        font-family: 'Share Tech Mono', monospace !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main Container - Glass Effect */
    .main .block-container {
        background: rgba(10, 10, 10, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 65, 0.1);
        border-radius: 15px;
        padding: 30px 40px !important;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 60px rgba(0, 255, 65, 0.05);
    }

    /* Sidebar Glass Effect */
    [data-testid="stSidebar"] {
        background: rgba(5, 5, 5, 0.85) !important;
        backdrop-filter: blur(20px);
        border-right: 2px solid rgba(0, 255, 65, 0.3) !important;
        box-shadow: 5px 0 40px rgba(0, 255, 65, 0.05) !important;
    }
    [data-testid="stSidebar"] * {
        color: #00ff41 !important;
    }

    /* Text Area */
    textarea {
        background: rgba(13, 13, 13, 0.8) !important;
        color: #00ff41 !important;
        border: 1px solid #00ff41 !important;
        border-radius: 8px !important;
        font-size: 15px !important;
        box-shadow: inset 0 0 30px rgba(0, 255, 65, 0.03) !important;
        transition: all 0.3s ease;
    }
    textarea:focus {
        box-shadow: 0 0 30px rgba(0, 255, 65, 0.15), inset 0 0 30px rgba(0, 255, 65, 0.05) !important;
        border-color: #00ff41 !important;
    }

    /* Glowing Button */
    .stButton > button {
        background: rgba(0, 255, 65, 0.05);
        color: #00ff41;
        border: 1.5px solid #00ff41;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 30px;
        transition: all 0.3s ease-in-out;
        text-transform: uppercase;
        letter-spacing: 2px;
        width: 100%;
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.05);
    }
    .stButton > button:hover {
        background: #00ff41 !important;
        color: #0a0a0a !important;
        box-shadow: 0 0 50px rgba(0, 255, 65, 0.4);
        transform: scale(1.02);
        border-color: #ffffff;
    }

    /* Success / Error / Warning Boxes */
    .stSuccess {
        background: rgba(0, 255, 65, 0.05) !important;
        border-left: 4px solid #00ff41 !important;
        border-radius: 8px !important;
        color: #00ff41 !important;
        backdrop-filter: blur(10px);
    }
    .stError {
        background: rgba(255, 0, 68, 0.05) !important;
        border-left: 4px solid #ff0044 !important;
        border-radius: 8px !important;
        color: #ff0044 !important;
        backdrop-filter: blur(10px);
    }
    .stWarning {
        background: rgba(255, 187, 0, 0.05) !important;
        border-left: 4px solid #ffbb00 !important;
        border-radius: 8px !important;
        color: #ffbb00 !important;
        backdrop-filter: blur(10px);
    }
    .stInfo {
        background: rgba(0, 150, 255, 0.05) !important;
        border-left: 4px solid #0096ff !important;
        border-radius: 8px !important;
        color: #0096ff !important;
        backdrop-filter: blur(10px);
    }

    /* Headers */
    h1, h2, h3, h4, h5 {
        color: #00ff41 !important;
        text-shadow: 0 0 20px rgba(0, 255, 65, 0.15);
    }

    /* Column headers */
    .css-1d391kg {
        color: #00ff41 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        background: #0a0a0a;
    }
    ::-webkit-scrollbar-thumb {
        background: #00ff41;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-track {
        background: #050505;
    }

    /* Divider */
    hr {
        border-top: 1px solid rgba(0, 255, 65, 0.15);
        margin: 20px 0;
    }

    /* Safety Card */
    .safety-card {
        background: rgba(0, 255, 65, 0.03);
        border: 1px solid rgba(0, 255, 65, 0.15);
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    .safety-card:hover {
        background: rgba(0, 255, 65, 0.08);
        border-color: rgba(0, 255, 65, 0.3);
        transform: translateX(5px);
    }
    .safety-icon {
        color: #00ff41;
        font-size: 20px;
        margin-right: 10px;
    }
</style>

<!-- Matrix Rain Canvas Background -->
<canvas id="matrix-canvas"></canvas>
<script>
    const canvas = document.getElementById('matrix-canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    const chars = '01';
    const fontSize = 14;
    const columns = Math.floor(canvas.width / fontSize);
    const drops = new Array(columns).fill(1);
    
    function draw() {
        ctx.fillStyle = 'rgba(10, 10, 10, 0.05)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.fillStyle = '#00ff41';
        ctx.font = fontSize + 'px monospace';
        
        for (let i = 0; i < drops.length; i++) {
            const char = chars[Math.floor(Math.random() * chars.length)];
            ctx.fillText(char, i * fontSize, drops[i] * fontSize);
            
            if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                drops[i] = 0;
            }
            drops[i]++;
        }
    }
    
    setInterval(draw, 50);
    
    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
</script>
""", unsafe_allow_html=True)

# ---------------------------
# 3. SIDEBAR - HUNTER BRANDING
# ---------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; border-bottom: 1px solid rgba(0, 255, 65, 0.2); padding-bottom: 20px; margin-bottom: 20px;">
        <pre style="color: #00ff41; font-size: 9px; line-height: 1.1; text-align: left;">
    ██╗  ██╗██╗   ██╗███╗   ██╗████████╗███████╗██████╗ 
    ██║  ██║██║   ██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗
    ███████║██║   ██║██╔██╗ ██║   ██║   █████╗  ██████╔╝
    ██╔══██║██║   ██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗
    ██║  ██║╚██████╔╝██║ ╚████║   ██║   ███████╗██║  ██║
    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
        </pre>
        <span style="color: #00ff41; font-size: 14px; letter-spacing: 3px;">// HUNTER v2.0</span><br>
        <span style="color: #666666; font-size: 10px;">[ FAKE JOB HUNTING TERMINAL ]</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Status
    st.markdown("""
    <div style="background: rgba(13, 13, 13, 0.8); padding: 12px; border: 1px solid rgba(0, 255, 65, 0.3); border-radius: 8px; margin-bottom: 20px;">
        <span style="color: #00ff41;">● SYSTEM: </span><span style="color: #00ff41; font-weight: bold;">ONLINE</span><br>
        <span style="color: #555555; font-size: 11px;">[ MODEL: LOGISTIC REGRESSION ]</span><br>
        <span style="color: #555555; font-size: 11px;">[ DATASET: 18,000 JOBS ]</span><br>
        <span style="color: #555555; font-size: 11px;">[ ACCURACY: 98.5% ]</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Red Flags
    st.markdown("""
    <div style="color: #666666; font-size: 12px; border-top: 1px solid rgba(0, 255, 65, 0.15); padding-top: 15px;">
        <span style="color: #00ff41; font-size: 14px;">▸ HUNTER RED FLAGS</span><br><br>
        <span style="color: #ff0044;">▸</span> Urgency (!!!, ASAP)<br>
        <span style="color: #ff0044;">▸</span> Money Symbols ($, $$$)<br>
        <span style="color: #ff0044;">▸</span> Wire Transfers<br>
        <span style="color: #ff0044;">▸</span> Fees / Payments<br>
        <span style="color: #ff0044;">▸</span> Suspicious Links<br>
        <span style="color: #ff0044;">▸</span> No Interview Required<br>
        <span style="color: #ff0044;">▸</span> Unrealistic Salaries
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# 4. 3D MODEL (Three.js Wireframe Cube)
# ---------------------------
st.markdown("""
<div style="position: fixed; bottom: 20px; right: 20px; z-index: 999; opacity: 0.2; pointer-events: none; width: 150px; height: 150px;">
    <canvas id="three-canvas"></canvas>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
    // Only run if Three.js loaded
    if (typeof THREE !== 'undefined') {
        const container = document.createElement('div');
        container.style.width = '150px';
        container.style.height = '150px';
        container.style.position = 'fixed';
        container.style.bottom = '20px';
        container.style.right = '20px';
        container.style.zIndex = '999';
        container.style.pointerEvents = 'none';
        container.style.opacity = '0.3';
        document.body.appendChild(container);
        
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
        camera.position.set(3, 2, 5);
        camera.lookAt(0, 0, 0);
        
        const renderer = new THREE.WebGLRenderer({ alpha: true });
        renderer.setSize(150, 150);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);
        
        // Wireframe Cube
        const geometry = new THREE.BoxGeometry(1.5, 1.5, 1.5);
        const edges = new THREE.EdgesGeometry(geometry);
        const material = new THREE.LineBasicMaterial({ color: 0x00ff41 });
        const cube = new THREE.LineSegments(edges, material);
        scene.add(cube);
        
        // Inner Wireframe Sphere
        const sphereGeo = new THREE.SphereGeometry(0.8, 16, 16);
        const sphereEdges = new THREE.EdgesGeometry(sphereGeo);
        const sphereMat = new THREE.LineBasicMaterial({ color: 0x00ff41, opacity: 0.3, transparent: true });
        const sphere = new THREE.LineSegments(sphereEdges, sphereMat);
        scene.add(sphere);
        
        // Ambient glow
        const ambientLight = new THREE.AmbientLight(0x00ff41, 0.1);
        scene.add(ambientLight);
        
        function animate() {
            requestAnimationFrame(animate);
            cube.rotation.x += 0.01;
            cube.rotation.y += 0.015;
            sphere.rotation.x -= 0.005;
            sphere.rotation.y -= 0.007;
            renderer.render(scene, camera);
        }
        animate();
        
        window.addEventListener('resize', () => {
            // Keep it small
        });
    }
</script>
""", unsafe_allow_html=True)

# ---------------------------
# 5. MAIN PAGE - HUNTER HEADER
# ---------------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
    <div style="border-bottom: 2px solid rgba(0, 255, 65, 0.2); padding-bottom: 15px; margin-bottom: 25px;">
        <span style="color: #00ff41; font-size: 34px; font-weight: bold;">🔍 HUNTER</span>
        <span style="color: #666666; font-size: 18px; margin-left: 15px;">| Fake Job Detection</span>
        <br>
        <span style="color: #555555; font-size: 13px; letter-spacing: 1px;">
        ⚡ Powered by Machine Learning • Real-time Scam Analysis
        </span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: right; padding-top: 10px;">
        <span style="color: #00ff41; font-size: 12px; border: 1px solid rgba(0,255,65,0.2); padding: 5px 12px; border-radius: 20px;">
        ● LIVE
        </span>
    </div>
    """, unsafe_allow_html=True)

# Terminal Prompt
st.markdown("""
<div style="color: #aaaaaa; font-size: 15px; margin-bottom: 20px; background: rgba(0, 255, 65, 0.02); padding: 12px 18px; border-radius: 8px; border-left: 3px solid #00ff41;">
    <span style="color: #00ff41;">hunter@system:~$</span> ./scan_job --target=job_description<br>
    <span style="color: #555555;">// Paste a job description below. HUNTER will analyze and hunt down scam patterns.</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# 6. CACHE TRAINING
# ---------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("fake_job_postings.csv")
    
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df["raw_text"] = df["description"].fillna("") + " " + df["requirements"].fillna("") + " " + df["company_profile"].fillna("")
    df["clean_text"] = df["raw_text"].apply(clean_text)
    
    df["text_length"] = df["raw_text"].str.len()
    df["exclamation_count"] = df["raw_text"].str.count(r'!')
    df["dollar_count"] = df["raw_text"].str.count(r'\$')
    df["question_count"] = df["raw_text"].str.count(r'\?')
    
    vectorizer = TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df["clean_text"])
    X_extra = csr_matrix(df[["text_length", "exclamation_count", "dollar_count", "question_count"]].values)
    X = hstack([X_text, X_extra])
    y = df["fraudulent"]
    
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, C=0.5)
    model.fit(X, y)
    
    return model, vectorizer

with st.spinner("⚡ [HUNTER] Loading Neural Matrix..."):
    model, vectorizer = train_model()

st.markdown("""
<div style="background: rgba(0, 255, 65, 0.03); padding: 8px 15px; border-left: 3px solid #00ff41; border-radius: 4px; margin-bottom: 20px;">
    <span style="color: #00ff41; font-size: 13px;">✔ HUNTER Model Loaded — Ready to Hunt</span>
    <span style="color: #555555; font-size: 12px; float: right;">v2.0</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# 7. PREDICTION FUNCTION
# ---------------------------
def clean_text_single(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_job(text):
    cleaned = clean_text_single(text)
    text_vec = vectorizer.transform([cleaned])
    
    extra = csr_matrix([
        [len(text), text.count('!'), text.count('$'), text.count('?')]
    ])
    combined = hstack([text_vec, extra])
    
    prob = model.predict_proba(combined)[0]
    fake_prob = prob[1] * 100
    real_prob = prob[0] * 100
    
    if fake_prob > 35:
        return "⚠️ FAKE JOB HUNTED", fake_prob, "red"
    else:
        return "✅ REAL JOB", real_prob, "green"

# ---------------------------
# 8. INPUT AREA (2 Columns)
# ---------------------------
st.markdown('<span style="color: #00ff41;">hunter@system:~$</span> <span style="color: #ffffff;">enter_job_description</span>', unsafe_allow_html=True)

user_input = st.text_area("", placeholder="Paste the full job description here... (minimum 20 characters)", height=180, label_visibility="collapsed")

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_btn = st.button("⚡ HUNT SCAM", type="primary", use_container_width=True)

# ---------------------------
# 9. RESULTS + SAFETY COLUMN (2 Columns)
# ---------------------------
if analyze_btn:
    if len(user_input.strip()) < 20:
        st.warning("⚠️ [ERROR] Input too short. Please paste a full job description.")
    else:
        label, confidence, color = predict_job(user_input)
        
        # Create 2 columns: Results (Left) | Safety Tips (Right)
        col_result, col_safety = st.columns([3, 2])
        
        with col_result:
            st.markdown("---")
            st.markdown("### 📡 HUNTER REPORT")
            
            if color == "red":
                st.error(f"### {label}")
                st.error(f"**Confidence:** {confidence:.1f}%")
                st.error("🚨 **Threat Detected:** This posting matches scam patterns.")
                
                st.markdown("#### 🔍 Red Flag Analysis:")
                red_flags_found = False
                
                if "urgent" in user_input.lower() or "immediately" in user_input.lower():
                    st.warning("⚠️ High-urgency language detected (e.g., 'urgent', 'immediately').")
                    red_flags_found = True
                if "$" in user_input:
                    st.warning("⚠️ Money symbols ('$') detected. Scammers often request payments.")
                    red_flags_found = True
                if "wire" in user_input.lower() or "transfer" in user_input.lower():
                    st.warning("⚠️ 'Wire transfer' detected. Legitimate jobs never ask for this.")
                    red_flags_found = True
                if "fee" in user_input.lower() or "payment" in user_input.lower():
                    st.warning("⚠️ Request for 'fee' or 'payment' detected. This is a major red flag.")
                    red_flags_found = True
                if "whatsapp" in user_input.lower() or "telegram" in user_input.lower():
                    st.warning("⚠️ Third-party messaging app mentioned. Scammers avoid official channels.")
                    red_flags_found = True
                if not red_flags_found:
                    st.info("🔍 No specific keywords found, but the AI pattern analysis flagged this as suspicious.")
            
            else:
                st.success(f"### {label}")
                st.success(f"**Confidence:** {confidence:.1f}%")
                st.success("✅ **System Verdict:** This posting appears legitimate.")
                st.markdown("#### 📝 Summary:")
                st.info("No major scam indicators detected. The language and structure match real job listings.")
        
        with col_safety:
            st.markdown("---")
            st.markdown("### 🛡️ SAFETY PRECAUTIONS")
            st.markdown("""
            <div class="safety-card">
                <span class="safety-icon">🔹</span> <b>NEVER pay to apply</b><br>
                <span style="color: #888; font-size: 12px;">Legitimate companies never ask for fees.</span>
            </div>
            <div class="safety-card">
                <span class="safety-icon">🔹</span> <b>Verify the company</b><br>
                <span style="color: #888; font-size: 12px;">Check official website and LinkedIn.</span>
            </div>
            <div class="safety-card">
                <span class="safety-icon">🔹</span> <b>Watch for urgency</b><br>
                <span style="color: #888; font-size: 12px;">Scammers create false time pressure.</span>
            </div>
            <div class="safety-card">
                <span class="safety-icon">🔹</span> <b>No wire transfers</b><br>
                <span style="color: #888; font-size: 12px;">Real jobs pay you, not the other way.</span>
            </div>
            <div class="safety-card">
                <span class="safety-icon">🔹</span> <b>Check for reviews</b><br>
                <span style="color: #888; font-size: 12px;">Search "Company Name + scam" online.</span>
            </div>
            <div class="safety-card">
                <span class="safety-icon">🔹</span> <b>Trust your gut</b><br>
                <span style="color: #888; font-size: 12px;">If it sounds too good, it probably is.</span>
            </div>
            """, unsafe_allow_html=True)

else:
    # Show safety tips even before scanning
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("---")
        st.markdown("""
        ### 📡 Ready to Hunt
        Paste a job description above and click **"HUNT SCAM"** to analyze it.
        """)
    with col_right:
        st.markdown("---")
        st.markdown("""
        ### 🛡️ Job Safety Tips
        - Never pay to apply
        - Verify the company
        - Watch for urgency
        - Avoid wire transfers
        - Check online reviews
        """)

# ---------------------------
# 10. FOOTER
# ---------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #444444; font-size: 11px; padding: 15px 0;">
    HUNTER v2.0 • Powered by Machine Learning • Always verify independently
</div>
""", unsafe_allow_html=True)