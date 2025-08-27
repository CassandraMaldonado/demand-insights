import streamlit as st
import pandas as pd

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    HAS_GRAPH_LIBS = True
except ImportError:
    HAS_GRAPH_LIBS = False
    st.warning("NetworkX and/or Matplotlib not available. Graph visualization will be disabled.")

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    st.warning("OpenAI library not available. LLM mode will be disabled.")

import textwrap

st.set_page_config(page_title="Aurora AI - Recomendador", layout="wide")

# 1) Reglas y estadísticas con los datos precargados
RULES = [
    {
        "id": "R01",
        "antecedent": [
            "Tortillas de maíz Mixtas (Maíz de Corazón) - Docena",
            "Kombucha de Cardamomo - 750 ml"
        ],
        "consequent": ["Kombucha de Jengibre - 750 ml"],
        "support": 0.0106,
        "confidence": 0.6737,
        "lift": 16.28
    },
    {
        "id": "R02",
        "antecedent": ["Kombucha de Jengibre - 750 ml"],
        "consequent": [
            "Tortillas de maíz Mixtas (Maíz de Corazón) - Docena",
            "Kombucha de Cardamomo - 750 ml"
        ],
        "support": 0.0106,
        "confidence": 0.2560,
        "lift": 16.28
    },
    {
        "id": "R03",
        "antecedent": [
            "Churros de amaranto chipotle con maíz y chía - 210 gr",
            "Pollo sin retazo | 1.5 a 1.7 kg aprox. - Paquete"
        ],
        "consequent": [
            "Churros de amaranto natural con maíz y chía - 210 gr",
            "Huevo de Gallina - 12 pzas - Docena",
            "Fresa - domo de 450 gr - 450 gr"
        ],
        "support": 0.0103,
        "confidence": 0.5299,
        "lift": 15.62
    },
    {
        "id": "R04",
        "antecedent": [
            "Churros de amaranto natural con maíz y chía - 210 gr",
            "Huevo de Gallina - 12 pzas - Docena",
            "Fresa - domo de 450 gr - 450 gr"
        ],
        "consequent": [
            "Churros de amaranto chipotle con maíz y chía - 210 gr",
            "Pollo sin retazo | 1.5 a 1.7 kg aprox. - Paquete"
        ],
        "support": 0.0103,
        "confidence": 0.3024,
        "lift": 15.62
    },
    {
        "id": "R05",
        "antecedent": [
            "Fresa - domo de 450 gr - 450 gr",
            "Churros de amaranto chipotle con maíz y chía - 210 gr",
            "Blueberry - domo de 170 gr - 170 gr"
        ],
        "consequent": [
            "Frambuesa - domo de 170 gr - 170 gr",
            "Churros de amaranto natural con maíz y chía - 210 gr",
            "Huevo de Gallina - 12 pzas - Docena"
        ],
        "support": 0.0104,
        "confidence": 0.4375,
        "lift": 14.93
    },
    {
        "id": "R06",
        "antecedent": [
            "Frambuesa - domo de 170 gr - 170 gr",
            "Churros de amaranto natural con maíz y chía - 210 gr",
            "Huevo de Gallina - 12 pzas - Docena"
        ],
        "consequent": [
            "Fresa - domo de 450 gr - 450 gr",
            "Churros de amaranto chipotle con maíz y chía - 210 gr",
            "Blueberry - domo de 170 gr - 170 gr"
        ],
        "support": 0.0104,
        "confidence": 0.3559,
        "lift": 14.93
    },
    {
        "id": "R07",
        "antecedent": [
            "Fresa - domo de 450 gr - 450 gr",
            "Churros de amaranto chipotle con maíz y chía - 210 gr",
            "Blueberry - domo de 170 gr - 170 gr"
        ],
        "consequent": [
            "Frambuesa - domo de 170 gr - 170 gr",
            "Churros de amaranto natural con maíz y chía - 210 gr"
        ],
        "support": 0.0126,
        "confidence": 0.5278,
        "lift": 14.90
    },
    {
        "id": "R08",
        "antecedent": [
            "Frambuesa - domo de 170 gr - 170 gr",
            "Churros de amaranto natural con maíz y chía - 210 gr"
        ],
        "consequent": [
            "Fresa - domo de 450 gr - 450 gr",
            "Churros de amaranto chipotle con maíz y chía - 210 gr",
            "Blueberry - domo de 170 gr - 170 gr"
        ],
        "support": 0.0126,
        "confidence": 0.3551,
        "lift": 14.90
    },
    {
        "id": "R09",
        "antecedent": [
            "Churros de amaranto natural con maíz y chía - 210 gr",
            "Pollo sin retazo | 1.5 a 1.7 kg aprox. - Paquete"
        ],
        "consequent": [
            "Churros de amaranto chipotle con maíz y chía - 210 gr",
            "Huevo de Gallina - 12 pzas - Docena",
            "Fresa - domo de 450 gr - 450 gr"
        ],
        "support": 0.0103,
        "confidence": 0.4429,
        "lift": 14.70
    },
    {
        "id": "R10",
        "antecedent": [
            "Churros de amaranto chipotle con maíz y chía - 210 gr",
            "Huevo de Gallina - 12 pzas - Docena",
            "Fresa - domo de 450 gr - 450 gr"
        ],
        "consequent": [
            "Churros de amaranto natural con maíz y chía - 210 gr",
            "Pollo sin retazo | 1.5 a 1.7 kg aprox. - Paquete"
        ],
        "support": 0.0103,
        "confidence": 0.3407,
        "lift": 14.70
    }
]

# Estadísticas generales que usaremos en prompts/respuestas
STATS = {
    "total_orders": 6042,
    "unique_products": 774,
    "avg_basket_size": 14.61,
    "total_items_sold": 88303,
    "top_products_support": [
        ("Huevo de Gallina - 12 pzas - Docena", 0.577),
        ("Nopal sin espina | 5 pzas - 5 piezas", 0.372),
        ("Jitomate saladette - kg - kg", 0.351),
        ("Blueberry - domo de 170 gr - 170 gr", 0.300),
        ("Frambuesa - domo de 170 gr - 170 gr", 0.296)
    ]
}

# Derivar lista de productos desde reglas y top products
PRODUCTS = sorted({p for r in RULES for p in (r["antecedent"] + r["consequent"])} |
                  {p for p, _ in STATS["top_products_support"]})

# 2) Helpers
def rules_df():
    rows = []
    for r in RULES:
        rows.append({
            "id": r["id"],
            "antecedent": " + ".join(r["antecedent"]),
            "consequent": " + ".join(r["consequent"]),
            "support": r["support"],
            "confidence": r["confidence"],
            "lift": r["lift"]
        })
    return pd.DataFrame(rows)

def find_products_in_text(text):
    text_low = text.lower()
    found = set()
    # match by substring: most robust for product labels
    for p in PRODUCTS:
        if p.lower() in text_low:
            found.add(p)
    return sorted(found)

def match_rules_by_products(products_set):
    matched = []
    for r in RULES:
        antecedent_set = set(r["antecedent"])
        # match if any antecedent product appears in user's text, but prefer full antecedent match when possible
        if antecedent_set.issubset(products_set):
            matched.append((r, "full"))
        elif len(antecedent_set & products_set) > 0:
            matched.append((r, "partial"))
    # sort: full matches, then by lift desc
    matched.sort(key=lambda x: (0 if x[1]=="full" else 1, -x[0]["lift"], -x[0]["confidence"]))
    return matched

def format_rule_short(r):
    return f"{r['id']}: IF {' & '.join(r['antecedent'])} THEN {' & '.join(r['consequent'])} (support={r['support']:.4f}, conf={r['confidence']:.3f}, lift={r['lift']:.2f})"

# Simple local rule-based answer generator
def rule_based_answer(question):
    products = set(find_products_in_text(question))
    if not products:
        # No product mentioned tehn it provides top suggestions / bundles
        top_bundles = [
            ("R01", "Kombucha Duo + Tortillas", "Kombucha de Cardamomo + Kombucha de Jengibre + Tortillas Mixtas", "8-10%"),
            ("R05", "Desayuno Antioxidante", "Huevos + Berries + Churros", "10% en tercer ítem"),
            ("R03", "Proteína & Snack Picante", "Pollo + Churros chipotle + Fresa", "5-8% o envío gratis")
        ]
        lines = [
            "No identifico productos explícitos en tu pregunta. Basado en el catálogo y reglas fuertes, sugiero:"
        ]
        for rid, title, cont, disc in top_bundles:
            lines.append(f"- {title}: {cont} (sugerido {disc}; basada en {rid})")
        return "\n".join(lines), []
    else:
        matched = match_rules_by_products(products)
        if not matched:
            return "Encontré productos, pero no hay reglas que los conecten directamente en las reglas cargadas. Puedes pedir sugerencias generales o pedir ver las reglas disponibles.", []
        # Build answer with best matches (top 3)
        answer_lines = []
        cited = []
        answer_lines.append(f"He detectado estos productos en tu consulta: {', '.join(products)}.")
        answer_lines.append("Reglas relevantes (ordenadas por correspondencia y fuerza):")
        for r, kind in matched[:6]:
            cited.append(r["id"])
            answer_lines.append(f"- {format_rule_short(r)} -- match={kind}")
            # action suggestion
            if r["lift"] > 10:
                action = "Recomendado activar como bundle o cross-sell"
            elif r["lift"] > 3:
                action = "Considerar como recomendación contextual"
            else:
                action = "Útil como dato, baja prioridad para acción."
            answer_lines.append(f"  → Acción sugerida: {action} (mostrar en PDP y carrito).")
        return "\n".join(answer_lines), cited

# System prompt for LLM mode
def build_system_prompt():
    header = (
        "Eres un asistente en español especializado en recomendaciones de producto para una tienda.\n"
        "Respond ONLY using the rules and statistics provided below. Do NOT invent facts or use external knowledge.\n"
        "When you reference a rule in your answer, include its rule id(s) in square brackets (e.g. [R01]).\n"
        "If the question cannot be answered using the rules or stats, say that no data is available and provide a safe, generic business suggestion.\n"
        "Always answer in Spanish and be concise (2-6 frases)."
    )
    stats_block = "\nEstadísticas globales:\n"
    stats_block += f"- Órdenes totales: {STATS['total_orders']}\n"
    stats_block += f"- Productos únicos: {STATS['unique_products']}\n"
    stats_block += f"- Tamaño promedio de carrito: {STATS['avg_basket_size']}\n"
    stats_block += f"- Items vendidos totales: {STATS['total_items_sold']}\n"
    stats_block += "- Top productos (soporte):\n"
    for p, s in STATS["top_products_support"]:
        stats_block += f"  * {p}: {s:.3f}\n"

    rules_text = "\nReglas (ID, antecedente => consecuente, support, confidence, lift):\n"
    for r in RULES:
        rules_text += (f"- {r['id']}: IF " + " + ".join(r['antecedent']) + " => " +
                       " + ".join(r['consequent']) +
                       f"  (support={r['support']:.4f}, confidence={r['confidence']:.4f}, lift={r['lift']:.2f})\n")

    return header + stats_block + rules_text

# 3) Styling

st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Custom gradient background for header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: none;
    }
    
    .stat-card h3 {
        margin: 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .stat-card .metric {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    /* Action buttons styling */
    .stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
    }
    
    /* Chat container */
    .chat-container {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Response cards */
    .response-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Top products styling */
    .product-item {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 10px;
        border-left: 3px solid #667eea;
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Rules table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: bold;">
            ✨ Aurora AI – Chatbot de Recomendaciones
        </h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Inteligencia artificial para maximizar tus ventas con recomendaciones personalizadas
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin: 1rem 0; padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px;">
        💡 <strong>Potencia tus ventas</strong> con reglas de asociación inteligentes y análisis predictivo en tiempo real
    </div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">⚙️ Configuración</h2>
    </div>
""", unsafe_allow_html=True)

# OpenAI options if the library is available
if HAS_OPENAI:
    st.sidebar.markdown("### 🤖 Modo de IA")
    mode = st.sidebar.selectbox("Selecciona el modo de respuesta:", 
                               ["🧠 Rule-based (local)", "🤖 LLM (OpenAI)"], 
                               index=0)
    
    if "LLM" in mode:
        st.sidebar.markdown("### 🔑 Configuración OpenAI")
        api_key = st.sidebar.text_input("API Key de OpenAI:", type="password", 
                                       help="Tu clave privada para acceder a OpenAI")
        model_choice = st.sidebar.selectbox("Modelo:", ["gpt-3.5-turbo", "gpt-4o"], index=0)
        st.sidebar.info("💡 El modelo usa SOLO las reglas mostradas en la app")
    else:
        api_key = None
        model_choice = None
else:
    mode = "🧠 Rule-based (local)"
    st.sidebar.warning("⚠️ OpenAI no disponible. Solo modo rule-based.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Datos del Sistema")
st.sidebar.metric("Reglas Activas", "10", "Top performers")
st.sidebar.metric("Productos", "774", "En catálogo")
st.sidebar.metric("Órdenes", "6,042", "Analizadas")

# Show main stats and quick actions with beautiful cards
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📈 Panel de Control")
    
    # Create metrics in a grid layout
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.markdown("""
            <div class="metric-container">
                <div style="color: #667eea; font-size: 2.5rem; margin: 0;">📦</div>
                <div style="font-size: 2rem; font-weight: bold; color: #333; margin: 0.5rem 0;">6,042</div>
                <div style="color: #666; font-size: 0.9rem;">Órdenes Totales</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="metric-container">
                <div style="color: #667eea; font-size: 2.5rem; margin: 0;">🛍️</div>
                <div style="font-size: 2rem; font-weight: bold; color: #333; margin: 0.5rem 0;">14.61</div>
                <div style="color: #666; font-size: 0.9rem;">Items Promedio por Carrito</div>
            </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown("""
            <div class="metric-container">
                <div style="color: #667eea; font-size: 2.5rem; margin: 0;">🎯</div>
                <div style="font-size: 2rem; font-weight: bold; color: #333; margin: 0.5rem 0;">774</div>
                <div style="color: #666; font-size: 0.9rem;">Productos Únicos</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="metric-container">
                <div style="color: #667eea; font-size: 2.5rem; margin: 0;">📊</div>
                <div style="font-size: 2rem; font-weight: bold; color: #333; margin: 0.5rem 0;">88,303</div>
                <div style="color: #666; font-size: 0.9rem;">Items Vendidos</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Top products 
    st.markdown("### 🏆 Top Productos (por frecuencia)")
    for i, (product, support) in enumerate(STATS["top_products_support"]):
        if i == 0:
            icon = "🥇"
        elif i == 1:
            icon = "🥈"
        elif i == 2:
            icon = "🥉"
        else:
            icon = "⭐"
            
        st.markdown(f"""
            <div class="product-item">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <strong>{icon} {product}</strong>
                    </div>
                    <div style="background: white; padding: 0.2rem 0.8rem; border-radius: 15px; 
                         font-weight: bold; color: #667eea;">
                        {support:.1%}
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 🎯 Acciones Rápidas")
    
    if st.button("📋 Ver Todas las Reglas"):
        st.dataframe(rules_df(), height=320)
    
    st.markdown("""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
             padding: 1rem; border-radius: 10px; margin: 1rem 0; text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">💡</div>
            <strong>¿Sabías que?</strong><br>
            Las reglas con Lift > 15 tienen una correlación
            <strong>16x más fuerte</strong> que el promedio
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
             padding: 1rem; border-radius: 10px; margin: 1rem 0; text-align: center;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🚀</div>
            <strong>Oportunidad</strong><br>
            Los bundles identificados pueden aumentar el 
            <strong>ticket promedio hasta un 25%</strong>
        </div>
    """, unsafe_allow_html=True)

# Network graph visualization
if HAS_GRAPH_LIBS:
    st.markdown("### 🔗 Mapa de Conexiones de Productos")
    st.markdown("""
        <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            💡 <strong>Visualización interactiva:</strong> Cada línea representa una relación fuerte entre productos. 
            Los productos más conectados son candidatos ideales para promociones cruzadas.
        </div>
    """, unsafe_allow_html=True)
    
    G = nx.Graph()
    # add nodes and edges derived from RULES
    for r in RULES:
        for a in r["antecedent"]:
            for c in r["consequent"]:
                G.add_node(a)
                G.add_node(c)
                G.add_edge(a, c)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    pos = nx.spring_layout(G, seed=42, k=0.6)
    
    # Draw with beautiful colors
    nx.draw_networkx_nodes(G, pos, node_color='#667eea', node_size=1000, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#a8edea', alpha=0.6, width=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='white', font_weight='bold', ax=ax)
    
    ax.set_title("Red de Productos Relacionados", fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    st.pyplot(fig, use_container_width=True)
else:
    st.markdown("### 🔗 Visualización de Red")
    st.info("💡 La visualización de grafo requiere NetworkX y Matplotlib para mostrar las conexiones entre productos")

st.markdown("---")

# Chat section
st.markdown("""
    <div class="chat-container">
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h2 style="color: white; margin: 0; font-size: 1.8rem;">
                🤖 Consulta a Aurora AI
            </h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
                Pregúntame sobre recomendaciones, bundles o estrategias de venta
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Example prompts
st.markdown("### 💡 Ejemplos de Preguntas")
example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    if st.button("🥚 ¿Qué va bien con Huevos?"):
        st.session_state.example_query = "Si un cliente lleva huevos, ¿qué le recomiendo?"

with example_col2:
    if st.button("🫐 Bundles con Berries"):
        st.session_state.example_query = "¿Qué productos debo vender junto con blueberries y frambuesas?"

with example_col3:
    if st.button("📈 Mejores Oportunidades"):
        st.session_state.example_query = "¿Cuáles son las mejores oportunidades de cross-selling?"

# Query input
query = st.text_area(
    "✍️ Escribe tu pregunta aquí:", 
    height=120,
    placeholder="Ejemplo: '¿Qué productos debo vender juntos para maximizar ventas?' o 'Si un cliente lleva kombucha, ¿qué más le sugiero?'",
    value=st.session_state.get('example_query', '')
)


if 'example_query' in st.session_state:
    del st.session_state.example_query


col_options1, col_options2 = st.columns(2)
with col_options1:
    show_rule_matches = st.checkbox("📊 Mostrar reglas coincidentes", value=True)
with col_options2:
    show_all_rules = st.checkbox("📋 Mostrar tabla completa abajo", value=False)


send_col1, send_col2, send_col3 = st.columns([1, 2, 1])
with send_col2:
    send_button = st.button("🚀 Consultar a Aurora AI", use_container_width=True)

if send_button:
    if not query.strip():
        st.warning("⚠️ Por favor, escribe una pregunta para que pueda ayudarte.")
    else:
        with st.spinner("🧠 Aurora AI está analizando tu consulta..."):
            if "Rule-based" in mode:
                answer, cited = rule_based_answer(query)
                
                st.markdown("""
                    <div class="response-card">
                        <h3 style="color: #667eea; margin-top: 0;">
                            🤖 Respuesta de Aurora AI (Análisis Local)
                        </h3>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style="background: #f8f9ff; padding: 1.5rem; border-radius: 10px; 
                         border-left: 4px solid #667eea; margin: 1rem 0;">
                        {answer.replace('\n', '<br>')}
                    </div>
                """, unsafe_allow_html=True)
                
                if show_rule_matches and cited:
                    st.markdown("### 📋 Reglas Utilizadas")
                    for i, rid in enumerate(cited):
                        r = next((x for x in RULES if x["id"] == rid), None)
                        if r:
                            confidence_color = "#4CAF50" if r["confidence"] > 0.5 else "#FF9800" if r["confidence"] > 0.3 else "#f44336"
                            lift_color = "#4CAF50" if r["lift"] > 10 else "#FF9800" if r["lift"] > 5 else "#f44336"
                            
                            st.markdown(f"""
                                <div style="background: white; padding: 1rem; border-radius: 10px; 
                                     margin: 0.5rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                        <span style="background: #667eea; color: white; padding: 0.2rem 0.8rem; 
                                             border-radius: 15px; font-size: 0.8rem; font-weight: bold;">{r["id"]}</span>
                                        <div style="display: flex; gap: 1rem;">
                                            <span style="background: {confidence_color}; color: white; padding: 0.2rem 0.6rem; 
                                                 border-radius: 10px; font-size: 0.7rem;">Conf: {r["confidence"]:.3f}</span>
                                            <span style="background: {lift_color}; color: white; padding: 0.2rem 0.6rem; 
                                                 border-radius: 10px; font-size: 0.7rem;">Lift: {r["lift"]:.2f}</span>
                                        </div>
                                    </div>
                                    <div style="color: #333; margin: 0.5rem 0;">
                                        <strong>Si:</strong> {' + '.join(r["antecedent"])}<br>
                                        <strong>Entonces:</strong> {' + '.join(r["consequent"])}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
            else:
                if not HAS_OPENAI:
                    st.error("❌ La librería OpenAI no está disponible. Instala 'openai' para usar el modo LLM.")
                elif not api_key:
                    st.error("🔑 Para usar el modo LLM, ingresa tu API key de OpenAI en la barra lateral.")
                else:
                    try:
                        # OpenAI library versions
                        from openai import OpenAI
                        client = OpenAI(api_key=api_key)
                        
                        system_prompt = build_system_prompt()
                        user_prompt = (
                            "Pregunta del usuario (español):\n"
                            + query
                            + "\n\nRespond in Spanish. Use ONLY the rules and stats provided. Cite rule ids used in [RXX] format."
                        )
                        
                        response = client.chat.completions.create(
                            model=model_choice,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=0.2,
                            max_tokens=600,
                        )
                        
                        llm_text = response.choices[0].message.content.strip()
                        
                        # LLM response
                        st.markdown("""
                            <div class="response-card">
                                <h3 style="color: #667eea; margin-top: 0;">
                                    🤖 Respuesta de Aurora AI (Powered by OpenAI)
                                </h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                 color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
                                {llm_text.replace('\n', '<br>')}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Extracting and displaying referenced rules
                        import re
                        refs = re.findall(r"\[R0?\d{1,2}\]", llm_text)
                        refs = [r.strip("[]") for r in refs]
                        if show_rule_matches and refs:
                            st.markdown("### 📊 Reglas Citadas por el Modelo")
                            for rid in refs:
                                r = next((x for x in RULES if x["id"] == rid), None)
                                if r:
                                    st.markdown(f"""
                                        <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; 
                                             border-radius: 10px; margin: 0.5rem 0; border-left: 3px solid #667eea;">
                                            <strong>{format_rule_short(r)}</strong>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                    except Exception as e:
                        st.error(f"❌ Error al conectar con OpenAI: {str(e)}")
                        st.info("💡 Verifica que tu API key sea correcta y tengas créditos disponibles.")

st.markdown("---")

# Rules table section
if show_all_rules:
    st.markdown("### 📋 Tabla Completa de Reglas (Top 10)")
    st.markdown("""
        <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            📊 <strong>Interpretación:</strong> 
            <span style="background: #4CAF50; color: white; padding: 0.2rem 0.5rem; border-radius: 5px; margin: 0 0.2rem;">Lift > 10</span> = Muy fuerte | 
            <span style="background: #FF9800; color: white; padding: 0.2rem 0.5rem; border-radius: 5px; margin: 0 0.2rem;">Lift 5-10</span> = Fuerte | 
            <span style="background: #f44336; color: white; padding: 0.2rem 0.5rem; border-radius: 5px; margin: 0 0.2rem;">Lift < 5</span> = Débil
        </div>
    """, unsafe_allow_html=True)
    
    # Style the dataframe
    styled_df = rules_df().style.format({
        'support': '{:.4f}',
        'confidence': '{:.3f}', 
        'lift': '{:.2f}'
    }).background_gradient(subset=['lift'], cmap='RdYlGn')
    
    st.dataframe(styled_df, height=400, use_container_width=True)

st.markdown("---")
st.markdown("### 📥 Exportar Datos")

col_download1, col_download2, col_download3 = st.columns(3)

with col_download1:
    csv = rules_df().to_csv(index=False).encode("utf-8")
    st.download_button(
        "📊 Descargar Reglas CSV", 
        data=csv, 
        file_name="aurora_ai_rules.csv", 
        mime="text/csv",
        use_container_width=True
    )

with col_download2:
    # Summary report
    summary_text = f"""
Aurora AI - Reporte de Reglas de Asociación

Estadísticas Generales:
- Órdenes analizadas: {STATS['total_orders']:,}
- Productos únicos: {STATS['unique_products']:,}
- Promedio items por carrito: {STATS['avg_basket_size']:.2f}
- Total items vendidos: {STATS['total_items_sold']:,}

Top 5 Productos (por frecuencia):
{chr(10).join([f"- {p}: {s:.1%}" for p, s in STATS['top_products_support']])}

Reglas Top 10:
{chr(10).join([f"- {r['id']}: Lift {r['lift']:.2f}, Conf {r['confidence']:.3f}" for r in RULES])}
    """
    
    st.download_button(
        "📄 Reporte Completo TXT",
        data=summary_text,
        file_name="aurora_ai_report.txt",
        mime="text/plain",
        use_container_width=True
    )

with col_download3:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
             padding: 1rem; border-radius: 10px; text-align: center; height: 60px; 
             display: flex; align-items: center; justify-content: center;">
            <div>
                <div style="font-size: 1.2rem;">📊</div>
                <div style="font-size: 0.8rem; font-weight: bold;">Datos Listos para BI</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
         color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
        <h3 style="margin: 0 0 1rem 0;"> Aurora AI</h3>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div style="text-align: center;">
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; margin: 2rem 0;">
        ℹ️ <strong>Nota:</strong> Las respuestas del modo LLM están limitadas exclusivamente a las reglas y estadísticas 
        mostradas en esta aplicación. El modelo no utiliza conocimiento externo para garantizar precisión y consistencia.
    </div>
""", unsafe_allow_html=True)
