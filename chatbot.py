# app.py
"""
Streamlit app: Recomendador / Chatbot basado en reglas de asociación.
- Modo local (rule-based) para respuestas rápidas y reproducibles.
- Modo LLM (usa la API de OpenAI con la API key que el usuario pega).
"""

import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import openai
import textwrap

st.set_page_config(page_title="Aurora AI - Recomendador (Rules)", layout="wide")

# -------------------------
# 1) Reglas y estadísticas (datos provistos)
# -------------------------
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

# -------------------------
# 2) Helpers
# -------------------------
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

# Simple local rule-based answer generator (Spanish)
def rule_based_answer(question):
    products = set(find_products_in_text(question))
    if not products:
        # No product mentioned -> provide top suggestions / bundles
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

# Build system prompt for LLM mode (serializa reglas)
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

# -------------------------
# 3) UI
# -------------------------
st.title("Aurora AI – Chatbot de Recomendaciones (Reglas pre-cargadas)")
st.markdown(
    "Esta aplicación usa las reglas de asociación pre-cargadas (Top-10) y estadísticas del catálogo. "
    "Puedes usar el modo local (rule-based) o usar tu propia clave de OpenAI para que el modelo formule respuestas en lenguaje natural pero **limitadas** a estas reglas."
)

# Sidebar
st.sidebar.header("Configuración")
api_key = st.sidebar.text_input("Pega tu OpenAI API key (si usarás modo LLM)", type="password")
model_choice = st.sidebar.selectbox("Modelo OpenAI (si usas LLM)", ["gpt-3.5-turbo", "gpt-4"], index=0)
mode = st.sidebar.radio("Modo de respuesta", ["Rule-based (local)", "LLM (OpenAI)"])
st.sidebar.markdown("Modo LLM: el texto enviado a OpenAI incluye SOLO las reglas y estadísticas mostradas abajo; el modelo debe citar reglas usadas.")

# Show main stats and quick actions
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Estadísticas rápidas")
    st.write(f"- Órdenes totales: **{STATS['total_orders']}**")
    st.write(f"- Productos únicos: **{STATS['unique_products']}**")
    st.write(f"- Tamaño promedio del carrito: **{STATS['avg_basket_size']}** items")
    st.write(f"- Total productos vendidos: **{STATS['total_items_sold']}**")
    st.write("Top productos (por soporte):")
    for p, s in STATS["top_products_support"]:
        st.write(f"- {p} → soporte {s:.3f}")
with col2:
    st.subheader("Acciones rápidas")
    if st.button("Mostrar reglas principales"):
        st.dataframe(rules_df(), height=320)

# network graph visualization (simple)
st.subheader("Mapa conceptual: clusters de productos (visualización)")
G = nx.Graph()
# add nodes and edges derived from RULES (simplified: each antecedent and consequent added, connect antecedent->consequent)
for r in RULES:
    for a in r["antecedent"]:
        for c in r["consequent"]:
            G.add_node(a)
            G.add_node(c)
            G.add_edge(a, c)
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42, k=0.4)
nx.draw(G, pos, with_labels=True, node_color="#88ccff", node_size=800, font_size=8, edge_color="#666666")
st.pyplot(plt.gcf())

st.markdown("---")
st.subheader("Chat / Consulta")
query = st.text_area("Escribe tu pregunta (ej: '¿Qué productos debo vender juntos?' o 'Si un cliente lleva Huevos, ¿qué le recomiendo?')", height=120)

col_a, col_b = st.columns([3,1])
with col_b:
    st.write("Opciones")
    show_rule_matches = st.checkbox("Mostrar reglas coincidentes (si las hay)", value=True)
    show_all_rules = st.checkbox("Mostrar tabla completa de reglas abajo", value=False)

if st.button("Enviar pregunta"):
    if not query.strip():
        st.warning("Escribe una pregunta o un ejemplo de carrito.")
    else:
        if mode == "Rule-based (local)":
            answer, cited = rule_based_answer(query)
            st.markdown("### Respuesta (Rule-based)")
            st.write(answer)
            if show_rule_matches and cited:
                st.markdown("**Reglas citadas:**")
                for rid in cited:
                    r = next((x for x in RULES if x["id"] == rid), None)
                    if r:
                        st.write(format_rule_short(r))
        else:
            # LLM mode -> need API key
            if not api_key:
                st.error("Para usar el modo LLM pega tu OpenAI API key en la barra lateral.")
            else:
                with st.spinner("Consultando OpenAI..."):
                    try:
                        openai.api_key = api_key
                        system_prompt = build_system_prompt()
                        user_prompt = (
                            "Pregunta del usuario (español):\n"
                            + query
                            + "\n\nRespond in Spanish. Use ONLY the rules and stats provided. Cite rule ids used in [RXX] format."
                        )
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                        resp = openai.ChatCompletion.create(
                            model=model_choice,
                            messages=messages,
                            temperature=0.2,
                            max_tokens=600,
                        )
                        llm_text = resp["choices"][0]["message"]["content"].strip()
                        st.markdown("### Respuesta (LLM usando reglas)")
                        st.write(llm_text)
                        # Attempt to extract referenced rules from the response (look for [Rxx])
                        import re
                        refs = re.findall(r"\[R0?\d{1,2}\]", llm_text)
                        refs = [r.strip("[]") for r in refs]
                        if show_rule_matches and refs:
                            st.markdown("**Reglas citadas por el modelo:**")
                            for rid in refs:
                                r = next((x for x in RULES if x["id"] == rid), None)
                                if r:
                                    st.write(format_rule_short(r))
                    except Exception as e:
                        st.error(f"Error llamando a OpenAI: {e}")

st.markdown("---")
if show_all_rules:
    st.subheader("Tabla completa de reglas (Top 10)")
    st.dataframe(rules_df(), height=320)

st.markdown("## Descarga / Export")
csv = rules_df().to_csv(index=False).encode("utf-8")
st.download_button("Descargar reglas (CSV)", data=csv, file_name="rules_top10.csv", mime="text/csv")

st.markdown("---")
st.caption("Nota: Las respuestas LLM dependen del modelo y la calidad de la prompt/clave. El prompt enviado a OpenAI solo contiene las reglas y estadísticas mostradas arriba; el modelo está instruido a no usar conocimiento externo.")
