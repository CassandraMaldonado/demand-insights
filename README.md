# Demand Insights & Recommendation Engine

- **Reglas de asociación (Apriori)** generadas a partir de históricos de órdenes.  
- **Estadísticas de ventas** (órdenes totales, productos más vendidos, tamaño promedio del carrito).  
- **Interfaz en Streamlit** para explorar reglas, visualizar redes de productos y consultar recomendaciones.  
- **Modo híbrido**:  
  - **Rule-based (local)** → respuestas determinísticas, basadas solo en las reglas.  
  - **LLM con OpenAI** → lenguaje natural más flexible, pero **limitado estrictamente a las reglas cargadas**.  

---

## 🚀 Características

- **Top-10 reglas de asociación** precargadas con *support*, *confidence* y *lift*.  
- **Visualización de red (NetworkX)** para identificar clusters de productos relacionados.  
- **Chat interactivo**:  
  - Puedes preguntar en español:  
    - “Si un cliente compra Huevos, ¿qué le recomiendo?”  
    - “¿Qué bundles puedo armar con Kombucha?”  
