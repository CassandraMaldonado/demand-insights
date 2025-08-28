# Demand Insights & Recommendation Engine

- Association rules (Apriori) mined from historical orders.  
- Sales statistics (total orders, top-selling products, average basket size).  
- Streamlit interface to explore rules, visualize product networks, and query recommendations.  
- Hybrid mode:  
  - Rule-based (local) -> deterministic answers, only using preloaded rules.  
  - LLM with OpenAI -> natural language answers, but strictly grounded in the loaded rules.  
