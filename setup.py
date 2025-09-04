import streamlit as st
import pandas as pd
from gemineo4j import run_query, get_graph_info, get_graph_schema, get_graph_data, add_document, text_to_response

status = st.status("test")

document = pd.read_csv("data/Sample Food Nutrition Contents(Nutrition).csv").to_csv(index=False)

base_schema = [
    {"snt": "Food_Variation",   "rt": "CONTAINS",       "tnt": "Ingredient"},
    {"snt": "Food_Variation",   "rt": "CONTAINS",       "tnt": "Nutrient"},
    {"snt": "Ingredients",      "rt": "CONTAINS",       "tnt": "Allergen"},
    {"snt": "Food_Item",        "rt": "HAS",            "tnt": "Food_Variation"},
    {"snt": "Food_Item",        "rt": "AVAILABLE_IN",   "tnt": "Region"}
    ]

add_document(status, document, base_schema)

node_count, relation_count = get_graph_info()
print(f"Node count: {node_count}\nRelation count: {relation_count}")