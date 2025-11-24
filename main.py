import streamlit as st
import pandas as pd
from gemineo4j import (write_chat, run_query, get_graph_info, get_graph_schema, get_graph_data, add_document, text_to_response)

st.set_page_config(page_title="NutriGraph", page_icon=":material/graph_3:", layout="wide", initial_sidebar_state="expanded")

session = st.session_state

if "chat" not in session:
    session["chat"] = []
if "schema" not in session:
    session["schema"] = get_graph_schema()
if "data" not in session:
    session["data"] = get_graph_data()
if "updated_schema" not in session:
    session["updated_schema"] = []
if "updated_data" not in session:
    session["updated_data"] = []
if ("node_count" not in session) or ("relation_count" not in session):
    session["node_count"], session["relation_count"] = get_graph_info()
if "setup_schema" not in session:
    session["setup_schema"] = """(Food_Item)-[HAS]->(Food_Variation)
(Food_Variation)-[CONTAINS]->(Ingredient)
(Ingredients)-[CARRY]->(Allergen)
(Food_Variation)-[CONTAINS]->(Nutrient)
(Food_Variation)-[AVAILABLE_IN]->(Region)"""

for message in session["chat"]:
    write_chat(message)
        
sidebar = st.sidebar 

sidebar.title(":material/graph_3: NutriGraph")

st.caption(
    """<style>
            MainMenu {visibility:hidden;}
            header {visibility:hidden;}
            footer {visibility:hidden;}
            body {overflow: hidden;}
            section[data-testid="stSidebar"] {min-width: 500px; max-width: 500px;}
            div[data-testid="stSidebarHeader"] {display: none}
            div[data-testid="stMainBlockContainer"] {padding: 2.5rem}
            div[data-testid="stSidebarCollapseButton"] {display: none;}
        </style>""",
    unsafe_allow_html=True,
)


with sidebar:
    
    with st.form("File"):
        uploaded_file = st.file_uploader("Upload a file", accept_multiple_files=False)
        schema = st.text_area("Schema", value=session["setup_schema"], height=175)
        submit = st.form_submit_button("Submit", icon=":material/send:", use_container_width=True)
    
    with st.expander("Graph Info"):
        st.markdown(f"- Node count: {session['node_count']}\n- Relation count: {session['relation_count']}")
        clear_graph = st.button("Clear Graph", icon=":material/warning:", use_container_width=True)

if clear_graph:
    run_query("MATCH (n) DETACH DELETE n", mode="WRITE")
    st.rerun()
    

if submit and uploaded_file:
    with st.spinner("Adding document...", show_time=True):
        session["setup_schema"] = None
        submit_progress = st.progress(0, text="Adding document...")
        submit_warning = st.empty()
        submit_warning.warning("This may take a few minutes...", icon="⚠️")
        document = pd.read_csv(uploaded_file).to_csv(index=False)        
        session["schema"], session["data"] , session["updated_schema"], session["updated_data"] = add_document(submit_progress, document, schema)
        session["node_count"], session["relation_count"] = get_graph_info()
        submit_warning.success("You can now continue with the chat...",  icon="✅")
    
prompt = st.chat_input("Ask a query...")

if prompt:
    message = {"sender": "user","message": prompt}
    write_chat(message)
    session["chat"].append(message)
    
    with st.chat_message("ai", avatar=":material/smart_toy:"):
        with st.spinner("Thinking...", show_time=True):
            query, ai_response = text_to_response(session["schema"], session["data"], prompt)
            query = "// Cypher Query\n" + query
        st.info(ai_response)
        st.code(query,language="cypher")
    session["chat"].append({"sender": "ai","message": ai_response, "query": query})

if st.secrets["DEV_ENV"]:
    sidebar.json(session, expanded=False)