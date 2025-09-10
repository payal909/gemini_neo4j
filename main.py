import streamlit as st
import pandas as pd
from gemineo4j import run_query, get_graph_info, get_graph_schema, get_graph_data, add_document, text_to_response

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

for message in session["chat"]:
    with st.chat_message(message["sender"]):
        st.write(message["message"])
        if message["sender"] == "ai":
            st.code(message["query"],language="cypher")

with st.sidebar:    

    setup_button = st.button("Setup", use_container_width=True)

    with st.form("File"):
        uploaded_file = st.file_uploader("Upload a file", accept_multiple_files=False)
        submit = st.form_submit_button("Submit")
    
    with st.expander("Graph Info"):
        st.markdown(f"- Node count: {session['node_count']}\n- Relation count: {session['relation_count']}")

    if submit and uploaded_file:
        status = st.status("Adding document...")
        document = pd.read_csv(uploaded_file).to_csv(index=False)        
        session["schema"], session["data"] , session["updated_schema"], session["updated_data"] = add_document(status, document)
        session["node_count"], session["relation_count"] = get_graph_info()
        
        with st.expander("New Schema"):
            st.json(session["updated_schema"])
            st.json(session["updated_data"])

if setup_button:
    setup_db(st.status("Setting up database..."))
        
prompt = st.chat_input("Ask a query...")
if prompt:
    
    with st.chat_message("user"):
        st.write(prompt)
    session["chat"].append({"sender": "user","message": prompt})
    
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            query, ai_response = text_to_response(session["schema"], session["data"], prompt)
            query = "// Cypher Query\n" + query
        st.write(ai_response)
        st.code(query,language="cypher")
    session["chat"].append({"sender": "ai","message": ai_response, "query": query})
