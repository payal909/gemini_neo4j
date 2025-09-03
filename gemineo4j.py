import os
import re
import json
import pandas as pd
import openpyxl
from neo4j import GraphDatabase
from google import genai
from google.genai import types
from pydantic import BaseModel
from pydantic.types import conlist
from typing import Union
from dotenv import load_dotenv
import warnings
import streamlit as st

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

_ = load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

URI = os.environ.get("NEO4J_URI")
AUTH = (os.environ.get("NEO4J_USERNAME"),os.environ.get("NEO4J_PASSWORD"))

os.environ["GEMINI_MODEL"] = [
    # "gemini-2.0-flash-thinking-exp-01-21",
    # "gemini-2.0-flash-lite",
    # "gemini-2.0-flash",
    # "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
][0]


def remove_duplicates(schema):
    schema_dict ={f"{record["snt"]}-{record["rt"]}-{record["tnt"]}": record for record in schema}
    new_schema = [schema_dict[key] for key in set(schema_dict.keys())]
    return schema

def run_query(query, mode="READ"):
    URI = os.environ.get("NEO4J_URI")
    AUTH = (os.environ.get("NEO4J_USERNAME"),os.environ.get("NEO4J_PASSWORD"))
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session(database = os.environ.get("NEO4J_DATABASE"),default_access_mode=mode) as session:
            result = session.run(query).data()        
    return result

def get_graph_info():
    node_count = run_query("MATCH (n) RETURN COUNT(n) AS _")[0]["_"]
    relation_count = run_query("MATCH (n)-[r]->(m) RETURN COUNT(r) AS _")[0]["_"]
    return node_count, relation_count
    
def get_graph_schema():
    URI = os.environ.get("NEO4J_URI")
    AUTH = (os.environ.get("NEO4J_USERNAME"),os.environ.get("NEO4J_PASSWORD"))
    records = []
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session(database = os.environ.get("NEO4J_DATABASE"),default_access_mode="READ") as session:
            result = session.run("MATCH (n)-[r]->(m) RETURN LABELS(n) AS sn,type(r) AS r, LABELS(m) AS tn")
            for record in result:
                records.append(record.data())

    all_relationships = {
        f"{record['sn'][0]}-{record['r']}-{record['tn'][0]}": {
            "snt": record['sn'][0],
            "rt": record['r'],
            "tnt": record['tn'][0],
            } for record in records
        }

    schema = [all_relationships[key] for key in set(all_relationships.keys())]
    return schema

def get_graph_data():
    URI = os.environ.get("NEO4J_URI")
    AUTH = (os.environ.get("NEO4J_USERNAME"),os.environ.get("NEO4J_PASSWORD"))
    records = []
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session(database = os.environ.get("NEO4J_DATABASE"),default_access_mode="READ") as session:
            records = session.run("MATCH (s)-[r]->(t) RETURN LABELS(s) AS sn,type(r) AS r,LABELS(t) AS tn,s,t").data()
    
    all_nodes = {f"{record['sn']}-{record['s']}": (record["sn"],record["s"]) for record in records}

    data = [all_nodes[key] for key in set(all_nodes.keys())]
    return data
    
# class Schema(BaseModel):
#     snt: str
#     rt: str
#     tnt: str

schema_response_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "snt": {"type": "string"},
            "rt": {"type": "string"},
            "tnt": {"type": "string"},
        }
        }
}

# class Property(BaseModel):
#     k: str
#     v: Union[str, float]

# class Relation(BaseModel):
#     t: str
#     p: conlist(Property, min_length=0, max_length=2)

# class Node(BaseModel):
#     t: str
#     n: str
#     p: conlist(Property, min_length=0, max_length=2)

# class Data(BaseModel):
#     sn: Node
#     r: Relation
#     tn: Node

data_response_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "sn": {"type": "object", "properties": {"t": {"type": "string"}, "n": {"type": "string"}, "p": {"type": "array", "minItems": 0, "maxItems": 2, "items": {"type": "object", "properties": {"k": {"type": "string"}, "v": {"type": "string"}}}}}},
            "r": {"type": "object", "properties": {"t": {"type": "string"}, "p": {"type": "array", "minItems": 0, "maxItems": 2, "items": {"type": "object", "properties": {"k": {"type": "string"}, "v": {"type": "string"}}}}}},
            "tn": {"type": "object", "properties": {"t": {"type": "string"}, "n": {"type": "string"}, "p": {"type": "array", "minItems": 0, "maxItems": 2, "items": {"type": "object", "properties": {"k": {"type": "string"}, "v": {"type": "string"}}}}}},
        }
    }
}

abbreviation = """
    snt = source node type
    rt = relation type
    tnt target node type
    k = key
    v = value
    rp = relation properties
    t = type
    n = name
    p = properties
    sn = source node
    r = relation
    tn = target node
    """

def update_schema(schema, document):

    schema_system_instruction = f"""
    You are an expert at extracting knowledge from various data sources. 
    Given a new data source, which can be structured (like a CSV, JSON, or database schema) or unstructured (like plain text or a document).
    You will be provided with the existing/current schema of the knowledge graph and are tasked to provide new additions on top of the existing schema.
    If existing Schema is blank assume no relationships exist and you are allowed to make new nodes and relationships, otherwise dont create duplicates of existing nodes and relationships
    For each identified relationship, you should:
    Name the Nodes and Relationship: Provide a clear, concise name for the nodes and relationship.
    Identify the Entities: Specify the two entities (subject and object) that are connected by this relationship.
    Do not give relationsships both ways i.e. if ['Food','CONTAINS','Ingredient'] is present then do not include ['Ingredient','CONTAINED_IN','Food'].
    Keep the relationship name straightforward, fundamental and genereic. USe verb or verb like short phrases.
    Do to incliude target node type in relationship e.g. ['Food','CONTAINS','Ingredient'] is correct but ['Food','CONTAINS_INGREDIENT','Ingredient'] is not correct.
    This is a crucial step in building a knowledge graph, so be thorough and pay close attention to the details. 
    Only return the additonal list of nodes and relationships and not the existing schema provided.
    Use the following case ['Title_Case','UPPER_CASE,'Title_Case']
    Here are the abbreviation: {abbreviation}
    """
        
    schema_config = types.GenerateContentConfig(
        temperature=0,
        system_instruction=schema_system_instruction,
        # tools=[{"url_context": {}}],
        thinking_config = types.ThinkingConfig(thinking_budget=1024),
        response_mime_type = "application/json",
        response_schema = schema_response_schema
        )

    schema_response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL"),
        contents=[f"Existing schema {json.dumps(schema)}", f"New data source {document}"],
        config=schema_config,
        )

    new_schema = json.loads(schema_response.text)
    new_schema = [{key: re.sub(r'[^a-zA-Z0-9 ]', '', value).replace(" ","_") for key, value in record.items()} for record in new_schema]
    new_schema = remove_duplicates(schema + new_schema)
    return new_schema

def update_data(schema, data, document):

    data_system_instruction = f"""
    You are an expert at extracting knowledge from various data sources. 
    Given a new data source, which can be structured (like a CSV, JSON, or database schema) or unstructured (like plain text or a document).
    You will be provided with the existing schema and dict of nodes and relationships of the knowledge graph and are tasked to provide all possible new nodes and relatiohships with respect to the existing schema.
    If existing list of nodes is blank assume no nodes and relationships exist and you are allowed to make new nodes and relationships, otherwise dont create duplicates of existing nodes and relationships.
    For each identified relationship, you should:
    Name the Nodes and Relationship: Provide a clear, concise name for the nodes and relationship.
    Identify the Entities: Specify the two entities (subject and object) that are connected by this relationship.
    Only use values provided in existing schema for snts, rts and tnts, values for source_node_name, target_node_name are to be extracted from the document provided.
    This is a crucial step in building a knowledge graph, so be thorough and pay close attention to the details. 
    Only return the additonal/new nodes and relationships and not the existing nodes and relationships provided.
    Here are the abbreviation: {abbreviation}
    """
        
    data_config = types.GenerateContentConfig(
        temperature=0,
        system_instruction=data_system_instruction,
        # tools=[{"url_context": {}}],
        thinking_config = types.ThinkingConfig(thinking_budget=1024),
        max_output_tokens=65536,
        response_mime_type = "application/json",
        response_schema = data_response_schema
        )

    data_response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL"),
        contents=[f"Existing schema {json.dumps(schema)}", f"Existing nodes {json.dumps(data)}",f"New data source {document}"],
        config=data_config,
        )

    new_data = json.loads(data_response.text)
    return new_data

def update_graph(data):
    URI = os.environ.get("NEO4J_URI")
    AUTH = (os.environ.get("NEO4J_USERNAME"),os.environ.get("NEO4J_PASSWORD"))
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session(database = os.environ.get("NEO4J_DATABASE"),default_access_mode="WRITE") as session:
            for record in data:
                snt = record["sn"]["t"]
                snp = {re.sub(r'[^a-zA-Z0-9 ]', '', prop["k"]).replace(" ","_"): f"'{prop['v']}'" for prop in record["sn"].get("p",[])}
                snp["name"] = f"'{re.sub(r'[^a-zA-Z0-9 ]', '', record['sn']['n'])}'"
                snp = json.dumps(snp).replace('"',"")

                rt = record["r"]["t"]
                rp = {re.sub(r'[^a-zA-Z0-9 ]', '', prop["k"]).replace(" ","_"): f"'{prop['v']}'" for prop in record["r"].get("p",[])}
                rp = json.dumps(rp).replace('"',"")

                tnt = record["tn"]["t"]
                tnp = {re.sub(r'[^a-zA-Z0-9 ]', '', prop["k"]).replace(" ","_"): f"'{prop['v']}'" for prop in record["tn"].get("p",[])}
                tnp["name"] = f"'{re.sub(r'[^a-zA-Z0-9 ]', '', record['tn']['n'])}'"
                tnp = json.dumps(tnp).replace('"',"")

                _ = session.run(f"""
                MERGE (s:{snt} {snp})
                MERGE (t:{tnt} {tnp})
                MERGE (s)-[:{rt} {rp}]->(t)
                """)

def add_document(status, document, schema=[]):
    URI = os.environ.get("NEO4J_URI")
    AUTH = (os.environ.get("NEO4J_USERNAME"),os.environ.get("NEO4J_PASSWORD"))
    with status:
        st.markdown("- Fetching existing graph schema")
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            driver.verify_connectivity()
            with driver.session(database = os.environ.get("NEO4J_DATABASE"),default_access_mode="READ") as session:
                old_nodes = session.run("MATCH (s) RETURN COUNT(s) AS _").data()[0]["_"]
                old_relations = session.run("MATCH ()-[r]->() RETURN COUNT(r) AS _").data()[0]["_"]
        existing_schema = get_graph_schema() 
        
        st.markdown("- Fetching existing graph data")
        existing_data = get_graph_data()

        st.markdown("- Extarcting new schema from document")
        if len(schema)==0:
            new_schema = update_schema(existing_schema, document)
        else:
            new_schema = remove_duplicates(schema + existing_schema)

        st.markdown("- Extarcting new data from document")            
        new_data = update_data(new_schema, existing_data, document)

        st.markdown("- Updating graph")
        update_graph(new_data)

        st.markdown("- Fetching updated graph schema")
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            driver.verify_connectivity()
            with driver.session(database = os.environ.get("NEO4J_DATABASE"),default_access_mode="READ") as session:
                new_nodes = session.run("MATCH (s) RETURN COUNT(s) AS _").data()[0]["_"]
                new_relations = session.run("MATCH ()-[r]->() RETURN COUNT(r) AS _").data()[0]["_"]

        st.markdown(f"- Updated node count from {old_nodes} to {new_nodes}. Updated relations count from {old_relations} to {new_relations}")
        
        new_schema = get_graph_schema()
        new_data = get_graph_data()
        updated_schema = [record for record in new_schema if record not in existing_schema]
        updated_data = [record for record in new_data if record not in existing_data]
        return new_schema, new_data, updated_schema, updated_data

def text_to_cypher(schema, data, text):

    cypher_system_instruction = """
    You are an expert at converting engligh based questions into neo4j cypher querie.
    The graph schema and list os nodes ([node_type, node_name]) is also provided for extra context.
    You will be provided with the existing schema of the neo4j knowlwdge graph your job is to return a cypher query that helps answering the question.
    If the question is not clear or the graph does not have the required data to answer the question return a default cypher query that give all node types."""
        
    cypher_config = types.GenerateContentConfig(
        temperature=0,
        system_instruction=cypher_system_instruction,
        # tools=[{"url_context": {}}],
        thinking_config = types.ThinkingConfig(thinking_budget=1024),
        response_mime_type = "application/json",
        response_schema = {"type": "string", "nullable": False},
        )

    cypher_response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL"),
        contents=[
            f"User question: {text}",
            f"Existing schema {json.dumps(schema)}",
            f"List of nodes: {json.dumps(data)}"
            ],
        config=cypher_config,
        )
        
    return json.loads(cypher_response.text)

def text_to_response(schema, data, text):
    cypher_query = text_to_cypher(schema, data, text)
    result = run_query(cypher_query)
    result = json.dumps(result)

    response_system_instruction = """
    You are given the user question about a neo4j knowledge graph along with the cypher query and result of running the cypher query.
    The graph schema and list os nodes ([node_type, node_name]) is also provided for extra context.
    Your task is to formulate a response considering the userquestion cypher query and the result of the cypher query.
    If the question is not clear or the graph does not have the required data to answer the question,
    use the provided data and let user know that you can anser question only about the schema and nodes provided."""
        
    response_config = types.GenerateContentConfig(
        temperature=0,
        system_instruction=response_system_instruction,
        # tools=[{"url_context": {}}],
        thinking_config = types.ThinkingConfig(thinking_budget=1024),
        )

    response_response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL"),
        contents=[
            f"User question: {text}",
            f"Cypher query: {cypher_query}",
            f"Cypher query result: {result}",
            f"Existing schema {json.dumps(schema)}",
            f"List of nodes: {json.dumps(data)}"
            ],
        config=response_config,
        )
        
    return cypher_query, response_response.text
