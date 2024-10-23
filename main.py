import streamlit as st
from streamlit_chat import message
from timeit import default_timer as timer

from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI

# import dotenv
import os

# dotenv.load_dotenv()

# OpenAI API configuration
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key= os.environ["OPENAI_API_KEY"]  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

#Neo4j configuration
# neo4j_url = os.getenv("NEO4J_CONNECTION_URL")
# neo4j_user = os.getenv("NEO4J_USER")
# neo4j_password = os.getenv("NEO4J_PASSWORD")

neo4j_url = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "12345678"

# cypher_generation_template = """
# You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided, following the instructions below:
# 1. Generate Cypher query compatible ONLY for Neo4j Version 5
# 2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use alias when using the WITH keyword
# 3. Use only nodes and relationships mentioned in the schema
# 4. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Person named John, use `toLower(entity.name) contains 'john'`. 
# 5. Never use relationships that are not mentioned in the given schema
# 6. When asked about entities, Match the properties using case-insensitive matching, E.g, to find a person named John, use `toLower(entity.name) contains 'john'`.
# 7. When asked about a person, Match the label property with the word "person", E.g, to find a person, use `toLower(entity.label) = 'person'`.
# 8. When asked about a place, Match the label property with the word "place", E.g, to find a place, use `toLower(entity.label) = 'place'`.
# 9. When asked about a object, Match the label property with the word "object", E.g, to find an object, use `toLower(entity.label) = 'object'`.
# 10. When asked about a event, Match the label property with the word "event", E.g, to find an event, use `toLower(entity.label) = 'event'`.
# 11. When asked about a miscellaneous entity, Match the label property with the word "miscellaneous", E.g, to find a miscellaneous entity, use `toLower(entity.label) = 'miscellaneous'`.
# 12. If a person, place, object, event or a miscellaneous entity does not match an entity in the graph, Try matching the description property or the metadata property of a relationship using case-insensitive matching, E.g, to find information about Joe, use toLower(r.description) contains 'joe' OR toLower(r.metadata) contains 'joe'.
# 13. When asked about any information of an entity, Do not simply give the entity label. Try to get the answer from the entity's relationship description or metadata property

# schema: {schema}

# Question: {question}
# """

# Cypher generation prompt
cypher_generation_template = """
You are an expert Neo4j Cypher translator who converts English to Cypher based on the Neo4j Schema provided, following the instructions below:
1. Generate Cypher query compatible ONLY for Neo4j Version 5
2. Do not use EXISTS, SIZE, HAVING keywords in the cypher. Use alias when using the WITH keyword
3. Use only nodes and relationships mentioned in the schema
4. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Person named John, use `toLower(entity.name) contains 'john'`. 
5. Never use relationships that are not mentioned in the given schema
6. When asked about entities, Match the properties using case-insensitive matching, E.g, to find a person named John, use `toLower(entity.name) contains 'john'`.
7. If a person, place, object, event or a miscellaneous entity does not match an entity in the graph, Try matching the description property or the metadata property of a relationship using case-insensitive matching, E.g, to find information about Joe, use toLower(r.description) contains 'joe' OR toLower(r.metadata) contains 'joe'.
8. When asked about any information of an entity, Do not simply give the entity label. Try to get the answer from the entity's relationship description or metadata property
9. Use regex to help find an entity that contains multiple words, E.g, to find a the 'Notre Dame Main Building', use toLower(e.name) =~ '.*\\\\b(not|notre|dame|main|building)\\\\b.*'
10. When using MATCH traverse the relationship in both directions, E.g, (e:Entity)-[r:RELATED]-(re:Entity)

schema: {schema}

Examples:
Question: Who is John?
Answer:
MATCH (e:Entity)-[r:RELATED]-(re:Entity)
WHERE toLower(r.description) CONTAINS 'john'
OR toLower(r.metadata) CONTAINS 'john'
RETURN e.name, r.metadata, r.description, re.name

Question: Where is Manila?
Answer: 
MATCH (e:Entity)-[r:RELATED]-(re:Entity)
WHERE toLower(e.name) = 'manila'
RETURN e.name, r.metadata, r.description, re.name

Question: List all the places mentioned in the document
Answer: 
MATCH (e:Entity)
WHERE e.label = place;
RETURN e

Question: Describe the Notre Dame main building
MATCH (e:Entity)-[r:RELATED]-(re:Entity)
WHERE toLower(e.name) =~ '.*\\\\b(not|notre|dame|main|building)\\\\b.*'
RETURN e.name, r.metadata, r.description, re.name

Question: {question}
# """

cypher_prompt = PromptTemplate(
    template = cypher_generation_template,
    input_variables = ["schema", "question"]
)

CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
If the provided information is empty, say that you don't know the answer.
Final answer should be easily readable and structured.
Information:
{context}

Question: {question}
Helpful Answer:"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)

def query_graph(user_input):
    graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt
        )
    result = chain(user_input)
    return result

def refine_query(previous_query, user_input):

   cypher_refine_template = f"""" 
   Context: I am working with a Neo4j database containing information.

   Initial Cypher Query:
      {previous_query}
   """

   cypher_refine_template += """
   Problem:
   The above Cypher Query returned no results. 
   I need to refine this query to achieve to answer the question "{question}": 

   Schema: {schema}

   Request:
   Can you please refine the Initial Cypher Query to answer the question?
   """
    
   cypher_refine_prompt = PromptTemplate(
      input_variables=["schema", "question"], template=cypher_refine_template
   )

   graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)
   chain = GraphCypherQAChain.from_llm(
       llm=llm,
       graph=graph,
       verbose=True,
       return_intermediate_steps=True,
       cypher_prompt=cypher_refine_prompt
   )
   
   print(cypher_refine_prompt.format(question=user_input, schema=graph.schema, prevquery=previous_query))
   result = chain(user_input)
   return result


st.set_page_config(layout="wide")

if "user_msgs" not in st.session_state:
    st.session_state.user_msgs = []
if "system_msgs" not in st.session_state:
    st.session_state.system_msgs = []

title_col, empty_col, img_col = st.columns([2, 1, 2])    

with title_col:
    st.title("Conversational Neo4J Assistant")
with img_col:
    st.image("https://dist.neo4j.com/wp-content/uploads/20210423062553/neo4j-social-share-21.png", width=200)

user_input = st.text_input("Enter your question", key="input")
if user_input:
    with st.spinner("Processing your question..."):
        st.session_state.user_msgs.append(user_input)
        start = timer()

        print(user_input)
        try:
            result = query_graph(user_input)
            intermediate_steps = result["intermediate_steps"]
            cypher_query = intermediate_steps[0]["query"]
            database_results = intermediate_steps[1]["context"]
            answer = result["result"]   

            if answer == "I don't know the answer.":
                result = refine_query(cypher_query[6:], user_input)
                intermediate_steps = result["intermediate_steps"]
                cypher_query = intermediate_steps[0]["query"]
                database_results = intermediate_steps[1]["context"]
                answer = result["result"]   
            
            st.session_state.system_msgs.append(answer)
            # else:
            #     st.session_state.system_msgs.append(answer)
        except Exception as e:
            st.write("Failed to process question. Please try again.")
            print(e)

    st.write(f"Time taken: {timer() - start:.2f}s")

    col1, col2, col3 = st.columns([1, 1, 1])

    # Display the chat history
    with col1:
        if st.session_state["system_msgs"]:
            for i in range(len(st.session_state["system_msgs"]) - 1, -1, -1):
                message(st.session_state["system_msgs"][i], key = str(i) + "_assistant")
                message(st.session_state["user_msgs"][i], is_user=True, key=str(i) + "_user")

    with col2:
        if cypher_query:
            st.text_area("Last Cypher Query", cypher_query, key="_cypher", height=240)
        
    with col3:
        if database_results:
            st.text_area("Last Database Results", database_results, key="_database", height=240)
    