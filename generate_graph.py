import os
from knowledge_graph_maker import GraphMaker, Ontology, OpenAIClient
from knowledge_graph_maker import Document
import datetime
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
from typing import Optional, List
from knowledge_graph_maker import Neo4jGraphModel

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = "TEST"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"
os.environ["NEO4J_URI"]= "bolt://localhost:7687"

# Pydantic data class
class Sentences(BaseModel):
    sentences: List[str]

def get_propositions(text: str, proposition_list: List[str]):
    obj = hub.pull("wfh/proposal-indexing")

    chunking_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # use it in a runnable
    runnable = obj | chunking_llm

    # Extraction
    structured_llm = chunking_llm.with_structured_output(Sentences)

    text = text.split(".")

    for t in text:
        
        if t:
            print("t:",t)
            runnable_output = runnable.invoke({
                "input": t
            }).content
        
            propositions = structured_llm.invoke(runnable_output).sentences
            proposition_list.extend(propositions)

    return proposition_list


def generate_summary(text: str, llm: OpenAIClient):
    SYS_PROMPT = (
        "Succintly summarise the text provided by the user. "
        "Respond only with the summary and no other comments"
    )
    try:
        summary = llm.generate(user_message=text, system_message=SYS_PROMPT)
    except:
        summary = ""
    finally:
        return summary

def generate(proposition_list: List[str]):

    ontology = Ontology(
    labels=[
        {"Person": "Person name without any adjectives, Remember a person may be referenced by their name or using a pronoun"},
        {"Object": "Objects are inanimate things that a person uses, Do not add the definite article 'the' in the object name"},
        {"Event": "An entity that happens at a specific time and place"},
        {"Place": "Places are locations where specific events took place and where persons can go to and where objects can be found"},
        {"Miscellaneous": "Any important concept can not be categorised with any other given label"},
    ],
    relationships=[
        "Relation between any pair of Entities"
        ],
    )

    ## Open AI models
    oai_model="gpt-4o-mini"

    ## OR Use OpenAI
    llm = OpenAIClient(model=oai_model, temperature=0.1, top_p=0.5)

    current_time = str(datetime.datetime.now())

    docs = map(
    lambda t: Document(text=t, metadata={"summary": t, 'generated_at': current_time}),
    proposition_list
    )
    graph_maker = GraphMaker(ontology=ontology, llm_client=llm, verbose=True)

    graph = graph_maker.from_documents(
        list(docs), 
        delay_s_between=0 ## delay_s_between because otherwise groq api maxes out pretty fast. 
        )
    
    create_indices = False
    neo4j_graph = Neo4jGraphModel(edges=graph, create_indices=create_indices)
    neo4j_graph.save()

    return True
