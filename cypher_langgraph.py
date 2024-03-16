import operator
import os
from datetime import datetime
from typing import Annotated, TypedDict, Union, Sequence, List
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.chains import create_extraction_chain
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain.agents import AgentExecutor
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from neo4j import GraphDatabase
import json
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('app.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
model_name = "openhermes"
#model_name = "mistral"
load_dotenv()

@tool
def get_now(format: str = "%Y-%m-%d %H:%M:%S"):
    """
    Get the current time
    """
    return datetime.now().strftime(format)

@tool
def data_analyst(user_question: str ):
    """
    Produce a plan to collect the data requried in order to answer the user question
    """
   
    prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """
    You are world class world class data analyst.  
    You are authorized to query customer data, merchant data and payment data from Graph DB.
    You are given the Graph data contains which customer pay which merchant,
    and has the payment amount stored in the relationship properties. 
    The relationship or the edge between customer and merchant is PAY.
    Apart from PAY there is no other relationship.
    Step 1 is to Produce a plan to collect the data requried.
    In order to collect the data, you need to provide the CYPHER graph query according to the schema of the graph:
    there are 2 types of nodes: customer and merchant.  Both in lower case.
    Node customer has 1 properties: nameOrig, which is the customer id.
    Node merchant has 1 properties: nameDest, which is the merchant id.
    The edge between the nodes is :PAY, which has 2 properties: amount and type, 
    the edge means the customer pays the merchant with the payment amount and of payment type.
    Apart from :PAY there is no other relationship. 
    You can only use :PAY as the relationship in the query.
    Please DO NOT use other relationship.
    Here is some example cypher queries:
    1.  user question: retrieve 100 customers
        cypher query:  MATCH (c:customer) RETURN c.nameOrig LIMIT 100;
    2.  user question: retrieve 100 merchants
        cypher query:  MATCH (m:merchant) RETURN m.nameDest LIMIT 100;
    3.  user question: Find all customers who paid a specific merchant:
        cypher query:  MATCH (c:customer)-[p:PAY]->(m:merchant) where m.nameDest = MERCHANT_ID RETURN c.nameOrig LIMIT 100;
    4.  user question: Find all merchants paid by a specific customer:
        cypher query:  MATCH (c:customer)-[p:PAY]->(m:merchant) where c.nameOrig = CUSTOMER_ID RETURN m.nameDest LIMIT 100;
    5.  user question: Retrieve payments of a certain type made by customers:
        cypher query:  MATCH (c:customer)-[p:PAY]->(m:merchant) where p.type = PAYMENT_TYPE RETURN c.nameOrig, m.nameDest, p.amount LIMIT 100;
    6.  user question: Find the total amount paid to each merchant:
        cypher query:  MATCH (c:customer)-[p:PAY]->(m:merchant) RETURN m.nameDest, SUM(p.amount) AS total_amount LIMIT 100;
    7.  user question: List customers who have made payments above a certain amount:
        cypher query:  MATCH (c:customer)-[p:PAY]->(m:merchant) WHERE p.amount > AMOUNT_THRESHOLD RETURN c.nameOrig, p.amount LIMIT 100;
    8.  user question: Retrieve the highest payments:
        cypher query:  MATCH (c:customer)-[p:PAY]->(m:merchant) RETURN c.nameOrig, m.nameDest, p.amount ORDER BY p.amount DESC LIMIT 100;
    9.  user question: count the number of merchant for a customer:
        cypher query:  MATCH (c:customer)-[p:PAY]->(m:merchant) where c.nameOrig = CUSTOMER_ID RETURN count(m) as count 
        
    Step 2 is to wait for the execution of all the cypher queries
    Step 3 After the execution of the cypher queries completed, analyse the data collected, 
    and base on the data to answer the user question.

    Please DO NOT answer the user questions without the cypher query execution.
    """),
    ("user", "{user_question}")
    ])
    llm = ChatOllama(model=model_name)
    chain = prompt | llm 
    response = chain.invoke({"user_question": user_question})
    logger.info("-----------------PLANNING TOOL-----------------------")
    logger.info(response)
    return response

@tool
def execute_cypher(cypher_query: str):
    """
        After the planning produced by the Data Analyst, execute the cypher query to collect customer data, merchant data and payment data from Graph DB.
    """
    schema = {
    "properties": {
        "cypher_query": {"type": "string"}
    },
    "required": ["cypher_query"],
    }

    extract_chain = create_extraction_chain(schema, 
            OllamaFunctions(model=model_name, temperature=0))
    extracted_response = extract_chain.invoke(cypher_query)
    logger.info(extracted_response)
    extracted_text = extracted_response.get('text','')
    logger.info(extracted_text)
    cypher_query_statement = extracted_text[0].get('cypher_query','').replace("\'","\"")
    URI = "neo4j+s://??????????"
    AUTH = ("neo4j", "?????????")
    logger.info(f"\n-------------executing cypher query ...")
    logger.info(cypher_query)
    try: 
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            driver.verify_connectivity()
            
            records, summary, keys = driver.execute_query(
                cypher_query_statement,
                database_="neo4j",
            )
            result = []
            for rec in records:
                result.append(json.dumps(rec))
            response = ",".join(result)
            logger.info("-----------------EXECUTE CYPHER QUERY TOOL-----------------------")
            logger.info(response)
            return response
    except Exception as e:
        return f"Query failed: {e}"

tools = [data_analyst, execute_cypher]

model = ChatOllama(model="openhermes",temperature=0)
prompt = hub.pull("hwchase17/react")
print(prompt)
agent_runnable = create_react_agent(model, tools, prompt)
tool_executor = ToolExecutor(tools)

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]




def execute_tools(state):
    print("Called `execute_tools`")
    messages = [state["agent_outcome"]]
    last_message = messages[-1]
    logger.info("------------executing tool-------------")
    logger.info(last_message)
    logger.info("---------------------------------------")
    tool_name = last_message.tool

    logger.info(f"Calling tool: {tool_name}")

    action = ToolInvocation(
        tool=tool_name,
        tool_input=last_message.tool_input,
    )
    response = tool_executor.invoke(action)
    logger.info(response)
    return {"intermediate_steps": [(state["agent_outcome"], response)]}


def run_agent(state):
    """
    #if you want to better manages intermediate steps
    inputs = state.copy()
    if len(inputs['intermediate_steps']) > 5:
        inputs['intermediate_steps'] = inputs['intermediate_steps'][-5:]
    """
    logger.info("-------------------agent starting--------------------")
    agent_outcome = agent_runnable.invoke(state)
    logger.info(agent_outcome)
    return {"agent_outcome": agent_outcome}


def should_continue(state):
    messages = [state["agent_outcome"]]
    last_message = messages[-1]
    logger.info("----------should_continue---------")
    logger.info(last_message)
    logger.info("----------")
    if "Action" not in last_message.log:
        return "end"
    else:
        return "continue"

workflow = StateGraph(AgentState)

workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)


workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "action", "end": END}
)


workflow.add_edge("action", "agent")
app = workflow.compile()

input_text = """
Who is the most valuable customer paying for the merchant M1979787155 ?
"""

inputs = {"input": input_text, "chat_history": []}
results = []
for result in app.stream(inputs):
    print(result)

