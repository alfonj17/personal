# LangChain Project

This repository contains the main code used for various purposes in the LangChain project. Each section below covers different functionalities such as prompt templates, memory management, sequential chains, and routing chains. The purpose of this code is to demonstrate how to use LangChain to build complex and efficient language models.

## PROMPT TEMPLATE

This section demonstrates how to create and use a prompt template with LangChain.

```python
from langchain.prompts import ChatPromptTemplate

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(template_string)

# Format messages using the template
customer_messages = prompt_template.format_messages(
    style=customer_style,
    text=customer_email
)

# Initialize the chat model
chat = ChatOpenAI(temperature=0.0, model=llm_model)

# Generate the customer response
customer_response = chat(customer_messages)
```

## MEMORY
This section shows how to manage conversation memory using LangChain, which helps in maintaining context across interactions.

### Basic Conversation Memory
```python
Copy code
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize the chat model
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# Set up conversation memory
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=True
)
```

### Windowed Conversation Memory

```python
Copy code
from langchain.memory import ConversationBufferWindowMemory

# Initialize windowed memory
memory = ConversationBufferWindowMemory(k=1)

# Save conversation context
memory.save_context({"input": "Hi"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})

# Initialize the chat model with windowed memory
llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose=False
)
```

## SINGLESEQUENTIALCHAIN
This section explains how to set up a simple sequential chain to process a series of prompts sequentially.

```python
Copy code
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

# Initialize the chat model
llm = ChatOpenAI(temperature=0.9, model=llm_model)

# Define the first prompt
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe a company that makes {product}?"
)
chain_one = LLMChain(llm=llm, prompt=first_prompt)

# Define the second prompt
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following company: {company_name}"
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

# Combine the chains into a simple sequential chain
overall_simple_chain = SimpleSequentialChain(
    chains=[chain_one, chain_two],
    verbose=True
)

# Run the chain with a given product
overall_simple_chain.run(product)
```

## MULTIPLESEQUENTIALCHAIN
This section covers how to create a more complex sequential chain that processes multiple prompts in sequence and uses the output of one as the input for the next.

```python
Copy code
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Initialize the chat model
llm = ChatOpenAI(temperature=0.9, model=llm_model)

# Define the first prompt
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to English:\n\n{Review}"
)
chain_one = LLMChain(llm=llm, prompt=first_prompt, output_key="English_Review")

# Define the second prompt
second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:\n\n{English_Review}"
)
chain_two = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")

# Define the third prompt
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)
chain_three = LLMChain(llm=llm, prompt=third_prompt, output_key="language")

# Define the fourth prompt
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow-up response to the following summary in the specified language:\n\nSummary: {summary}\n\nLanguage: {language}"
)
chain_four = LLMChain(llm=llm, prompt=fourth_prompt, output_key="followup_message")

# Combine the chains into a sequential chain
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary", "followup_message"],
    verbose=True
)

# Run the chain with a given review
overall_chain.run(review)
```

## ROUTING CHAIN
This section demonstrates how to use a routing chain to select the appropriate prompt based on the input.

```python
Copy code
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate

# Define the prompt information
prompt_infos = [
    {
        "name": "physics", 
        "description": "Good for answering questions about physics", 
        "prompt_template": physics_template
    },
    {
        "name": "math", 
        "description": "Good for answering math questions", 
        "prompt_template": math_template
    },
    {
        "name": "history", 
        "description": "Good for answering history questions", 
        "prompt_template": history_template
    },
    {
        "name": "computer science", 
        "description": "Good for answering computer science questions", 
        "prompt_template": computerscience_template
    }
]

# Initialize the chat model
llm = ChatOpenAI(temperature=0, model=llm_model)

# Create destination chains for each prompt
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain  

# Create the list of destinations
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

# Define the default prompt
default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

# Define the router template
MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a language model, select the model prompt best suited for the input. 
You will be given the names of the available prompts and a description of what the prompt is best suited for. 
You may also revise the original input if you think that revising it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}
REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR it can be "DEFAULT" if the input is not well-suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
template=router_template,
input_variables=["input"],
output_parser=RouterOutputParser(),
)

Create the router chain
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

Create the multi-prompt chain
chain = MultiPromptChain(
router_chain=router_chain,
destination_chains=destination_chains,
default_chain=default_chain,
verbose=True
)

Run the chain with any input
chain.run("WHATEVER INPUT")
```

## LANGCHAIN Q&A

This section demonstrates how to set up a Q&A system using LangChain with vector stores and document retrievers.

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings

# Create an index from document loaders
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

# Embed a query using OpenAI embeddings
embeddings = OpenAIEmbeddings()
embed = embeddings.embed_query("Hi my name is Harrison")

# Create a vector store from documents
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

# Create a retriever from the vector store
retriever = db.as_retriever()

# Set up a RetrievalQA chain
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

# Run a query through the QA chain
response = qa_stuff.run(query)

# Query the index directly
response = index.query(query, llm=llm)
```

## AGENTS
This section demonstrates how to create and use agents in LangChain for various tasks.

Basic Agent Setup

```python
Copy code
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI

# Initialize the chat model
llm = ChatOpenAI(temperature=0, model=llm_model)

# Load tools for the agent
tools = load_tools(["llm-math","wikipedia"], llm=llm)

# Initialize the agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# Run a query through the agent
agent("What is the 25% of 300?")
```

## Python Agent

```python
Copy code
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.chat_models import ChatOpenAI

# Initialize the Python agent
agent = create_python_agent(
    llm,
    tool=PythonREPLTool(),
    verbose=True
)

# Run a Python command through the agent
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""")

# Enable debugging
import langchain
langchain.debug=True

# Run another command with debugging
agent.run(f"""Sort these customers by \
last name and then first name \
and print the output: {customer_list}""")

# Disable debugging
langchain.debug=False
```

## CUSTOMIZE AGENT TOOL
This section shows how to customize an agent tool for specific tasks.

```python
Copy code
from langchain.agents import tool
from datetime import date

@tool
def time(text: str) -> str:
    """Returns today's date. Use this for any questions related to knowing today's date.
    The input should always be an empty string, and this function will always return today's date.
    Any date mathematics should occur outside this function."""
    return str(date.today())

# Add the custom tool to the agent
agent = initialize_agent(
    tools + [time], 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)
```

## Conclusion
This repository provides various examples of using LangChain for different purposes. Each section illustrates how to implement and use specific features, helping you to build robust and efficient language models. Feel free to explore and modify the code to suit your needs.

