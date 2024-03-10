# Access to env varibales
import os
# Set up to read API keys from .env
#install dotenv
from dotenv import find_dotenv, load_dotenv
load_dotenv("../.env")             # now you have access to os.environ["HUGGINGFACEHUB_API_TOKEN"]
print("Hello ",os.environ["HUGGINGFACEHUB_API_TOKEN"])

#Ref https://www.sitepoint.com/langchain-python-complete-guide/
from langchain.llms import OpenAI
llm = OpenAI(temperature=0)

from langchain.agents import load_tools
from langchain.agents import AgentType, initialize_agent
# The tools we'll give the Agent access to. Note that the 'llm-math' tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("How much energy did wind turbines produce worldwide in 2022?")