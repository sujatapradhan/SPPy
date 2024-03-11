import sys
sys.path.append('C:/Users/a138821/OneDrive - Eviden/Sujata/Dynamic/Work/SPCode/SPPy/')
# load a source module from a file

from Common import initEnv 
initEnv.spInitEnv
import Common
Common.initEnv.spInitEnv
print (Common.initEnv.spInitEnv)
print (Common.initEnv.getPath)


import os
print("Hello ",os.environ["HUGGINGFACEHUB_API_TOKEN"])
print("Serpapi",__file__)
print(os.getcwd())

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