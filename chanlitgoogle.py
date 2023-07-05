from langchain import PromptTemplate
import os
from langchain import PromptTemplate, LLMChain, OpenAI, LLMMathChain, SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from constants import openai_key, serpapikey
os.environ["OPENAI_API_KEY"] = openai_key
os.environ["SERPAPI_API_KEY"] = serpapikey


import chainlit as cl


@cl.langchain_factory(use_async=False)
def load():
    llm = ChatOpenAI(temperature=0, streaming=True)
    llm1 = OpenAI(temperature=0, streaming=True)
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools =    [     
        
           Tool(
        name="search",
        func= search.run,
        description="useful for when you need to answer questions about current events"
        ),

        Tool(
        name="calculator",
        func= llm_math_chain.run,
        description="useful for when you need to calculate"
        ),

    ]
    # prompt=PromptTemplate(template=template, input_variables=["question"])
    # llm_chain =  LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)
    return initialize_agent(

        tools, llm1, agent = "chat-zero-shot-react-description", verbose = True
    )