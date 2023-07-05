from langchain import PromptTemplate
import os
from langchain import PromptTemplate, LLMChain, OpenAI
from constants import openai_key
os.environ["OPENAI_API_KEY"] = openai_key

import chainlit as cl

template = """Question: {question}

Answer: Let's think step by step."""

@cl.langchain_factory(use_async=True)
def factory():
    prompt=PromptTemplate(template=template, input_variables=["question"])
    llm_chain =  LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)
    return llm_chain