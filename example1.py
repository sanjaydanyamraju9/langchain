import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
import streamlit as st
from langchain.chains import LLMChain 

from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain

from langchain.memory import ConversationBufferMemory

# Memory

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')


st.title("Celebrity Search") 

os.environ["OPENAI_API_KEY"] = openai_key
input_text = st.text_input("Search the topic you want")


llm = OpenAI(temperature=0.8)

first_prompt = PromptTemplate(
                        input_variables=['name'],
                        template='Celebrity name: {name}'
                        )

LLm_chain_1 = LLMChain(llm=llm, prompt=first_prompt, verbose=True,output_key = 'person', memory=person_memory)

second_prompt = PromptTemplate(
                        input_variables=['person'],
                        template='when was {person} born?')

LLm_chain_2 = LLMChain(llm=llm, prompt=second_prompt, verbose=True, output_key='dob', memory=dob_memory)


third_prompt = PromptTemplate(
                        input_variables=['dob'],
                        template='mention 5 major events happened around {dob} in the world')

LLm_chain_3 = LLMChain(llm=llm, prompt=third_prompt, verbose=True, output_key='description', memory=descr_memory)

# parentchain= SimpleSequentialChain(chains=[LLm_chain_1,LLm_chain_2], verbose=True)

parentchain2= SequentialChain(chains=[LLm_chain_1,LLm_chain_2,LLm_chain_3], input_variables=['name'], output_variables=['person','dob','description'],verbose=False)

if input_text:

    st.write(parentchain2({'name':input_text}))

    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)