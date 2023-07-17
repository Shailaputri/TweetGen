'''
The app uses an old da-vinci model of GPT to search up-to-date einformation on Wikipedia using Langchain to produce tweets.
'''
import os
import openai

import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

openai_api_key = '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'
os.environ['OPENAI_API_KEY'] = openai_api_key



#Simple UI
st.title('									🦜⛓🐦										')
prompt = st.text_input("What's on your mind?")

# template for the title
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a one liner short tweet about {topic}'
)

# template for the tweet
tweet_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    template='write me a tweet on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} which is different from title. '
)

# wrapper for Wikipedia data
wiki = WikipediaAPIWrapper()

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
tweet_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Llms
llm = OpenAI(model_name="text-davinci-003", temperature=0.9)
# llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
tweet_chain = LLMChain(llm=llm, prompt=tweet_template, verbose=True, output_key='tweet', memory=tweet_memory)

# Chaining the components and displaying outputs
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    tweet = tweet_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(tweet) 

    # with st.expander('Title History'): 
    #     st.info(title_memory.buffer)

    with st.expander('Detailed Read'): 
        st.info(wiki_research)

    with st.expander('Tweet Details'): 
        st.info(tweet_memory.buffer)

title_memory.clear()
tweet_memory.clear()    


