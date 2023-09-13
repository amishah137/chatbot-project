#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 08:22:18 2023

@author: ami
"""
from flask import Flask, render_template, request

import re
import pandas as pd
from langchain.schema import Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain #,VectorDBQA
from langchain.llms import GooglePalm, LlamaCpp, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

path = "myDataset.csv"
persist_directory = "./vector_db"
openai_key = "your_openai_key"

app = Flask(__name__)
# app.config.from_object(__name__)

@app.route('/sync', methods=['POST'])
def saveData():
    docs = []
    data = pd.read_csv(path)

    for i in range(data.shape[0]):
        d = data.iloc[i]
        print(d['Name'])
        page_content = f"Employee {d['Name']} has experience of {d['Experience']} years in the role of {d['Role']} with skills of {d['Technology']}." 
        metadata = dict({"name":d['Name'], "role":d['Role'], 
                         "technology": d['Technology'],
                         "experience": d['Experience'], "source":path})
        docs.append(Document(page_content=page_content, metadata=metadata))
        docs = filter_complex_metadata(docs)
    
    with open("vectordb.txt","w+") as f:
        for items in docs:
           f.write('%s\n' %items)
    f.close()

    # embedding_function = SentenceTransformerEmbeddings(model_name=model)
    embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=openai_key)
    db = Chroma.from_documents(docs, embedding_function, persist_directory=persist_directory)
    print("Data Sync Successfully")
    return render_template('form.html')

def runChat(query):
    metadata_field_info = [
        AttributeInfo(
            name="name",
            description="name of a employee",
            type="string",
        ),
        AttributeInfo(
            name="role",
            description="Designation of the employee",
            type="string",
        ),
        AttributeInfo(
            name="technology",
            description=" skillset of employee which contains the skills the employee knows well",
            type="string",
        ),
        AttributeInfo(
            name="experience",
            description="years of experience employee has in the role and technology",
            type="int or string",
        )
    ]

    #"You are a helpful assistant that provides information about machine health to user."
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "To assist you in finding information about employee skillset, recoomand the list of employees with combined skilset to fulfill the requirement. The model is designed to handle variations in the input, including spelling mistakes, special characters, and different case characters. I will do my best to locate the relevant information based on the context."
                )
            ),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    document_content_description = "Information about employees with designation/Role, skillset and years of experience. Convert all information to string format."

    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.5, top_p=0.5, openai_api_key=openai_key)
    persist_directory = "./vector_db"
    
    model = "text-embedding-ada-002"
    embedding_function = OpenAIEmbeddings(model=model,openai_api_key=openai_key)

    
    # Now we can load the persisted database from disk, and use it as normal. 
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

    retriever = SelfQueryRetriever.from_llm(
        llm, db, document_content_description, metadata_field_info, verbose=False,
    )

    qa = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    
    res = qa(template.format_messages(input=query),return_only_outputs=False)
    print(res)
    return res['answer']  


@app.route('/')
def welcome():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def result():
    query = request.form.get("Query", type=str, default=0)
    entry = runChat(query)
    return render_template('form.html', entry=entry)

if __name__ == '__main__':
    app.run(debug=True)