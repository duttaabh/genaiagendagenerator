import csv
import json
import os

import boto3
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import BedrockChat
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain_core.prompts import PromptTemplate
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth

#Declaring the environment variables and OpenSearch Resource details
region = 'us-east-1'
opensearch_endpoint = 'https://00n6mldeg966h7pktxve.us-east-1.aoss.amazonaws.com'
client = boto3.client('opensearchserverless', region_name=region)
service = 'aoss'
credentials = boto3.Session().get_credentials()

awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,
                   region, service, session_token=credentials.token)

#Setting up cohere.embed-english-v3 for vector embedding
embeddings = BedrockEmbeddings(
        region_name=region,
        model_id='cohere.embed-english-v3'
        # endpoint_url='https://prod.us-west-2.dataplane.bedrock.aws.dev'
    )  # create a Cohere Embeddings client

def convertStringToTime(input):
    format = '%m/%d/%y %H:%M'
    import datetime
    dt = datetime.datetime.strptime(input, format)
    return str(dt)

#Performing Data Transformation
def convertSummitDataFromCsvtoJson(csvfilePath, jsonFilePath):
    jsonf = open(jsonFilePath, 'a', encoding='utf-8')
    data_collection = []
    with open(csvfilePath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for rows in reader:
            data = {'summitName': rows['\ufeffSummit'] , 'awsSessionID': rows['AWS Session ID'] , 'sessionName': rows['Session name'] , 'sessionAbstract': rows['Abstract'] , 'sessionCapacity': rows['Capacity'],
                   'sessionDate': rows['Session Date'], 'sessionStartTime': convertStringToTime(str(rows['Start Time']).replace('1/1/00', rows['Session Date'])) , 'sessionEndTime': convertStringToTime(str(rows['End Time']).replace('1/1/00', rows['Session Date'])), 'sessionDuration': rows['Session Duration'], 'sessionTopic': rows['Session Topic'],
                   'sessionTracks': rows['Session Tracks'], 'sessionType': rows['Session Type'], 'sessionLevel': rows['Session Level'], 'sessionAreaOfInterest': rows['Area of Interest']}
            data_collection.append(data)
    jsonf.write(json.dumps(data_collection, indent=4))
    jsonf.flush()
    jsonf.close()

#Uploading data to the Opensearch Serverless Collection
def uploadSummitDataToOSS(jsonFilePath, index_name): #creates and returns an in-memory vector store to be used in the application
    opensearch_vector_search = OpenSearchVectorSearch(
        opensearch_url = opensearch_endpoint, #"https://5fso0en8s31bts1p2so5.us-east-1.aoss.amazonaws.com",
        index_name = index_name.lower(),
        embedding_function = embeddings,
        http_auth=awsauth,
        timeout = 300,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )

    with open(jsonFilePath, 'r') as f:
        # Load the JSON data
        data = json.load(f)

    for document in data:
        with open('test.json', 'w') as testf:
                testf.write(json.dumps(document))
        loader = TextLoader('test.json')
        documents = loader.load()
        response = opensearch_vector_search.add_documents(documents=documents)
        print(f'Document updated with response id {response[0]}')
        os.remove('test.json')
    # opensearch_vector_search.add_documents(documents=docs)
    return opensearch_vector_search

#Setting up the Anthropic Calude 3 Haiku LLM for RAG
def getLLM():
    model_kwargs_claude = {
        "temperature": 0.0,
        "top_k": 250
    }
    llm = BedrockChat(model_id="anthropic.claude-3-haiku-20240307-v1:0",
                  model_kwargs=model_kwargs_claude)
    # llm = BedrockChat(model_id="anthropic.claude-instant-v1",
    #               model_kwargs=model_kwargs_claude)
    return llm

def getMemory():  # create memory for this chat session
    memory = ConversationBufferWindowMemory(memory_key="chat_history",
                                            return_messages=True)  # Maintains a history of previous messages
    return memory


#Generating Keywords from questions
def rephraseQuestions(input_text):
    nysummit_date = '2023-07-26'
    prompt_template = """If the {question} does not start with a verb, rephrase the {question} to generate an agenda in a single sentence. Modify the start and end times with "between/and" cluase if mentioned like 8am-3pm, 9am-3pm, 09:00:00-15:00:00 etc. All start or end times converted to HH:MM:SS format. Do not generate any agenda or multi-line messages in the response or add any start or end time if not present in the {question}.
    
            Keywords:"""

    prompt = PromptTemplate.from_template(prompt_template)

    llm_chain = LLMChain(llm=getLLM(), prompt=prompt)

    response = llm_chain.invoke(input_text, return_only_outputs=False)
    # print(response['text'])
    arrAnswers = response['text'].split('\n')
    if len(arrAnswers) > 0:
        keywords = arrAnswers[len(arrAnswers) -1]
    return keywords

#Generating the final agenda using RAG
def generateAgendaItems(input_text, index_name):
    retriever = OpenSearchVectorSearch(
            opensearch_url=opensearch_endpoint,
            index_name=index_name,
            embedding_function=embeddings,
            http_auth=awsauth,
            timeout=300,
            use_ssl=True,
            verify_certs=True,
            engine="faiss",
            connection_class=RequestsHttpConnection
        ).as_retriever(search_type="mmr", search_kwargs={'k': 10, 'lambda_mult': 0.25})
    search = OpenSearchVectorSearch(
        opensearch_url=opensearch_endpoint,
        index_name=index_name,
        embedding_function=embeddings,
        http_auth=awsauth,
        timeout=300,
        use_ssl=True,
        verify_certs=True,
        engine="faiss",
        connection_class=RequestsHttpConnection
    )
    # Prompt Massaging
    prompt_template = """{question} with awsSessionID based on the following context information. Do not change the sessionStartTime and sessionEndTime in the generated agenda. Your goal is to generate a final agenda based on the session start time, end time and duration that has no overlapping or conflicting sessions, while strictly adhering to the session Start Time and session End Time provided without adjusting them to fit the agenda based on the following {context}. If there are multiple sessions that start or end around the same time within 15-30 mins and of similar duration, choose the one that starts first. 

            {context}

            Failure Instructions:
            If you are unable to create an agenda that meets the requirements of having no overlapping or conflicting sessions and without changing the session Start and End Times, the task will be considered a failure. In the case of a failure, you should provide a detailed explanation outlining the reasons why the agenda could not be successfully generated within the given constraints.

            Question: {question}

            Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(llm=getLLM(),
                                     chain_type="stuff",
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={"prompt": PROMPT, "verbose": True},
                                     verbose=True)

    chat_response = qa.invoke(input_text, return_only_outputs=False)
    return chat_response

def factCheckRag(input_text, index_name):
    retriever = OpenSearchVectorSearch(
            opensearch_url=opensearch_endpoint,
            index_name=index_name,
            embedding_function=embeddings,
            http_auth=awsauth,
            timeout=300,
            use_ssl=True,
            verify_certs=True,
            engine="faiss",
            connection_class=RequestsHttpConnection
        ).as_retriever(search_type="mmr", search_kwargs={'k': 10, 'lambda_mult': 0.25})

    # Prompt Massaging
    prompt_template = """You are tasked to check accuracy of sessionStart and sessionEndTime in the generated "{question}" using the following context. Please provide detailed explanation if any of the details are inaccurate.:

            {context}

            Question: {question}   

            Answer:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(llm=getLLM(),
                                     chain_type="stuff",
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={"prompt": PROMPT, "verbose": True},
                                     verbose=True)

    chat_response = qa.invoke(input_text, return_only_outputs=False)
    return chat_response

if __name__ == '__main__':
    question = 'I want a 9am-3pm agenda focused in Machine Learning and Big Data. Leave me an hour for lunch.'
    # question = 'I want a 4 hour agenda including some hands on workshops. Compute and Open Source are most interesting to me.'
    # question = 'Build a 5 hour agenda. Please include 1 hands-on workshop for Compute'
    rephrasedquestion = rephraseQuestions(question)
    print(rephrasedquestion)
    rag_response = generateAgendaItems(rephrasedquestion, 'ny_summit_session_metadata')
    print(rag_response['result'])
    # print(factCheckRag(rag_response['result'], 'ny_summit_session_metadata')['result'])
