import csv
import json
import os

import boto3
from langchain.chains import RetrievalQA
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

#Performing Data Transformation
def convertSummitDataFromCsvtoJson(csvfilePath, jsonFilePath):
    jsonf = open(jsonFilePath, 'a', encoding='utf-8')
    data_collection = []
    with open(csvfilePath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for rows in reader:
            data = {'summitName': rows['\ufeffSummit'] , 'awsSessionID': rows['AWS Session ID'] , 'sessionName': rows['Session name'] , 'sessionAbstract': rows['Abstract'] , 'sessionCapacity': rows['Capacity'],
                   'sessionDate': rows['Session Date'], 'sessionStartTime': str(rows['Start Time']).replace('1/1/00 ', '') , 'sessionEndTime': str(rows['End Time']).replace('1/1/00 ', ''), 'sessionDuration': rows['Session Duration'], 'sessionTopic': rows['Session Topic'],
                   'sessionTracks': rows['Session Tracks'], 'sessionType': rows['Session Type'], 'sessionLevel': rows['Session Level'], 'sessionAreaOfInterest': rows['Area of Interest']}
            data_collection.append(data)
    jsonf.write(json.dumps(data_collection, indent=4))
    jsonf.flush()
    jsonf.close()

def convertSummitDataFromCsvtoText(csvfilePath, jsonFilePath):
    jsonf = open(jsonFilePath, 'a', encoding='utf-8')
    with open(csvfilePath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for rows in reader:
            summitName = rows['\ufeffSummit']
            data = f"summitName: {summitName},awsSessionID:{rows['AWS Session ID']},sessionName: {rows['Session name']},sessionAbstract:{rows['Abstract']},sessionCapacity:{rows['Capacity']},\
                   sessionDate:{rows['Session Date']},sessionStartTime:{str(rows['Start Time']).replace('1/1/00 ', '')},sessionEndTime:{str(rows['End Time']).replace('1/1/00 ', '')},sessionDuration:{rows['Session Duration']},sessionTopic:{rows['Session Topic']},\
                   sessionTracks:{rows['Session Tracks']},sessionType:{rows['Session Type']},sessionLevel:{rows['Session Level']},sessionAreaOfInterest:{rows['Area of Interest']}"
            jsonf.write('{' + data + '}' + '\n')
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

def uploadSummitTxtDataToOSS(jsonFilePath, index_name): #creates and returns an in-memory vector store to be used in the application
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
        for data in f.readlines():
            with open('test.json', 'w') as testf:
                    testf.write(data)
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
        "temperature": 0.0
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


#Prompt Massaging
prompt_template = """You are tasked to answer the "{question}" with awsSessionID based on the following context information in html format. Your goal is to generate a final agenda that has no overlapping or conflicting sessions, while strictly adhering to the session Start Time and session End Time provided without adjusting them to fit the agenda based on the following context.

        {context}

        Failure Instructions:
        If you are unable to create an agenda that meets the requirements of having no overlapping or conflicting sessions and strictly adhering to the provided session Start and End Times, the task will be considered a failure. In the case of a failure, you should provide a detailed explanation outlining the reasons why the agenda could not be successfully generated within the given constraints.

        Question: {question}   

        Answer:"""

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
        ).as_retriever(search_type="mmr", search_kwargs={'k': 100, 'lambda_mult': 0.25})

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
    convertSummitDataFromCsvtoJson('session_metadata_newyork_final.csv', 'session_metadata_newyork_final.json')
    # convertSummitDataFromCsvtoText('session_metadata_newyork_final.csv', 'session_metadata_newyork_final.txt')
    # uploadSummitTxtDataToOSS('session_metadata_newyork_final.txt', 'ny_summit_session_metadata_txt')
    # rag_response = generateAgendaItems('I want a 9am-3pm agenda focused in Machine Learning and Big Data. Leave me an hour for lunch', 'ny_summit_session_metadata_txt')
    # # rag_response = generateAgendaItems('I want a 4 hour agenda including some hands on workshops. Compute and Open Source are most interesting to me.', 'ny_summit_session_metadata_txt')
    # print(rag_response['result'])