import csv
import datetime
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

# Declaring the static variables and constants
region = 'us-east-1'
opensearch_endpoint = 'https://00n6mldeg966h7pktxve.us-east-1.aoss.amazonaws.com'
client = boto3.client('opensearchserverless', region_name=region)
service = 'aoss'
credentials = boto3.Session().get_credentials()

awsauth = AWS4Auth(credentials.access_key, credentials.secret_key,
                   region, service, session_token=credentials.token)

# Function tp set up cohere.embed-english-v3 for vector embedding
embeddings = BedrockEmbeddings(
        region_name=region,
        model_id='cohere.embed-english-v3'
        # endpoint_url='https://prod.us-west-2.dataplane.bedrock.aws.dev'
    )  # create a Cohere Embeddings client

# Fucntion to format Date time attributes
def convertStringToTime(input):
    format = '%m/%d/%y %H:%M'
    import datetime
    dt = datetime.datetime.strptime(input, format)
    return str(dt)

# Fucntion to perform Data Transformation for OpenSearch Serverless collection
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

# Function to upload data to the Opensearch Serverless Collection
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

# Function to set up the Anthropic Calude 3 Haiku LLM for RAG
def getLLM():
    model_kwargs_claude = {
        "temperature": 0.0,
        "top_k": 250
    }
    llm = BedrockChat(model_id="anthropic.claude-3-haiku-20240307-v1:0",
                  model_kwargs=model_kwargs_claude)
    return llm

# Fucntion to set up memory based chat history
def getMemory():  # create memory for this chat session
    memory = ConversationBufferWindowMemory(memory_key="chat_history",
                                            return_messages=True)  # Maintains a history of previous messages
    return memory

# Fucntion to generate the social cues
def findSocialActivities(input_text):
    prompt_template = """
                             Find the social activities in this sentence - {input_text} and put then in a comma separated string without any additional explanation

                    Answer:"""
    prompt = PromptTemplate.from_template(prompt_template)

    llm_chain = LLMChain(llm=getLLM(), prompt=prompt, verbose=False)

    response = llm_chain.invoke(input_text, return_only_outputs=False)
    # print(response)

    response = response['text']

    return response

# Fucntion to generate the RAW agenda using LLM
def generateAgendaItems(input_text, index_name, timezone):
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
        ).as_retriever()

    keywords = findSocialActivities(input_text)
    # print(keywords)
    # Prompt Massaging
    prompt_template = """
            Please generate an agenda in jsonArray format based on the \"{question}\" and {context} only, strictly adhereing to the start and end times and without trying to generate anything on your own.
            Include Lunch in the agenda if mentioned in the {question}.
            Include additional activities based on """ + keywords + """ related to the {question}.
            Please include the following attributes awsSessionID, sessionName, sessionDate, sessionStartTime, sessionEndTime and sessionDuration in mins for every session.
            Convert session start and end times from America/New York to """ + timezone + """\
            Failure Instructions:
            If you are unable to create an agenda, the task will be considered a failure. No need to explain the cause of failure.

            Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(llm=getLLM(),
                                     chain_type="stuff",
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={"prompt": PROMPT, "verbose": False},
                                     verbose=False)

    rag_response = qa.invoke(input_text, return_only_outputs=False)
    # print(rag_response['result'])
    agenda_response = sorted(validateJsonResponse(rag_response['result']), key=lambda x: datetime.datetime.strptime(x['sessionStartTime'], '%Y-%m-%d %H:%M:%S'))

    return json.dumps(agenda_response)

# Function for checking overlaps, formatting the data for final and performing final validations
def overlapCheckJson(input_text, timezone):
    # print("oss_message: ", input_text)
    prompt_template = """
                         Please do not create any agenda which does not match {input_text}
                         Remove any overlapping or conflicting activities in the {input_text}
                         Remove any conflicting or overlapping sessions based on their start or end time without manipulating the session start or end times in the {input_text}.
                         Remove any session which is longer than three hours. If more than one sessions have overlapping or same start time, keep the most relevant one.
                         If more than one sessions have overlapping or same end time, keep the most relevant one.
                         Change the sessionDate to DD-MMM-YYYY format and session times to 'AM/PM Timezone' format
                         No need to provide any explanation.

                Answer:"""
    prompt = PromptTemplate.from_template(prompt_template)

    llm_chain = LLMChain(llm=getLLM(), prompt=prompt, verbose=False)

    response = llm_chain.invoke(input_text, return_only_outputs=False)
    # print(response)

    response = response['text']

    return response

# Utility functions to print the data in a pre-defined format
def validateJsonResponse(response):
    # print('response: ', response)
    if len(response) > 0:
        # try:
        if response.index('[')>-1 and response.index(']')>-1:
            start_index = response.find('[')
            end_index = response.rindex(']') + 1
            # Extract the JSON message as a string
            json_message = response[start_index:end_index]
            # print("message: ", json_message)
            # Parse the JSON message
            data = json.loads(json_message)
            # print("data: ", data)
            # data = formatJsonMessage(data)
        else:
            data = ''
    return data

def formatJsonMessage(jsonmessage):
    # print(jsonmessage)
    if len(jsonmessage) > 0:
        final_agenda = "\nPlease find below the recommended agenda based on your requirements\n=============================================================================="
        for session in jsonmessage:
            if 'awsSessionID' not in session:
                sessionID = 'NA'
            else:
                sessionID = session['awsSessionID']
            if 'sessionName' in session:
                final_agenda = final_agenda + '\n' + f"{session['sessionDate']} {session['sessionStartTime']} - {session['sessionEndTime']}\n==============================================================================\nSession ID: {sessionID} \nTopic: {session['sessionName']} \nSession Duration: {session['sessionDuration']}\n=============================================================================="
    else:
        final_agenda = ''
    return final_agenda

# Function to get current server date time
def currentDateTime():
    from datetime import datetime
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%Y%m%d%H%M%S")
    return currentTime

if __name__ == '__main__':
    question = 'I want a 9am-3pm agenda focused in Machine Learning and Big Data. Leave me an hour for lunch.'
    # question = 'I want a 4 hour agenda including some hands on workshops. Compute and Open Source are most interesting to me.'
    # question = 'Build an agenda focused on ML, please leave me a 2 hour window so I can explore the conference booths.'
    # rephrasedquestion = rephraseQuestions(question)
    # print(rephrasedquestion)
    timezone = 'America/Austin'
    rag_response = generateAgendaItems(question, 'ny_summit_session_metadata', timezone)
    print('********************************************************************')
    print(formatJsonMessage(validateJsonResponse(overlapCheckJson(rag_response, timezone))))
    print('********************************************************************')
    # print(overlapCheckJson(rag_response, question, 'America/Austin'))
    # print('********************************************************************')
