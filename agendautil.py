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
        # .as_retriever(search_type="mmr", search_kwargs={'k': 5})
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
    prompt_template = """Generate an agenda based on the \"{question}\" including awsSessionID, sessionName, sessionStartTime in 'sessionDate am/pm' format, sessionEndTime in 'sessionDate am/pm' format and sessionDuration in minutes in a proper time sequence without any time manipulation, overlap or conflict based on the {context}.
            if awsSessionID is not found for a session, please generate awsSessionID, sessionName, sessionDate in DD-MMM-YYYY format, sessionStartTime in 'am/pm TIMEZONE' format, sessionEndTime in 'am/pm TIMEZONE' format and sessionDuration in minutes based on the context, otherwise use existing values.
            if Lunch is mentioned in the {question}, use Lunch as awsSessionID, Lunch as sessionName and set the session Start and End Time between 12pm and 1 pm if possible, otherwise do not include Lunch in the agenda.
            convert the sessionStartTime and sessionEndTime based on local timezone as """ + timezone + """
            Failure Instructions:
            If you are unable to create an agenda, the task will be considered a failure. Properly explain the cause of failure

            Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(llm=getLLM(),
                                     chain_type="stuff",
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={"prompt": PROMPT, "verbose": False},
                                     verbose=False)

    chat_response = qa.invoke(input_text, return_only_outputs=False)
    return chat_response['result']

# Function for checking overlaps, formatting the data for final and performing final validations
def overlapCheckJson(input_text, context, timezone):
    prompt_template = f"""
                         convert all the times to {timezone} timezone
                         Sort the {input_text} by sessionStartTime in ascending order.
                         Remove any overlapping sessions from the {input_text}.
                         if awsSessionID is not found for a session, please generate awsSessionID, sessionName, sessionDate in DD-MMM-YYYY format, sessionStartTime in 'am/pm timezone' format, sessionEndTime in 'am/pm timezone' with {timezone} format and sessionDuration in minutes based on the context, otherwise use existing values.
                         if Lunch is mentioned in the """ + context + """, use Lunch as awsSessionID, Lunch as sessionName and set the session Start and End Time between 12pm and 1 pm if possible, otherwise do not include Lunch in the agenda.
                         Generate the final agenda in json array format without any time manipulation. Do to make up an agenda that does not match {input_text}.
                         Must include the json attributes: awsSessionID, sessionName, sessionDate in DD-MMM-YYYY format, sessionStartTime in 'am/pm timezone' format, sessionEndTime in 'am/pm timezone' format and sessionDuration in minutes in the response.
                         Remove any conflicting or overlapping sessions based on their start or end time without manipulating the session start or end times in the {input_text}.
                         Remove any session which is longer than three hours. If more than one sessions have overlapping or same start time, keep the one with that matches the most with the """ + context + \
                      """If more than one sessions have overlapping or same end time, keep the one with that matches the most with the """ + context + \
                      """No need to provide any explanation.

                Answer:"""
    prompt = PromptTemplate.from_template(prompt_template)

    llm_chain = LLMChain(llm=getLLM(), prompt=prompt, verbose=False)

    response = llm_chain.invoke(input_text, return_only_outputs=False)
    # print(response)

    response = validateJsonResponse(response['text'])

    return response

# Utility functions to print the data in a pre-defined format
def validateJsonResponse(response):
    # print('response: ' + str(response))
    try:
        start_index = response.find('[')
        end_index = response.rindex(']') + 1
        # Extract the JSON message as a string
        json_message = response[start_index:end_index]
        # print("message: ", json_message)
        # Parse the JSON message
        data = json.loads(json_message)
        data = formatJsonMessage(data)
    except Exception as error:
        # print("No valid JSON message found.")
        print(error)
        data = response
    return data

def formatJsonMessage(jsonmessage):
    final_agenda = "\nPlease find below the recommended agenda based on your requirements\n========================================================================================"
    if 'agenda' in jsonmessage:
        for session in jsonmessage['agenda']:
            if 'awsSessionID' not in session:
                sessionID = 'NA'
            else:
                sessionID = session['awsSessionID']
            if 'sessionName' in session:
                final_agenda = final_agenda + '\n' + f"{session['sessionDate']} {session['sessionStartTime']} - {session['sessionEndTime']}\n========================================================================================\nSession ID: {sessionID} \nTopic: {session['sessionName']} \nSession Duration: {session['sessionDuration']}\n========================================================================================"
    else:
        for session in jsonmessage:
            if 'awsSessionID' not in session:
                sessionID = 'NA'
            else:
                sessionID = session['awsSessionID']
            if 'sessionName' in session:
                final_agenda = final_agenda + '\n' + f"{session['sessionDate']} {session['sessionStartTime']} - {session['sessionEndTime']}\n========================================================================================\nSession ID: {sessionID} \nTopic: {session['sessionName']} \nSession Duration: {session['sessionDuration']}\n========================================================================================"
    return final_agenda

# Function to get current server date time
def currentDateTime():
    from datetime import datetime
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%Y%m%d%H%M%S")
    return currentTime

if __name__ == '__main__':
    # question = 'I want a 9am-3pm agenda focused in Machine Learning and Big Data. Leave me an hour for lunch.'
    # question = 'I want a 4 hour agenda including some hands on workshops. Compute and Open Source are most interesting to me.'
    question = 'Build an agenda focused on ML, please leave me a 2 hour window so I can explore the conference booths.'
    # rephrasedquestion = rephraseQuestions(question)
    # print(rephrasedquestion)
    rag_response = generateAgendaItems(question, 'ny_summit_session_metadata')
    print('********************************************************************')
    print(rag_response)
    print('********************************************************************')
    print(overlapCheckJson(rag_response, question))
    print('********************************************************************')
