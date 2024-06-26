import time

import streamlit as st
import os

from streamlit_javascript import st_javascript

import agendautil as tsd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

ny_summit_metadata = 'ny_summit_session_metadata'

# Code snippet to find the browser timezone to convert the session start and end time
timezone = st_javascript("""await (async () => {
            const userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
            console.log(userTimezone)
            return userTimezone
})().then(returnValue => returnValue)""")

if 'converted_timezone' not in st.session_state:
    converted_timezone = tsd.getCustomerTimezone(timezone)
    st.session_state['converted_timezone'] = converted_timezone
else:
    converted_timezone = st.session_state['converted_timezone']
# Function to generate the agenda from the UI
def generateAgenda(query, question_keywords):
    # Calling the RAG Method
    # return tsd.validateJsonResponse(tsd.overlapCheckJson(tsd.generateAgendaItems(query, ny_summit_metadata, timezone), timezone))
    return tsd.validateJsonResponse(
        tsd.generateAgendaItems(query, ny_summit_metadata, timezone, converted_timezone, question_keywords))

def removeItemFromAgenda(jsonArray, key):
    newJsonArray = []
    j = 1
    for message in jsonArray:
        if position != j:
            newJsonArray.append(message)
        j = j + 1
    return newJsonArray

# Function to save the generated agenda ina downloadable PDF format
def save_to_pdf(text, current_time):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate('output_'+current_time+'.pdf', pagesize=letter)
    elements = []
    for message in text.split('\n'):
        p = Paragraph(message, styles["BodyText"])
        elements.append(p)
    doc.build(elements)

# UI Code begins here
if __name__ == '__main__':
    st.title("AWS Summit Agenda Generator")

    query = st.text_input("Please tell us your session requirements")

    if 'question_keywords' in st.session_state:
        st.session_state.pop('question_keywords')

    if query:
        try:
            popSearchResult = True
            if 'searchResult' in st.session_state:
                jsonArray = st.session_state['searchResult']
                arrayLen = len(jsonArray)
                for i in range(1, arrayLen + 1):
                    if ("chk_" + str(i)) in st.session_state and st.session_state[("chk_" + str(i))] == False:
                        position = i
                        popSearchResult = False
                        break;
                    else:
                        popSearchResult = True

                if popSearchResult:
                    st.session_state.pop('searchResult')

            if 'question_keywords' not in st.session_state:
                question_keywords = tsd.findSocialActivities(query)
            else:
                question_keywords = st.session_state['question_keywords']
            start = time.time()
            with st.spinner("Generating..."):
                    # print(st.session_state['searchResult'])
                    position = 0

                    if 'searchResult' in st.session_state:
                        jsonArray = st.session_state['searchResult']
                        arrayLen = len(jsonArray)
                        for i in range(1, arrayLen + 1):
                            if ("chk_" + str(i)) in st.session_state and st.session_state[("chk_" + str(i))] == False:
                                position = i
                                break;
                        jsonArray = removeItemFromAgenda(jsonArray, position)
                        st.session_state['searchResult'] = jsonArray
                    else:
                        jsonArray = generateAgenda(query, question_keywords)

                    st.session_state['searchResult'] = jsonArray

                    results = tsd.formatJsonMessage(st.session_state['searchResult'])

                    st.success('Useful information: Please uncheck the boxes below to remove any session from the agenda.')
                    st.success(
                        'Please be cautious with your choice as you might not be able to add them back in the agenda.')
                    end = time.time()
                    st.success(f'Time Elapsed: {str(int(end - start))} seconds.')
                    if len(results) > 0:
                        current_time = tsd.currentDateTime()
                        save_to_pdf(str(results).replace("#chk#", ''), current_time)
                        with open('output_'+current_time+'.pdf', "rb") as pdf_file:
                            pdf = pdf_file.read()
                        os.remove('output_' + current_time + '.pdf')
                        st.download_button('Save & Download the agenda',
                                           data=pdf,
                                           file_name='agenda_'+tsd.currentDateTime()+'.pdf'
                                          )
                        messages = str(results).split('\n')
                        i = 0
                        for message in messages:
                            if len(message) > 0 and '#chk#' in message:
                                i = i + 1
                                st.checkbox(label=message.replace('#chk#', ''), key=('chk_'+str(i)), value=True)
                            else:
                                st.write(message)
                    else:
                        st.error("Sorry, unable to generate the agenda based on your requirements. Please try again later.")
        except Exception as e:
            # print(e)
            import traceback
            traceback.print_exc()
            st.error("Sorry, unable to generate the agenda due to some system errors. Please try again later.")
