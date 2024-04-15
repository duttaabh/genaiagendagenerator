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

# Function to generate the agenda from the UI
def generateAgenda(query):
    # Calling the RAG Method
    # rephrasedQuery = tsd.rephraseQuestions(query)
    return tsd.overlapCheckJson(tsd.generateAgendaItems(query, ny_summit_metadata, timezone), query, timezone)

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
st.title("NY Summit Agenda Generator")

query = st.text_input("Please tell us your session requirements")

if query:
    with st.spinner("Generating..."):
        results = generateAgenda(query)
        current_time = tsd.currentDateTime()
        save_to_pdf(str(results), current_time)
        with open('output_'+current_time+'.pdf', "rb") as pdf_file:
            pdf = pdf_file.read()
        os.remove('output_' + current_time + '.pdf')
        st.download_button('Save the agenda',
                           data=pdf,
                           file_name='agenda_'+tsd.currentDateTime()+'.pdf'
                          )
        messages = str(results).split('\n')
        for message in messages:
            st.write(message)
