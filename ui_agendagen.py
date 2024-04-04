import streamlit as st
import agendautil as tsd

ny_summit_metadata = 'ny_summit_session_metadata_txt'

def search_function(query):
    # Calling the RAG Method
    return tsd.generateAgendaItems(query, ny_summit_metadata)

st.title("NY Summit Agenda Generator")

query = st.text_input("Please tell us your session requirements")

if query:
    with st.spinner("Generating..."):
        results = search_function(query)
    st.success("Your Agenda has been sucessfully generated!")
    messages = str(results['result']).split('\n')
    for message in messages:
        st.write(message)