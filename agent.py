import streamlit as st
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    st.error("Google API Key is required")
    st.stop()

llm = ChatGoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key)

concept_extraction_prompt = PromptTemplate(
    input_variables=["query"],
    template="""Extract the key research concepts from the following query: {query}.
Return the concepts as a comma-separated list."""
)

related_concept_prompt = PromptTemplate(
    input_variables=["concept"],
    template="""For the research concept '{concept}', list 5 closely related concepts or ideas as a comma-separated list."""
)

concept_extraction_chain = llm | concept_extraction_prompt
related_concepts_chain = llm | related_concept_prompt

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Interactive Research Concept Mapper")

user_input = st.text_input("Enter your research query:", key="query_input")
if st.button("Generate Concept Network") and user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    
    formatted_extraction_prompt = concept_extraction_prompt.format(query=user_input)
    raw_concepts = concept_extraction_chain.invoke(formatted_extraction_prompt)
    raw_concepts_str = str(raw_concepts)
    concepts_list = [c.strip() for c in raw_concepts_str.split(",") if c.strip()]

    concept_network = {}
    for concept in concepts_list:
        formatted_related_prompt = related_concept_prompt.format(concept=concept)
        raw_related = related_concepts_chain.invoke(formatted_related_prompt)
        raw_related_str = str(raw_related)
        related_list = [r.strip() for r in raw_related_str.split(",") if r.strip()]
        concept_network[concept] = related_list

    st.session_state.history.append({"role": "assistant", "content": concept_network})

    st.write("### Concept Network:")
    st.json(concept_network)

if st.session_state.history:
    st.write("### Conversation History")
    for msg in st.session_state.history:
        st.write(f"**{msg['role'].capitalize()}:** {msg['content']}")
