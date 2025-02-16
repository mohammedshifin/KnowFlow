import streamlit as st
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    st.error("Google API Key is required")
    st.stop()

# Initialize the chat model with gemini-pro and temperature=0
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0)

# Define prompt templates for concept extraction and related concept generation
concept_extraction_prompt = PromptTemplate(
    input_variables=["query"],
    template="""Extract key research concepts from: {query}
Return as a comma-separated list. Do not include explanations."""
)

related_concept_prompt = PromptTemplate(
    input_variables=["concept"],
    template="""List 5 closely related concepts for '{concept}'.
Return as a comma-separated list. No explanations."""
)

# Create chains by combining the prompt with the LLM and parsing output as a string
concept_extraction_chain = concept_extraction_prompt | llm | StrOutputParser()
related_concepts_chain = related_concept_prompt | llm | StrOutputParser()

# Initialize conversation history in session state if it doesn't exist
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Interactive Research Concept Mapper")

user_query = st.text_input("Enter your research query:")

if st.button("Generate Concept Network") and user_query:
    st.session_state.history.append({"role": "user", "content": user_query})
    
    try:
        # Extract key concepts from the query
        concepts_str = concept_extraction_chain.invoke({"query": user_query})
        concepts_list = [c.strip() for c in concepts_str.split(",") if c.strip()]
        st.write("**Extracted Concepts:**", concepts_list)
        
        # Build a network by expanding each concept with related ideas
        concept_network = {}
        for concept in concepts_list:
            try:
                related_str = related_concepts_chain.invoke({"concept": concept})
                related_list = [r.strip() for r in related_str.split(",") if r.strip()]
                concept_network[concept] = related_list
                st.write(f"**Related to {concept}:**", related_list)
                time.sleep(1)  # Optional: to avoid hitting API rate limits
            except Exception as e:
                st.error(f"Error processing {concept}: {str(e)}")
                continue

        st.session_state.history.append({"role": "assistant", "content": concept_network})
        
        st.write("### Concept Network:")
        st.json(concept_network)
    except Exception as e:
        st.error(f"Critical error: {str(e)}")

if st.session_state.history:
    st.write("### Conversation History")
    for msg in st.session_state.history:
        st.write(f"**{msg['role'].capitalize()}:** {msg['content']}")
