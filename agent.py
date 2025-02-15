from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("Google API Key is required")

# Switch to a supported model ID (using the text model here)
llm = ChatGoogleGenerativeAI(model="gemini-pro")

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

query = "Recent advancements in renewable energy storage technologies"

formatted_extraction_prompt = concept_extraction_prompt.format(query=query)
raw_concepts = concept_extraction_chain.invoke(formatted_extraction_prompt)
# Convert the output to a string before processing
raw_concepts_str = str(raw_concepts)
concepts_list = [concept.strip() for concept in raw_concepts_str.split(",") if concept.strip()]

# Build the concept network by expanding each concept
concept_network = {}
for concept in concepts_list:
    formatted_related_prompt = related_concept_prompt.format(concept=concept)
    raw_related = related_concepts_chain.invoke(formatted_related_prompt)
    raw_related_str = str(raw_related)
    related_list = [r.strip() for r in raw_related_str.split(",") if r.strip()]
    concept_network[concept] = related_list