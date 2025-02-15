from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import time

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("Google API Key is required")

# Use a supported model (verify access to gemini-1.5-pro or switch to gemini-pro)
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

# Define prompt templates
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

# Build chains with proper prompt->model order
concept_extraction_chain = concept_extraction_prompt | llm | StrOutputParser()
related_concepts_chain = related_concept_prompt | llm | StrOutputParser()

# Example query
query = "Recent advancements in renewable energy storage technologies"

try:
    # Extract initial concepts
    concepts_str = concept_extraction_chain.invoke({"query": query})
    concepts_list = [c.strip() for c in concepts_str.split(",") if c.strip()]
    print("Extracted Concepts:", concepts_list)

    # Get related concepts with rate limiting
    concept_network = {}
    for concept in concepts_list:
        try:
            related_str = related_concepts_chain.invoke({"concept": concept})
            related_list = [r.strip() for r in related_str.split(",") if r.strip()]
            concept_network[concept] = related_list
            print(f"Related to {concept}: {related_list}")
            time.sleep(1)  # Add delay between requests
        except Exception as e:
            print(f"Error processing {concept}: {str(e)}")
            continue

    print("\nConcept Network:")
    for k, v in concept_network.items():
        print(f"- {k}: {', '.join(v)}")

except Exception as e:
    print(f"Critical error: {str(e)}")