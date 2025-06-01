from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = Ollama(model ="llama3.2")
template = """
You are an expert in answering question about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
while True:
    print("\n\n---------------------------------------------------")
    question = input("Enter your question ('q' to quit): ")
    print("---------------------------------------------------\n\n")
    if question ==  "q":
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": "What is the best pizza place in town?"})
    print(result)