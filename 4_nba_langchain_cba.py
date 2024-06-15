from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#Change the prompt template from telling a joke to teaching you about a topic. We will see
#a demonstration of the LLM hallucinating on topics outside of its training data
# hallucinations: confidently wrong, it won't say "i think...", out of date information
# limitations: not up to date, lack knowledge outside data it was trained on, how to combat this?
# try to break it :)

llm = ChatOllama(model='llama3')

prompt = ChatPromptTemplate.from_template("Teach me about this {topic}")

chain = prompt | llm | StrOutputParser()

user_message = input("Enter a topic you want to learn: ")
print(chain.invoke({'topic': user_message}))
