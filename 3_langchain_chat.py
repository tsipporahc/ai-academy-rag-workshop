from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#Use Langchain to create Prompt Templates.  This can help with structuring prompts if you want your
#LLM to only serve specific purposes such as telling jokes
# Create chat bot that protect against prompt injection

llm = ChatOllama(model='llama3')

# prompt template, give response in the form of a joke, protect from prompt injection
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

# defines chain of commands to the LLM, langchain can provide memory and modify previous response
chain = prompt | llm | StrOutputParser()

# Drake joke lol
user_message = input("What topic of a joke would you like to hear?")
print(chain.invoke({'topic': user_message}))
