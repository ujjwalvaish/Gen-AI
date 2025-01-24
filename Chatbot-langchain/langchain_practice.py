from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
import json


'''
Simple calling the local model
'''
print("1. -----------CALLING OLLAMA------------")
llm_model = OllamaLLM(model="llama3.1")
print(llm_model)
response = llm_model.invoke("Hi, How are you?")
# print(response)


'''
Prompt Template
'''
print("\n2. ---------------PROMPT TEMPLATE------------------")
# Principles - Be clear and specific, give it time
template_string = """
Translate the text whiich is delimited by triple backticks \
into a style that is {style}.\
text:```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)
# input_variables, messages
# print(type(prompt))
# for key in prompt:
#     print(key)

print(prompt_template.messages[0].prompt.template)
style="Pirate English"
text="""Hello, how are you"""

prompt = prompt_template.format_messages(
    style=style,
    text=text
)

# print(type(prompt))
# print(prompt)
# for key in prompt:
#     print(key)
print(prompt[0].content)
response = llm_model.invoke(prompt)
print(response)

'''
Output Parser
'''
print("\n3. ----------------OUTPUT PARSER--------------------")
customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """
For the following text, extract the following information:
gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output into a JSON with following keys:
gift
delivery_days
price_value

text:{text}
"""

prompt_template = ChatPromptTemplate.from_template(review_template)
prompt = prompt_template.format_messages(text=customer_review)
# response = llm_model.invoke(prompt)
# print(response) # still returns as a string - need output parser
# print(response["gift"]) # ERROR
# TODO - check why does the following not work
# json_response = json.loads(response)
# print(json_response)

gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema, delivery_days_schema, price_value_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
# print(format_instructions)


review_template_2 = """
For the following text, extract the following information:\

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text:{text}

{format_instructions}
"""
prompt_template = ChatPromptTemplate.from_template(review_template_2)
prompt = prompt_template.format_messages(
    text = customer_review,
    format_instructions = format_instructions
)

# print(prompt)
response = llm_model.invoke(prompt)
# print(response)

# response_dict = output_parser.parse(response) # WORKS SOMETIMES
# print(response_dict)
# print(type(response_dict))
# print(response_dict.get("delivery_days")) # WORKS!
# print(response_dict.get("gift")) # WORKS!

# Example
# from langchain_groq import ChatGroq
# from langchain_core.prompts import PromptTemplate
# from langchain.output_parsers import CommaSeparatedListOutputParser

# llm = ChatGroq(model="llama3-8b-8192")

# parser_list = CommaSeparatedListOutputParser()
# prompt_list = PromptTemplate.from_template(
#     template = "Answer the question.\n{format_instructions}\n{question}",
#     input_vairables = ["question"],
#     partial_variables = {"format_instructions": parser_list.get_format_instructions()},
# )

# prompt_value = prompt_list.invoke({"question": "List 4 chocolate brands"})
# response = llm.invoke(prompt_value)
# print(response.content)

# returned_object = parser_list.parse(response.content)
# print(type(returned_object))

"""
Memory
"""
# DOES NOT WORK -> Need memory
# print("TEST..................")
# print(llm_model.invoke("My name is XYZABC"))
# print(llm_model.invoke("What is AI"))
# print(llm_model.invoke("What is my name?"))

print("\n4. ----------------MEMORY--------------------")
memory=ConnectionAbortedError()
conversation=RunnableWithMessageHistory(
    llm=llm_model,
    memory=memory, 
    verbose=True
)

print(conversation.predict("My name is XYZABC"))
print(conversation.predict("What is AI"))
print(conversation.predict("What is my name?"))
