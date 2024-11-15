from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import  SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")

# Initialize the chat model with streaming enabled
chat_model = ChatGroq(api_key=groq_api, model="llama-3.1-70b-versatile")
SYSTEM_PROMPT = """As an experienced AI financial assistant, I bring extensive knowledge in stock market analysis, risk management, and portfolio strategy. I am here to provide impartial, well-informed guidance tailored to your specific financial goals and investment preferences.
{metrics}
Please feel free to share any questions or investment considerations. I’ll use my expertise—and your provided context, if applicable—to help you make informed financial decisions that align with your risk tolerance and objectives."
"""

def stream_chat_response(context, user_query):

    # Insert metrics into the system prompt
    system_prompt_filled = SYSTEM_PROMPT.format(metrics=context)

    # Create the prompt template with roles
    chat_temp = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt_filled),
        HumanMessage(content=user_query)
    ])

    # Format messages to a list of BaseMessages that can be passed to the model
    formatted_messages = chat_temp.format_messages()
    response =chat_model.invoke(formatted_messages)
    message_content = response.content
    return message_content


