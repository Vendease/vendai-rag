import os
import json
import openai
import main_rag
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain.prompts import PromptTemplate

# ------------------------------------------------------------------------------
# Environment & OpenAI Setup
# ------------------------------------------------------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_ID = "gpt-4"  # Use the latest GPT-4 model that supports function calling
client = OpenAI()
# ------------------------------------------------------------------------------
# Define Tool Specifications for Function Calling
# ------------------------------------------------------------------------------
tools = [
    {
        "name": "generate_sql",
        "description": (
            "Generate a SQL query from a user's natural language data query. "
            "Return a valid SQL query that retrieves the requested data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's query describing the desired data retrieval."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "perform_rag",
        "description": (
            "Tool to resolve errors on the vendease platform via RAG. "
            "Return a concise, informative answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's query for which to retrieve context and generate an answer."
                }
            },
            "required": ["query"]
        }
    }
]

# ------------------------------------------------------------------------------
# Role Prompt for the Agent
# ------------------------------------------------------------------------------
role_prompt = """
You are VendAI, a Super Intelligent Chatbot with Advanced Capabilities. You were developed by the AI team at Vendease to only understand and respond to various queries related to customers orders, customer deliveries, customers purchases, customers information, Vendease product information called SKUs.\
You also have the ability to answer questions about fixing and resolving errors that users would have about using the platform, and general information about vendease.
When a user's query is about constructing a data query or retrieving data, use the generate_sql tool.\
When the user's query is about resolving an error or fixing a problem, use the perform_rag tool.\
If the query is ambiguous, choose the tool that best fits the user's intent.\
Return a function call in JSON format with the appropriate tool and its arguments.
When there is no clear answer, ask follow-up questions to clarify the user's intent.\
"""
# ------------------------------------------------------------------------------
# Local Functions (Python Native Implementations)
# ------------------------------------------------------------------------------
def local_generate_sql(query: str) -> str:
    """
    Generate a SQL query from the user's natural language request.
    (This uses a simple prompt via ChatOpenAI for demonstration.)
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=150
    )
    prompt_template = PromptTemplate(
        template="Generate a SQL query for the following request: {query}",
        input_variables=["query"]
    )
    formatted_prompt = prompt_template.format(query=query)
    sql_query = llm.predict(formatted_prompt)
    return sql_query

# def local_perform_rag(query: str) -> str:
#     """
#     Perform retrieval augmented generation (RAG) on the user's query.
#     (For demonstration, we use a simple LLM prompt. In practice, you might call your own RAG module.)
#     """
#     llm = ChatOpenAI(
#         model="gpt-4o",
#         temperature=0,
#         max_tokens=300
#     )
#     prompt_template = PromptTemplate(
#         template="Answer the following question with relevant context: {query}",
#         input_variables=["query"]
#     )
#     formatted_prompt = prompt_template.format(query=query)
#     answer = llm.predict(formatted_prompt)
#     return answer

# ------------------------------------------------------------------------------
# Function Calling: Extract the Function Call from OpenAI
# ------------------------------------------------------------------------------
def extract_function_call(user_query: str):
    messages = [
        {"role": "system", "content": role_prompt},
        {"role": "user", "content": user_query}
    ]
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        functions=tools,
        function_call="auto",  # Let the model decide if a function should be called
        temperature=0,
        max_tokens=500
    )
    message = response.choices[0].message
    print(message)

    if message.function_call:
        func_call = message.function_call
        func_name = func_call.name
        arguments_str = func_call.arguments or "{}"
        try:
            args = json.loads(arguments_str)
        except Exception as e:
            print("Error parsing function arguments:", e)
            args = {}
        return func_name, args
    else:
        return None, None

# ------------------------------------------------------------------------------
# Main Application Entry Point
# ------------------------------------------------------------------------------

# def main():
#     user_query = input("Enter your query: ")
#     func_name, args = extract_function_call(user_query)
#     if func_name is None:
#         print("No function call was returned. Here is the raw query:")
#         print(user_query)
#     else:
#         if func_name == "generate_sql":
#             query_text = args.get("query", "")
#             result = local_generate_sql(query_text)
#             print("\n[SQL Generation Result]:")
#             print(result)
#             return result
#         elif func_name == "perform_rag":
#             query_text = args.get("query", "")
#             result = main_rag.retrieve_information(query_text)
#             print(result)
#             return result
#         else:
#             print("Unknown function call:", func_name)

# if __name__ == "__main__":
#     main()


# ------------------------------------------------------------------------------
# Streamlit Registration Page
# ------------------------------------------------------------------------------
def registration_page():
    st.title("User Registration")
    with st.form("registration_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        age = st.number_input("Age", min_value=0, max_value=120, value=18)
        submitted = st.form_submit_button("Register")
        if submitted:
            st.session_state["user_info"] = {"name": name, "email": email, "age": age}
            st.success("Registration successful!")
            # st.experimental_rerun()


def chat_page():
    st.title("VendAI Chat")
    if "user_info" not in st.session_state:
        st.warning("Please register first!")
        return
    user_info = st.session_state["user_info"]
    user_context = f"Welcome {user_info['name']}!"  # Simple user context; expand as needed
    st.write(user_context)

    system_prompt = f"""
    You are VendAI, a Super Intelligent Chatbot with Advanced Capabilities.\
    You were developed by the AI team at Vendease to only understand and respond to various queries related to customers orders, customer deliveries, customers purchases, customers information, Vendease product information called SKUs.\
    You use the user's name to always personalize the conversation.\

    Name = {st.session_state["user_info"]["name"]}\
    """
    
    # Initialize chat history if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display existing messages using st.chat_message
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Use st.chat_input to get new user input
    if prompt := st.chat_input("Start typing..."):
        # Append user's message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Extract function call using the new chat message and user context
        func_name, args = extract_function_call(prompt + user_context)
        if func_name is None:
            # Fallback: stream a generic response using conversation history
            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model=st.session_state.get("openai_model", MODEL_ID),
                    messages= [{"role": "system", "content": system_prompt}] +[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            # Dispatch based on function call
            if func_name == "generate_sql":
                query_text = args.get("query", "")
                result = local_generate_sql(query_text)
                with st.chat_message("assistant"):
                    st.markdown(f"**[SQL Generation Result]:** {result}")
                st.session_state.messages.append({"role": "assistant", "content": result})
            elif func_name == "perform_rag":
                query_text = args.get("query", "")
                result = main_rag.retrieve_information(query_text)
                with st.chat_message("assistant"):
                    st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"Unknown function call: {func_name}")
                st.session_state.messages.append({"role": "assistant", "content": f"Unknown function call: {func_name}"})

# ------------------------------------------------------------------------------
# Main Application with Navigation
# ------------------------------------------------------------------------------
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Register", "Chat"))
    if page == "Register":
        registration_page()
    elif page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()