import streamlit as st
from langchain_core.messages import HumanMessage
import uuid 

from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_huggingface import HuggingFaceEndpoint
import requests
import streamlit as st

# ================================
# Hugging Face LLM Configuration
# ================================

HF_API_TOKEN = st.secrets["HUGGINGFACE_API_KEY"]

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",   # You can change model
    huggingfacehub_api_token=HF_API_TOKEN,
    temperature=0.4,
    max_new_tokens=512,
)

# ================================
# Tools
# ================================

@tool
def search_about_movies(query: str):
    """
    Use this tool to search specifically for new, trending, or specific movies
    using DuckDuckGo. Useful for finding movies based on user queries.
    """
    search = DuckDuckGoSearchRun(region='in')
    search_query = f"{query} movie list"
    return search.invoke(search_query)


@tool
def get_movie_data(movie_title: str) -> str:
    """
    Fetch details for a movie (Title, Year, Rating, Plot) using the OMDb API.
    Input should be the name of the movie (e.g. 'Inception', 'The Matrix').
    """
    api_key = st.secrets["OMDB_API_KEY"]

    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}"

    try:
        r = requests.get(url)
        data = r.json()

        if data.get("Response") == "False":
            return "❌ Movie not found."

        return f"""
🎬 **{data.get("Title")} ({data.get("Year")})**

⭐ Rating: {data.get("imdbRating")}
🎭 Genre: {data.get("Genre")}
🎬 Director: {data.get("Director")}
👥 Actors: {data.get("Actors")}

📝 Plot:
{data.get("Plot")}

🏆 Awards: {data.get("Awards")}
💰 Box Office: {data.get("BoxOffice")}
🖼 Poster: {data.get("Poster")}
"""

    except Exception as e:
        return f"⚠️ Error fetching movie data: {str(e)}"


tools = [search_about_movies, get_movie_data]
llm_with_tools = llm.bind_tools(tools)

# ================================
# LangGraph State
# ================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)
checkpointer = MemorySaver()

# ================================
# Graph Definition
# ================================

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# ================================
# Helper
# ================================

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)



def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_theard(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_theard(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get("messages", [])

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []
add_theard(st.session_state['thread_id'])



st.sidebar.title('LangGraph Chatbot')
if st.sidebar.button('New Chat',width='stretch'):
    reset_chat()

st.sidebar.header('Old Chat')
for thread_id in st.session_state['chat_threads']:
   if st.sidebar.button(str(thread_id)):
       st.session_state['thread_id'] = thread_id
       messages = load_conversation(thread_id)

       temp_messages = []

       for msg in messages:
           if isinstance(msg, HumanMessage):
               role='user'
           else:
               role='assistant'
           temp_messages.append({'role':role, 'content': msg.content})
        
       st.session_state['message_history'] = temp_messages
       
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("type Here")

if user_input:
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config = CONFIG,
                stream_mode='messages'
            )
            if metadata.get("langgraph_node") != "tools" 
        )
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
