import streamlit as st
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleWebPageReader, Settings
from llama_index.embeddings.gemini import GeminiEmbedding


st.set_page_config(page_title="幸福人生之路", page_icon="🌱", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("大白在线")
st.info("有一条更好的路", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "人生困境左右都找不到路的时候，有一条更好的路!",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    # Create an instance of SimpleWebPageReader
    reader = SimpleWebPageReader()
    
    # URL of the web page you want to read
    url = "https://chinesefamily.org"  # Replace with the target URL
    
    # Read the content from the web page
    docs = reader.load_data(url)
    
    # Display the extracted content
    # for element in docs:
    #     print(element)  # Each element is typically a dictionary with the content

    Settings.chunk_size = 1500
    Settings.chunk_overlap = 50
    Settings.embed_model = GeminiEmbedding()
    
    Settings.llm = Gemini(
        model="models/gemini-1.5-flash",
        # the higher the more creative, the lower more firm
        temperature=0.2,
        # system_prompt="""You are a an expert on the work of Rabindrath Tagore, and you love to use quotations from his booksto illustrate your points.
        # Answer the question using the provided documents, which contain relevant excerpts from the work of Rabindrath Tagore.
        # The context for all questions is the work of Rabindrath Tagore. Whenver possible, include a quotation from the provided excerpts of his work to illustrate your point.
        # Respond using a florid but direct tone, typical of an early modernist writer.
        # Keep your answers under 100 words.""",

        #if only want to use local documents, change the prompt here to say if you can't answer from the local document, then just say no
        # put more strict ristrictions here
        system_prompt="""You are a an expert on bibilical conseling, and you love to help people to seek hapiness and peace. 
        Answer the question using the provided documents, which contain relevant excerpts from some spiritual growth books.
        Whenver possible, include a quotation from the provided excerpts of his work to illustrate your point.
        Respond using a florid, warm, encouraging but direct tone, typical of an old wise and kind counsellor. 
        Detect the input language and answer back in the same language.
        Keep your answers under 150 words.""",
        
        api_key = st.secrets.google_gemini_key,
        safe = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
],
    )
 

    index = VectorStoreIndex.from_documents(docs)
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context", verbose=True, streaming=False,
    )

if prompt := st.chat_input(
    "问一个问题"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = ""
        try:
            response_stream = st.session_state.chat_engine.stream_chat(prompt)
        except:
            st.error("We got an error from Google Gemini - this may mean the question had a risk of producing a harmful response. Consider asking the question in a different way.")        
        if response_stream != "":
            with st.spinner("waiting"):
                try:
                    st.write_stream(response_stream.response_gen)
                except:
                    st.error("We hit a bump - let's try again")
                    try:
                        resp = st.session_state.chat_engine.chat(prompt)[0]
                        st.write(resp)
                    except:
                        st.error("We got an error from Google Gemini - this may mean the question had a risk of producing a harmful response. Consider asking the question in a different way.")
            message = {"role": "assistant", "content": response_stream.response}
            # Add response to message history
            st.session_state.messages.append(message)
