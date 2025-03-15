import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Imports LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def create_chain(docs, existing_memory=None):
    """
    Crée ou recrée la chaîne de question/réponse avec mémoire,
    à partir d'une liste de documents (docs).
    existing_memory peut être un objet ConversationBufferMemory
    qu'on souhaite conserver (pour ne pas perdre l'historique).
    """

    # Étape 1 : découpage en segments
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitted_docs = text_splitter.split_documents(docs)

    # Étape 2 : création des embeddings et du vector store
    embeddings = OpenAIEmbeddings()  # nécessite OPENAI_API_KEY
    vectorstore = FAISS.from_documents(splitted_docs, embeddings)

    # Étape 3 : mémoire conversationnelle
    if existing_memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    else:
        memory = existing_memory

    # Étape 4 : création de la chaîne
    llm = OpenAI()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain

def main():
    st.set_page_config(page_title="Chatbot multi-PDF persistant", layout="wide")
    st.title("Chatbot multi-PDF avec conservation des documents et de la mémoire")

    # Initialiser la session
    if "all_docs" not in st.session_state:
        st.session_state["all_docs"] = []  # liste de tous les documents PDF
    if "chain" not in st.session_state:
        st.session_state["chain"] = None
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    # Barre latérale pour afficher l'historique
    with st.sidebar:
        st.header("Historique de la conversation")
        if st.session_state["chain"] is not None:
            chat_history = st.session_state["memory"].load_memory_variables({})["chat_history"]
            with st.expander("Afficher / Masquer l'historique"):
                for i, message in enumerate(chat_history):
                    role = "Utilisateur" if message.type == "user" else "Assistant"
                    st.markdown(f"**{role}** : {message.content}")
                    if i < len(chat_history) - 1:
                        st.markdown("---")

    # Téléversement de plusieurs PDF
    uploaded_files = st.file_uploader(
        "Chargez vos PDFs (ils s'ajoutent aux existants)",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Bouton pour ajouter les PDFs téléversés à la liste générale
    if st.button("Ajouter ces documents à la collection") and uploaded_files:
        for i, up_file in enumerate(uploaded_files):
            # On charge le PDF dans des Documents
            temp_filename = f"temp_{i}.pdf"
            with open(temp_filename, "wb") as f:
                f.write(up_file.read())
            loader = PyPDFLoader(temp_filename)
            new_docs = loader.load()
            st.session_state["all_docs"].extend(new_docs)
        st.success("Documents ajoutés ! Vous pouvez maintenant (re)indexer.")

    # Bouton pour réindexer tous les documents (anciens + nouveaux)
    if st.button("Réindexer tous les documents"):
        if len(st.session_state["all_docs"]) == 0:
            st.warning("Aucun document n'a été chargé.")
        else:
            # On recrée la chaîne en conservant la mémoire existante
            st.session_state["chain"] = create_chain(
                st.session_state["all_docs"],
                existing_memory=st.session_state["memory"]
            )
            st.success("Réindexation terminée !")

    # Champ de question
    user_question = st.text_input("Votre question :")

    # Traitement de la question
    if st.session_state["chain"] and user_question:
        result = st.session_state["chain"]({"question": user_question})
        st.markdown("**Réponse :**")
        st.write(result["answer"])

if __name__ == "__main__":
    main()
