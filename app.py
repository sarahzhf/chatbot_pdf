import streamlit as st
import os
import json
import datetime
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()  # Charge les variables d'environnement (ex. pour OPENAI_API_KEY)

###################################
# Gestion des utilisateurs et abonnement
###################################
def load_users():
    """Charge le dictionnaire {email: {...}} depuis un fichier JSON (users.json)."""
    if os.path.exists("users.json"):
        with open("users.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users):
    """Sauvegarde le dictionnaire {email: {...}} dans un fichier JSON (users.json)."""
    with open("users.json", "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def is_subscription_valid(end_date_str):
    """Vérifie si la date de fin d'abonnement n'est pas dépassée."""
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    return end_date >= datetime.date.today()

def days_until_expiration(end_date_str):
    """Retourne le nombre de jours avant expiration."""
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    delta = end_date - datetime.date.today()
    return delta.days

def send_email_notification(to_email, days_left):
    """Envoie un e-mail de notification pour prévenir de la fin prochaine de l'abonnement."""
    # Adaptez ces paramètres à votre fournisseur SMTP
    from_email = "votre_adresse@example.com"
    smtp_server = "smtp.example.com"
    smtp_port = 587
    smtp_user = "votre_adresse@example.com"
    smtp_password = "motdepasseSMTP"

    subject = "Votre abonnement arrive bientôt à expiration"
    body = (
        f"Bonjour,\n\n"
        f"Il vous reste {days_left} jours avant l'expiration de votre abonnement.\n"
        f"Renouvelez vite !\n\n"
        f"Cordialement,\nL'équipe"
    )

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        print(f"E-mail envoyé à {to_email}")
    except Exception as e:
        print("Erreur lors de l'envoi de l'email:", e)

###################################
# Imports LangChain pour le chatbot
###################################
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
    # Récupérer la clé OpenAI depuis st.secrets
    openai_key = st.secrets["default"]["OPENAI_API_KEY"]

    # Étape 1 : découpage en segments
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitted_docs = text_splitter.split_documents(docs)

    # Étape 2 : création des embeddings et du vector store
    embeddings = OpenAIEmbeddings(api_key=openai_key)
    vectorstore = FAISS.from_documents(splitted_docs, embeddings)

    # Étape 3 : mémoire conversationnelle
    if existing_memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    else:
        memory = existing_memory

    # Étape 4 : création de la chaîne (LLM)
    llm = OpenAI(api_key=openai_key)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain

###################################
# Application Streamlit principale
###################################
def main():
    st.set_page_config(page_title="Chatbot multi-PDF avec abonnement", layout="wide")
    st.title("Chatbot multi-PDF + Abonnement par e-mail")

    # Charger la liste des utilisateurs
    users = load_users()

    # Vérifier si l'utilisateur est déjà connecté
    if "user_email" not in st.session_state:
        st.session_state["user_email"] = None

    # Menu principal (Inscription ou Connexion)
    if st.session_state["user_email"] is None:
        menu = st.radio("Menu :", ["Inscription", "Connexion"])
        if menu == "Inscription":
            st.subheader("Créer un compte")
            email_input = st.text_input("Votre email")
            password_input = st.text_input("Mot de passe", type="password")
            # On donne 1 an d'abonnement par défaut
            default_end = (datetime.date.today() + datetime.timedelta(days=365)).strftime("%Y-%m-%d")

            if st.button("S'inscrire"):
                if email_input in users:
                    st.error("Un compte existe déjà avec cet e-mail.")
                else:
                    users[email_input] = {
                        "password": password_input,
                        "subscription_end": default_end
                    }
                    save_users(users)
                    st.success("Inscription réussie ! Vous pouvez maintenant vous connecter.")

        else:  # "Connexion"
            st.subheader("Se connecter")
            email_input = st.text_input("Email")
            password_input = st.text_input("Mot de passe", type="password")

            if st.button("Se connecter"):
                if email_input in users and users[email_input]["password"] == password_input:
                    end_date_str = users[email_input]["subscription_end"]
                    if not is_subscription_valid(end_date_str):
                        st.error("Votre abonnement est expiré. Veuillez le renouveler.")
                    else:
                        days_left = days_until_expiration(end_date_str)
                        st.session_state["user_email"] = email_input
                        st.success(f"Connexion réussie ! Il vous reste {days_left} jours d'abonnement.")
                        
                        # Envoi d'un mail si l'abonnement se termine dans < 10 jours
                        if days_left < 10:
                            send_email_notification(email_input, days_left)
                            st.info("Un e-mail de rappel vous a été envoyé.")

                        st.experimental_rerun()  # recharger la page pour afficher le chatbot
                else:
                    st.error("Identifiants invalides.")

        return  # On arrête ici si l'utilisateur n'est pas connecté

    # Si l'utilisateur est connecté et a un abonnement valide, on affiche le chatbot
    user_email = st.session_state["user_email"]
    st.write(f"Connecté en tant que : **{user_email}**")
    if st.button("Se déconnecter"):
        st.session_state["user_email"] = None
        st.experimental_rerun()

    # Gérer le chatbot
    # Initialiser la session chatbot si pas déjà fait
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

    if st.button("Ajouter ces documents à la collection") and uploaded_files:
        for i, up_file in enumerate(uploaded_files):
            temp_filename = f"temp_{i}.pdf"
            with open(temp_filename, "wb") as f:
                f.write(up_file.read())
            loader = PyPDFLoader(temp_filename)
            new_docs = loader.load()
            st.session_state["all_docs"].extend(new_docs)
        st.success("Documents ajoutés ! Vous pouvez maintenant (re)indexer.")

    if st.button("Réindexer tous les documents"):
        if len(st.session_state["all_docs"]) == 0:
            st.warning("Aucun document n'a été chargé.")
        else:
            st.session_state["chain"] = create_chain(
                st.session_state["all_docs"],
                existing_memory=st.session_state["memory"]
            )
            st.success("Réindexation terminée !")

    user_question = st.text_input("Votre question :")
    if st.session_state["chain"] and user_question:
        result = st.session_state["chain"]({"question": user_question})
        st.markdown("**Réponse :**")
        st.write(result["answer"])

if __name__ == "__main__":
    main()
