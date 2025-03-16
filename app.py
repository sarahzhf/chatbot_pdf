import streamlit as st
import os
import json
import datetime
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import random

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

def send_email_notification(to_email, subject, body):
    """Envoie un e-mail (notification abonnement ou code de validation)."""
    # Adaptez ces paramètres à votre fournisseur SMTP
    from_email = "votre_adresse@example.com"
    smtp_server = "smtp.example.com"
    smtp_port = 587
    smtp_user = "votre_adresse@example.com"
    smtp_password = "motdepasseSMTP"

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
    st.set_page_config(page_title="Chatbot multi-PDF + Vérification Email", layout="wide")
    st.title("Chatbot multi-PDF + Validation d'email + Abonnement")

    users = load_users()

    if "user_email" not in st.session_state:
        st.session_state["user_email"] = None

    # Menu principal (S'inscrire, Valider email, Se connecter)
    if st.session_state["user_email"] is None:
        menu = st.radio("Menu :", ["S'inscrire", "Valider mon email", "Se connecter"])

        if menu == "S'inscrire":
            st.subheader("Créer un compte")
            email_input = st.text_input("Votre email")
            password_input = st.text_input("Mot de passe", type="password")

            if st.button("S'inscrire"):
                if email_input in users:
                    st.error("Un compte existe déjà avec cet e-mail.")
                else:
                    # Générer un code de validation
                    code = str(random.randint(100000, 999999))
                    # Stocker l'utilisateur avec verified=False
                    users[email_input] = {
                        "password": password_input,
                        "verified": False,
                        "verification_code": code,
                        # On ne définit pas encore la date d'abonnement
                        "subscription_end": None
                    }
                    save_users(users)

                    # Envoyer un mail avec le code
                    subject = "Code de validation - Votre inscription"
                    body = f"Bonjour,\n\nVoici votre code de validation : {code}\n\nMerci de le saisir dans l'onglet 'Valider mon email'.\n"
                    send_email_notification(email_input, subject, body)

                    st.success("Compte créé ! Un code de validation vous a été envoyé par email.")

        elif menu == "Valider mon email":
            st.subheader("Valider mon adresse e-mail")
            email_val = st.text_input("Votre email")
            code_val = st.text_input("Code de validation (reçu par email)")

            if st.button("Valider"):
                if email_val in users:
                    user_data = users[email_val]
                    if not user_data["verified"]:
                        if user_data["verification_code"] == code_val:
                            # Email validé
                            user_data["verified"] = True
                            # On donne 1 an d'abonnement à partir d'aujourd'hui
                            end_date = (datetime.date.today() + datetime.timedelta(days=365)).strftime("%Y-%m-%d")
                            user_data["subscription_end"] = end_date
                            save_users(users)
                            st.success("Email validé ! Vous pouvez maintenant vous connecter.")
                        else:
                            st.error("Code de validation incorrect.")
                    else:
                        st.info("Votre email est déjà validé. Vous pouvez vous connecter.")
                else:
                    st.error("Aucun compte trouvé pour cet email.")

        else:  # "Se connecter"
            st.subheader("Se connecter")
            email_input = st.text_input("Email")
            password_input = st.text_input("Mot de passe", type="password")

            if st.button("Se connecter"):
                if email_input in users:
                    user_data = users[email_input]
                    if user_data["password"] == password_input:
                        if not user_data["verified"]:
                            st.error("Vous devez d'abord valider votre adresse e-mail.")
                        else:
                            # Vérifier l'abonnement
                            if user_data["subscription_end"] is None:
                                st.error("Votre abonnement n'est pas défini. Contactez l'administrateur.")
                            else:
                                end_date_str = user_data["subscription_end"]
                                if not is_subscription_valid(end_date_str):
                                    st.error("Votre abonnement est expiré. Veuillez le renouveler.")
                                else:
                                    days_left = days_until_expiration(end_date_str)
                                    st.session_state["user_email"] = email_input
                                    st.success(f"Connexion réussie ! Il vous reste {days_left} jours d'abonnement.")

                                    # Si l'abonnement se termine dans < 10 jours, envoi d'un mail
                                    if days_left < 10:
                                        subject = "Abonnement proche de l'expiration"
                                        body = (
                                            f"Bonjour,\n\n"
                                            f"Il vous reste {days_left} jours avant l'expiration de votre abonnement.\n"
                                            f"Renouvelez vite !\n\n"
                                            f"Cordialement,\nL'équipe"
                                        )
                                        send_email_notification(email_input, subject, body)
                                        st.info("Un e-mail de rappel vous a été envoyé.")

                                    st.rerun()
                    else:
                        st.error("Mot de passe incorrect.")
                else:
                    st.error("Aucun compte trouvé pour cet e-mail.")

        return  # On arrête ici si l'utilisateur n'est pas connecté

    # Si l'utilisateur est connecté
    user_email = st.session_state["user_email"]
    st.write(f"Connecté en tant que : **{user_email}**")
    if st.button("Se déconnecter"):
        st.session_state["user_email"] = None
        st.rerun()

    ###################################
    # Gérer le chatbot (déjà validé + abonnement OK)
    ###################################
    if "all_docs" not in st.session_state:
        st.session_state["all_docs"] = []
    if "chain" not in st.session_state:
        st.session_state["chain"] = None
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

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
