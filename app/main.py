from finsoros import FinSorosBot
from hyperparams import HYPERPARAMS
import streamlit as st
from inital_prompts import FeedParser
import random
from PIL import Image


import time
from enum import Enum

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


class Actor(Enum):
    USER = "user"
    ASSISTANT = "assistant"

BOT_AVATAR = Image.open('botimage.png')  
USER_AVATAR = Image.open('user.png')
IMAGES = {
    Actor.ASSISTANT.name: BOT_AVATAR,
    Actor.USER.name: USER_AVATAR
}


st.set_page_config(page_title="FinSoros", layout="centered")
st.title("FinSoros")
st.logo(Image.open("botlogo.png"), size="large")
# st.logo(BOT_AVATAR)

st.caption("I answer question with insights from George Soros trading strategies. \n Note: The knowledge is limited to bot and refrain asking simple questions")

HPARAMS = HYPERPARAMS(
  save_model="model.h5",
  max_samples= 50000,
  vocab_size=8151,
  num_units=2048,
  batch_size=80,
  max_length=65,
  num_layers=4,
  dropout=0.05,
  num_heads=8,
  d_model=512,
)








def send_message(role):
    return st.chat_message(role)


def call_user_selection(q):
    print("Called user button")
    with send_message(Actor.USER.name):
        st.markdown(q)
    with st.spinner("Thinking..."):
      response = st.session_state.finsoros.get_response(q)
    with send_message(Actor.ASSISTANT.name):
      st.write_stream(response_generator(response))


if "finsoros" not in st.session_state:
    with st.spinner("Booting Model..."):
      finsoros = FinSorosBot(HPARAMS)
      st.session_state.finsoros:FinSorosBot = finsoros
      st.session_state.feedparser:FeedParser = FeedParser()

def click_prompt(x):
    def click():
        st.session_state.user_prompt = x
    return click

if ("user_prompt" not in st.session_state) and not st.session_state.finsoros.memory.last_conversation():
    questions = st.session_state.feedparser.get_random_feeds(st.session_state.finsoros.flan)
    st.title("💬 Start a Conversation")
    st.markdown("### Choose a prompt to begin:")
    col1, col2, col3 = st.columns(3)

    with col1:
        q = (f"{questions[0]}" or 'How can I use futures contracts as part of a pair trading strategy?')
        st.button(f"💡 "+ q, on_click=click_prompt(q))

    with col2:
        q = (questions[1] or "How has the relationship between cyclical and defensive stocks changed given recent economic data?")
        st.button(f"🤖 "+ q, on_click=click_prompt(q))

    with col3:
        q = (f"{questions[2]}" or 'How can I hedge market risk while maintaining exposure to specific pair trades?')
        st.button("🌍 "+ q, on_click=click_prompt(q))

elif ("user_prompt" in st.session_state) and (st.session_state.user_prompt != "cleared"):
    call_user_selection(st.session_state.user_prompt)
    st.session_state.user_prompt = "cleared"
else:
  for message in st.session_state.finsoros.memory.conversation_history:
      with send_message(Actor.USER.name):
          st.markdown(message["question"])
      with send_message(Actor.ASSISTANT.name, ):
          st.markdown(message["answer"])

# if 'user_prompt' in st.session_state:
    
#     del st.session_state.user_prompt
    

if prompt := st.chat_input("Gold, Currencies, Markets, Finance..."):
    with send_message(Actor.USER.name):
        st.markdown(prompt)
    with st.spinner("Thinking..."):
      response = st.session_state.finsoros.get_response(prompt)
    with send_message(Actor.ASSISTANT.name):
      response = st.write_stream(response_generator(response))


