from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.chains import ConversationChain
import os

# 専門家ごとのプロンプトテンプレート
templates = {
    "心理カウンセラー": """
    あなたは心理カウンセラーとして振る舞い、ユーザーの感情や心理的な悩みに寄り添いながら、
    優しく丁寧にアドバイスを提供してください。
    会話履歴:
    {chat_history}
    入力: {input}
    出力:
    """,
    "キャリアアドバイザー": """
    あなたはキャリアアドバイザーとして振る舞い、ユーザーの職業やキャリアに関する悩みに対して、
    実践的かつ具体的なアドバイスを提供してください。
    会話履歴:
    {chat_history}
    入力: {input}
    出力:
    """
}

# LangChainのChatモデルとメモリを初期化
if "chain" not in st.session_state:
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.5, openai_api_key=os.getenv("OPENAI_API_KEY"))
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state["chain"] = {
        expert: ConversationChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["chat_history", "input"],
                template=template
            ),
            memory=memory
        ) for expert, template in templates.items()
    }

st.title("お悩み相談アプリ")

# アプリの概要や操作方法を表示
st.markdown("""
### 
このアプリでは、心理カウンセラー・キャリアアドバイザーといった専門家に相談することができます。
入力フォームに相談内容を記入し、専門家の種類を選択して「送信」ボタンを押してください。

### 操作方法
1. **専門家を選択**: 「心理カウンセラー」または「キャリアアドバイザー」を選択してください。
2. **相談内容を入力**: 入力フォームに相談内容を記入してください。
3. **送信ボタンを押す**: AIが相談内容に基づいて回答を表示します。
""")

# ラジオボタンで専門家の種類を選択
expert_type = st.radio(
    "どの専門家に相談しますか？",
    ("心理カウンセラー", "キャリアアドバイザー")
)

# 入力フォーム
input_text = st.text_input("相談内容を入力してください：", placeholder="例：仕事の悩みについて相談したいです")

# LLMからの回答を取得する関数
def get_response(input_text, expert_type):
    """
    入力テキストと専門家の種類を基にLLMからの回答を取得する関数。
    """
    return st.session_state["chain"][expert_type].predict(input=input_text)

# 送信ボタン
if st.button("送信"):
    if input_text:
        # 関数を利用してLLMからの回答を取得
        response = get_response(input_text, expert_type)
        # 回答を表示
        st.write(response)
    else:
        st.warning("相談内容を入力してください！")