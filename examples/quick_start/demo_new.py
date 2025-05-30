import streamlit as st
import time
from pathlib import Path
import uuid
import json
from flashrag.config import Config
from flashrag.utils import get_retriever, get_generator
from flashrag.prompt import PromptTemplate
import base64
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,6"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none" # 控制 Streamlit 文件热重载机制 的环境变量，主要作用是决定 Streamlit 监听哪些源文件的变动，以自动重新运行 app。
# 配置RAG相关参数
config_dict = {
    "save_note": "demo",
    'data_dir': 'dataset/',
    'index_path': 'indexes/e5_Flat.index',
    'corpus_path': 'indexes/general_knowledge.jsonl',
    'model2path': {'e5': '/home/u2021201791/workspace/Model/intfloat/e5-base-v2'},
    'openai_setting':{'api_key':'sk-40b478c515bf4de38b60511ee63f56b5', 'base_url':'https://api.deepseek.com/v1'},
    'generator_model': 'deepseek-chat',
    'framework': 'openai',
    'retrieval_method': 'e5',
    'metrics': ['em','f1','acc'],
    'retrieval_topk': 1,
    'save_intermediate_data': True
}

# 缓存加载检索器（避免重复加载，提升效率）
@st.cache_resource
def load_retriever(_config):
    return get_retriever(_config)

# 缓存加载生成器
@st.cache_resource
def load_generator(_config):
    return get_generator(_config)

# 预加载配置和模型
@st.cache_resource
def initialize_models():
    """在应用启动时预加载所有模型和配置"""
    config = Config("../methods/my_config.yaml", config_dict=config_dict)
    retriever = load_retriever(config)
    generator = load_generator(config)
    return config, retriever, generator

# 首次访问时初始化会话状态变量
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []  # 保存对话记录
    
if 'memory_chat_history' not in st.session_state:
    st.session_state.memory_chat_history = []  # 用于LLM内部记忆的对话历史
    
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())    # 当前会话唯一标识符
    
if 'user_avatar' not in st.session_state:
    st.session_state.user_avatar = "👤"  # 用户头像默认值
    
if 'assistant_avatar' not in st.session_state:
    st.session_state.assistant_avatar = "🤖"  # 助手机器人头像默认值

if 'query_input' not in st.session_state:
    st.session_state.query_input = ""   # 初始化用户输入内容为空

# 初始化模型加载状态
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    
if 'models_loading' not in st.session_state:
    st.session_state.models_loading = False

    
# 提交问题后清空输入框
def clear_input():
    st.session_state.query_input = ""

# 保存当前对话记录到本地 JSON 文件
def save_conversation():
    conversations_dir = Path("conversations")
    conversations_dir.mkdir(exist_ok=True)  # 如果目录不存在，创建对话目录
    
    conversation_file = conversations_dir / f"{st.session_state.conversation_id}.json"
    
    with open(conversation_file, 'w') as f:
        json.dump({
            "id": st.session_state.conversation_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.conversation_history,
            "memory_chat_history": st.session_state.memory_chat_history  # 保存内部记忆历史
        }, f)

# 读取历史对话记录
def load_conversation(conversation_id):
    conversation_file = Path("conversations") / f"{conversation_id}.json"
    if conversation_file.exists():
        with open(conversation_file, 'r') as f:
            data = json.load(f)
            st.session_state.conversation_history = data["messages"]
            # 加载内部记忆历史，如果不存在则初始化为空列表
            st.session_state.memory_chat_history = data.get("memory_chat_history", [])
            st.session_state.conversation_id = conversation_id

# 启动新的对话并清空历史记录
def start_new_conversation():
    st.session_state.conversation_history = []
    st.session_state.memory_chat_history = []  # 同时清空内部记忆历史
    st.session_state.conversation_id = str(uuid.uuid4())

# 将对话历史格式化为适合LLM内存的格式
def format_chat_history_for_llm(history):
    formatted_messages = []
    for msg in history:
        if msg["role"] == "user":
            formatted_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            formatted_messages.append({"role": "assistant", "content": msg["content"]})
    return formatted_messages

def main():
    # Streamlit页面配置
    custom_theme = {
        "primaryColor": "#0099cc",
        "backgroundColor": "#f0f0f0",
        "secondaryBackgroundColor": "#d3d3d3",
        "textColor": "#121212",
        "font": "sans serif"
    }
    
    # Page configuration
    st.set_page_config(
        page_title="GSAI学生成长助手", 
        page_icon="🤖",
        layout="wide"
    )

    # 自定义前端 CSS 样式，包括实现聊天气泡、底部输入框等
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 6rem;
        }
        .stTextArea textarea {
            border-radius: 20px;
        }
        .user-bubble {
            background-color: #CCE6Fe;
            padding: 10px 15px;
            border-radius: 18px;
            margin: 5px 0;
            display: inline-block;
            max-width: 80%;
            float: right;
            clear: both;
            position: relative;
        }
        .assistant-bubble {
            background-color: #E6F2F9;
            padding: 10px 15px;
            border-radius: 18px;
            margin: 5px 0;
            display: inline-block;
            max-width: 80%;
            float: left;
            clear: both;
            position: relative;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .chat-container {
            overflow-y: auto;
            height: calc(100vh - 250px);
            display: flex;
            flex-direction: column;
        }
        .reference-box {
            background-color: #f8f9fa;
            border-left: 3px solid #66b3d9;
            padding: 10px;
            margin: 10px 0;
            font-size: 0.9em;
        }
        .chat-input {
            position: fixed;
            bottom: 0;
            width: 65%;
            background-color: white;
            padding: 1rem 0;
            z-index: 1000;
            # box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        }
        .new-chat-btn {
            margin-bottom: 20px;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        .memory-info {
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
            font-style: italic;
        }
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }
        /* 输入框聚焦时边框变蓝 */
        .stTextArea textarea:focus {
            outline: none !important;
            border: 2px solid #0099cc !important;
            box-shadow: 0 0 5px rgba(0, 153, 204, 0.5);
        }

        /* 按钮聚焦时字体颜色变蓝 */
        button {
            /* 默认状态：文字左对齐 + 浅灰色字体 */
            text-align: left !important;
            color: #313131 !important;
            background-color: white;
            border: 1px solid #ccc;
            padding: 10px 15px;
            font-size: 14px;
            cursor: pointer;
        }

        /* 悬停状态 */
        button:hover {
            color: #66b3d9 !important;
            border: 1px solid #0099cc !important;
        }

        /* 聚焦状态 */
        button:focus {
            color: #66b3d9 !important;
            background-color: #f8f9fa !important;
            border: none !important;
            outline: none;
            box-shadow: 0 0 5px 2px rgba(0, 153, 204, 0.5) !important;
        }

        .loading-indicator {
            background-color: #e3f2fd;
            border: 1px solid #66b3d9;
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            text-align: center;
            color: #1976d2;
        }
                
    </style>
    """, unsafe_allow_html=True)

    # 在页面开始时就启动模型加载
    col1, main_col, col3 = st.columns([2, 4, 2])
    
    with main_col:
        # 欢迎页面
        if not st.session_state.models_loaded and not st.session_state.models_loading:
            st.session_state.models_loading = True
            
            # 显示加载提示
            loading_placeholder = st.empty()
            with loading_placeholder:
                st.markdown("""
                <div class="loading-indicator">
                    <h3>🚀 正在初始化AI助手...</h3>
                    <p>首次加载需要一些时间，请稍候片刻</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 预加载模型
            try:
                config, retriever, generator = initialize_models()
                st.session_state.config = config
                st.session_state.retriever = retriever
                st.session_state.generator = generator
                st.session_state.models_loaded = True
                st.session_state.models_loading = False
                
                # 清除加载提示
                loading_placeholder.empty()
                
                # 显示加载完成提示（短暂显示后消失）
                success_placeholder = st.empty()
                with success_placeholder:
                    st.success("✅ AI助手已准备就绪！")
                
                # 2秒后清除成功提示并刷新页面
                time.sleep(2)
                success_placeholder.empty()
                st.rerun()
                
            except Exception as e:
                loading_placeholder.empty()
                st.error(f"❌ 模型加载失败: {str(e)}")
                st.session_state.models_loading = False
                return
    
    # 如果正在加载，显示加载界面
    if st.session_state.models_loading:
        st.markdown("""
        <div class="loading-indicator">
            <h3>🚀 正在初始化AI助手...</h3>
            <p>首次加载需要一些时间，请稍候片刻</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # 对话参数
    temperature = 0.5
    topk = 5
    max_new_tokens = 256
    memory_window = 5
    st.session_state.user_avatar = "👤"
    st.session_state.assistant_avatar = "🤖"
    # 左侧边栏设置与历史对话
    with st.sidebar:
        st.title("_GSAI_ :blue[学生成长助手]")
        
        # # 显示模型状态
        # if st.session_state.models_loaded:
        #     st.success("🟢 AI助手已就绪")
        # else:
        #     st.warning("🟡 AI助手加载中...")
        
        # 如果新建对话，则刷新页面新建对话
        if st.button("新建对话", key="new_chat", use_container_width=True):
            start_new_conversation()
            st.rerun()
        
        st.divider()
        
        # 对话历史
        st.subheader("对话历史")
        
        # 左边栏的历史对话
        conversations_dir = Path("conversations")
        conversations_dir.mkdir(exist_ok=True)
        conversation_files = list(conversations_dir.glob("*.json"))

        conversations = []

        # 加载所有对话并提取时间戳信息
        for conv_file in conversation_files:
            try:
                with open(conv_file, 'r') as f:
                    data = json.load(f)
                    timestamp = data.get("timestamp", "")
                    conversations.append((timestamp, data))
            except Exception as e:
                st.error(f"Error loading conversation: {e}")

        # 按 timestamp 逆序排序（越新越靠前）
        conversations.sort(key=lambda x: x[0], reverse=True)

        if conversations:
            for timestamp, data in conversations:
                # 历史对话的 title
                title = data.get("timestamp", "Untitled")
                for msg in data.get("messages", []):
                    if msg["role"] == "user":
                        title = msg["content"][:20] + "..." if len(msg["content"]) > 20 else msg["content"]
                        break

                if st.button(title, key=f"conv_{data['id']}", use_container_width=True):
                    load_conversation(data["id"])
                    st.rerun()
        else:
            st.write("没有先前对话历史")
    
    # 使用三列布局，将中间列用于对话
    col1, main_col, col3 = st.columns([2, 4, 2])
    
    with main_col:
        # 欢迎页面
        if not st.session_state.conversation_history:
            # 读取并编码图片
            image_path = "/home/u2021201791/workspace/FlashRAG/examples/quick_start/ai_logo.jpg"
            try:
                with open(image_path, "rb") as f:
                    data = f.read()
                    encoded = base64.b64encode(data).decode()

                st.markdown(f"""
                <div style="text-align: center; display: flex; align-items: center; margin-top: 200px; width: fit-content; margin-left: auto; margin-right: auto;">
                    <img src="data:image/jpeg;base64,{encoded}"
                        style="height: 120px; margin-right: 20px;" />
                    <div>
                        <h1 style="color: #66b3d9; margin: 0;">欢迎👏我是高瓴AI学生成长助手</h1>
                        <p style="margin-top: 0.3em;">我可以帮你解决学习生活上遇到的问题，不清楚的地方快来问问我吧～</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except FileNotFoundError:
                st.markdown("""
                <div style="text-align: center; margin-top: 200px;">
                    <h1 style="color: #66b3d9;">欢迎👏我是高瓴AI学生成长助手</h1>
                    <p>我可以帮你解决学习生活上遇到的问题，不清楚的地方快来问问我吧～</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 展示对话内容，也就是聊天气泡
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.conversation_history:
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    avatar = "/home/u2021201791/workspace/FlashRAG/examples/quick_start/user_avatar.jpg"
                    with open(avatar, "rb") as f:
                        data = f.read()
                        encoded_user = base64.b64encode(data).decode()
                    st.markdown(f'''
                    <div style="display: flex; justify-content: flex-end; align-items: flex-start; margin: 10px 0;">
                        <div class="user-bubble">{content}</div>
                        <img src="data:image/jpeg;base64,{encoded_user}" style="height: 24px; margin-left: 10px;" />
                    </div>
                    <div style="clear: both;"></div>
                    ''', unsafe_allow_html=True)
                elif role == "assistant":
                    avatar = "/home/u2021201791/workspace/FlashRAG/examples/quick_start/assistant_avatar.jpg"
                    with open(avatar, "rb") as f:
                        data = f.read()
                        encoded_user = base64.b64encode(data).decode()
                    st.markdown(f'''
                    <div style="display: flex; justify-content: flex-start; align-items: flex-start; margin: 10px 0;">
                        <img src="data:image/jpeg;base64,{encoded_user}" style="height: 24px; margin-right: 10px;" />
                        <div class="assistant-bubble">{content}</div>
                    </div>
                    <div style="clear: both;"></div>
                    ''', unsafe_allow_html=True)
                

                    # 显示引用文档
                    if "references" in message:
                        with st.expander("查看答案来源", expanded=False):
                            for i, ref in enumerate(message["references"]):
                                doc_title = ref.get("title", "No Title")
                                doc_text = "\n".join(ref["contents"].split("\n")[1:]) if "contents" in ref else "No content available"
                                st.markdown(f"""
                                <div class="reference-box">
                                    <strong>[{i+1}]: {doc_title}</strong>
                                    <p>{doc_text}</p>
                                </div>
                                """, unsafe_allow_html=True)
    
        # 底部悬浮的输入框区域
        # 只有在模型加载完成后才显示输入框
        if st.session_state.models_loaded:
            query = st.chat_input("随便问点什么")
            if query:
                # Get the stored query
                user_query = query
                
                # 添加新的用户消息到对话历史
                st.session_state.conversation_history.append({
                    "role": "user",
                    "content": user_query
                })
                
                # 同时更新内存对话历史
                st.session_state.memory_chat_history.append({
                    "role": "user",
                    "content": user_query
                })
                
                # 使用预加载的模型实例
                config = st.session_state.config
                generator = st.session_state.generator
                retriever = st.session_state.retriever
                
                # 获取设置中的memory_window值
                memory_window = st.session_state.get('memory_window', 3)
                
                # 只保留最近的几轮对话作为上下文
                recent_memory = st.session_state.memory_chat_history[-2*memory_window:] if len(st.session_state.memory_chat_history) > 2*memory_window else st.session_state.memory_chat_history
                
                # 构建带有对话历史的system prompt
                chat_history_str = ""
                if recent_memory:
                    for msg in recent_memory:
                        role = "Human" if msg["role"] == "user" else "Assistant"
                        chat_history_str += f"{role}: {msg['content']}\n\n"
                
                system_prompt_rag = (
                    "你是一个友好的人工智能助手。"
                    "请对用户的输出做出高质量的响应，生成类似于人类的内容，并尽量遵循输入中的指令。"
                    "\n\n以下是之前的对话历史，请基于这些历史提供连贯的回复：\n\n{chat_history}"
                    "\n\n下面是一些可供参考的文档，你可以使用它们来回答问题：\n\n{reference}"
                )
                
                base_user_prompt = "{question}"
                prompt_template_rag = PromptTemplate(config, system_prompt=system_prompt_rag, user_prompt=base_user_prompt)
                
                # with st.spinner("检索并思考中..."):
                with st.status("正在努力思考中...", expanded=False) as status:
                    with st.spinner("正在检索相应文档...", show_time=True):
                        retrieved_docs = retriever.search(user_query, num=topk)

                    st.write("已检索相关文档")
                    with st.spinner("正在回顾历史信息...", show_time=True):
                        # Generate response with RAG and conversation history
                        input_prompt_with_rag = prompt_template_rag.get_string(
                            question=user_query, 
                            retrieval_result=retrieved_docs,
                            chat_history=chat_history_str  # 添加对话历史到prompt
                        )
                    st.write("已回顾历史信息")
                    with st.spinner("正在生成最终回复...", show_time=True):
                        response_with_rag = generator.generate(
                            input_prompt_with_rag, 
                            temperature=temperature, 
                            max_new_tokens=max_new_tokens
                        )[0]
                    # 把回复添加到UI显示的对话历史
                    st.write("已生成最终回复")
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": response_with_rag,
                        "references": retrieved_docs
                    })
                    
                    # 同时添加到内部记忆
                    st.session_state.memory_chat_history.append({
                        "role": "assistant",
                        "content": response_with_rag
                    })
                    
                    # 保存对话
                    save_conversation()
                    status.update(
                        label="成功!", state="complete", expanded=False
                    )
                    # 重新开始
                    st.rerun()

if __name__ == '__main__':
    main()