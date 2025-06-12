import streamlit as st
import os
import json
import sqlite3
from datetime import datetime
import hashlib
from typing import List, Dict, Optional
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
import openai

from dotenv import load_dotenv
load_dotenv()  # Loads .env into os.environ

# Configuration
class Config:
    CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', '')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    DB_PATH = 'chat_memory.db'
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Small, fast model
    MAX_CONTEXT_LENGTH = 4000  # Keep costs low
    SIMILARITY_THRESHOLD = 0.7
    
    # Model configurations with pricing info (approximate)
    CLAUDE_MODELS = {
        'claude-opus-4-20250514': {'name': 'Claude 4 Opus', 'cost': 'Premium', 'best_for': 'Complex analysis'},
        'claude-sonnet-4-20250514': {'name': 'Claude 4 Sonnet', 'cost': 'Medium', 'best_for': 'Balanced tasks'},
        'claude-3-5-haiku-20241022': {'name': 'Claude 3.5 Haiku', 'cost': 'Low', 'best_for': 'Fast responses'},
        'claude-3-haiku-20240307': {'name': 'Claude 3 Haiku', 'cost': 'Very Low', 'best_for': 'Budget-friendly'}
    }
    
    OPENAI_MODELS = {
        'gpt-4o-mini': {'name': 'GPT-4o Mini', 'cost': 'Low', 'best_for': 'General tasks'},
        'o1-mini': {'name': 'o1 Mini', 'cost': 'Medium', 'best_for': 'Reasoning'},
        'gpt-4o-mini-search-preview': {'name': 'GPT-4o Mini Search', 'cost': 'Medium', 'best_for': 'Search tasks'},
        'gpt-3.5-turbo': {'name': 'GPT-3.5 Turbo', 'cost': 'Very Low', 'best_for': 'Budget-friendly'},
        'gpt-4.1': {'name': 'GPT-4.1', 'cost': 'Flagship', 'best_for': 'Complex tasks'},
        'gpt-4.1-mini': {'name': 'GPT-4.1 Mini', 'cost': 'Balanced', 'best_for': 'Intelligence, speed, and cost'}
    }

class MemoryManager:
    def __init__(self):
        self.db_path = Config.DB_PATH
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for memory storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                content_hash TEXT UNIQUE,
                category TEXT DEFAULT 'general'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                model_used TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_memory(self, content: str, category: str = 'general'):
        """Add new memory with embedding"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if content already exists
        cursor.execute('SELECT id FROM memories WHERE content_hash = ?', (content_hash,))
        if cursor.fetchone():
            conn.close()
            return
        
        # Generate embedding
        embedding = self.embedding_model.encode([content])[0]
        embedding_blob = embedding.tobytes()
        
        cursor.execute('''
            INSERT INTO memories (content, embedding, content_hash, category)
            VALUES (?, ?, ?, ?)
        ''', (content, embedding_blob, content_hash, category))
        
        conn.commit()
        conn.close()
    
    def search_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for relevant memories using semantic similarity"""
        query_embedding = self.embedding_model.encode([query])[0]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT content, embedding, timestamp, category FROM memories')
        memories = cursor.fetchall()
        conn.close()
        
        if not memories:
            return []
        
        # Calculate similarities
        similarities = []
        for content, embedding_blob, timestamp, category in memories:
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            
            if similarity > Config.SIMILARITY_THRESHOLD:
                similarities.append({
                    'content': content,
                    'similarity': similarity,
                    'timestamp': timestamp,
                    'category': category
                })
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:limit]
    
    def save_conversation(self, user_message: str, ai_response: str, model_used: str):
        """Save conversation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (user_message, ai_response, model_used)
            VALUES (?, ?, ?)
        ''', (user_message, ai_response, model_used))
        
        conn.commit()
        conn.close()

class AIManager:
    def __init__(self):
        self.claude_client = anthropic.Anthropic(api_key=Config.CLAUDE_API_KEY) if Config.CLAUDE_API_KEY else None
        self.openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY) if Config.OPENAI_API_KEY else None
    
    def get_claude_response(self, prompt: str, context: str = "", model: str = "claude-3-5-haiku-20241022") -> str:
        """Get response from Claude API with model selection"""
        if not self.claude_client:
            return "Claude API key not configured"
        
        full_prompt = f"{context}\n\nUser: {prompt}" if context else prompt
        
        try:
            # Adjust max_tokens based on model capability
            max_tokens = 8000 if 'opus' in model else 4000 if 'sonnet' in model else 2000
            
            response = self.claude_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Claude API Error: {str(e)}"
    
    def get_gpt_response(self, prompt: str, context: str = "", model: str = "gpt-4o-mini") -> str:
        """Get response from GPT API with model selection"""
        if not self.openai_client:
            return "OpenAI API key not configured"
        
        messages = []
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Adjust max_tokens and temperature based on model
            max_tokens = 8000 if 'o1' in model else 4000
            temperature = 0.1 if 'o1' in model else 0.7  # o1 models use lower temperature
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API Error: {str(e)}"
    
    def auto_select_model(self, prompt: str) -> tuple:
        """Intelligently select the best model based on prompt characteristics"""
        prompt_lower = prompt.lower()
        prompt_length = len(prompt)
        
        # Keywords that suggest complex reasoning
        complex_keywords = [
            'analyze', 'explain', 'complex', 'detailed', 'research', 'compare', 'evaluate', 'diagnose',
            'theory', 'mechanism', 'how does', 'why', 'meta-cognition', 'emergent', 'decompose', 'model', 
            'breakdown', 'derive', 'philosophy', 'argument', 'counterfactual', 'paradox', 'proof', 'debate', 'socratic'
        ]

        innovation_keywords = [
            'innovate', 'create', 'invent', 'blueprint', 'design', 'build', 'prototype', 'quantum', 'AGI', 'ASI', 'LLM',
            'Large Language Model', 'AI', 'artificial intelligence', 'machine learning', 'deep learning', 'Artificial General Intelligence',
            'Artificial Superintelligence', 'neural network', 'deep neural network', 'transformer model', 'self-supervised learning',
            'architecture', 'pipeline', 'orchestrate', 'fusion', 'breakthrough', 'paradigm', 'novel', 'paradigm shift',
            'systemic', 'recursive', 'neural', 'transformer', 'cybernetic', 'genius', 'synthesize', 'meta', 'emergent'
        ]

        reasoning_keywords = [
            'solve', 'solution', 'calculate', 'logic', 'step by step', 'problem', 'math', 'deduce', 'compute',
            'estimate', 'optimize', 'proof', 'decision', 'algorithm', 'reason', 'sequence', 'pattern', 'puzzle', 'explain reasoning',
            'deductive', 'inductive', 'proof', 'conjecture'
        ]

        search_keywords = [
            'search', 'find', 'lookup', 'recent', 'current', 'news', 'latest', 'trending', 'today', 'update', 'event',
            'now', 'happening', 'information', 'data', 'facts', 'insight', 'source', 'summary', 'trend', 'realtime', 'live'
        ]

        
        # Enhanced keyword mapping
        quantum_advanced = [
            'quantum', 'superposition', 'entanglement', 'meta-parallelization', 'supermind', 'neuroflow', 'chronocortex', 
            'multi-agent', 'sentient', 'emergent', 'singularity', 'hyperdimensional', 'fractal', 'nonlinear', 'topological',
            'hypergraph', 'alien', 'beyond human', 'exo-cortex', 'time dilation', 'infinite', 'self-improving', 'infinite recursion'
        ]

        startup_keywords = [
            'startup', 'MVP', 'YC', 'Y Combinator', 'demo day', 'product-market fit', 'pitch', 'fundraise', 'incubator',
            'accelerator', 'growth hack', 'scale', 'bootstrap', 'venture', 'unicorn', 'exit', 'founder', 'deck', 'go to market'
        ]

        productivity_keywords = [
            'habit', 'tracker', 'optimize', 'schedule', 'calendar', 'workflow', 'automation', 'routine', 'efficiency', 
            'system', 'focus', 'pomodoro', 'checklist', 'deadline', 'prioritize', 'goal', 'objective', 'review', 'task',
            'kanban', 'reminder', 'sprint', 'execution'
        ]


        research_heavy = ['crypto competition', 'kaggle', 'cutting edge', 'breakthrough', 'ascended',
            'research', 'study', 'paper', 'whitepaper', 'arxiv', 'publication', 'hypothesis', 'empirical', 'statistical',
            'experiment', 'dataset', 'survey', 'citation', 'reference', 'findings', 'result', 'discover', 'review', 'meta-analysis',
            'benchmark', 'leaderboard', 'sota', 'evaluation', 'literature', 'analysis', 'report'
        ]

        dev_keywords = [
            'AI', 'machine learning', 'deep learning', 'python', 'code', 'script', 'repo', 'github', 'open source',
            'deploy', 'docker', 'api', 'SDK', 'train', 'inference', 'LLM', 'weights', 'benchmark', 'dataset', 'label', 'feature engineering',
            'autonomous', 'agent', 'LangChain', 'AutoGen', 'vLLM', 'mistral', 'llama', 'qwen', 'deepseek', 'transformers'
        ]

        architecture = ['master stack', 'overhaul', 'system design', 'infrastructure']
        creative_dev = ['mini project', 'dashboard', 'plugin', 'tracker', 'generator']
        rapid_proto = ['quick', 'prototype', 'test', 'iterate', 'emoji', 'password']

        
        # Check for complexity indicators
        is_complex = any(keyword in prompt_lower for keyword in complex_keywords)
        is_innovation = any(keyword in prompt_lower for keyword in innovation_keywords)
        needs_reasoning = any(keyword in prompt_lower for keyword in reasoning_keywords)
        needs_search = any(keyword in prompt_lower for keyword in search_keywords)
        is_quantum_advanced = any(keyword in prompt_lower for keyword in quantum_advanced)
        is_startup_related = any(keyword in prompt_lower for keyword in startup_keywords)
        is_productivity_related = any(keyword in prompt_lower for keyword in productivity_keywords)
        is_research_heavy = any(keyword in prompt_lower for keyword in research_heavy)
        is_dev_related = any(keyword in prompt_lower for keyword in dev_keywords)
        is_architecture_related = any(keyword in prompt_lower for keyword in architecture)
        is_creative_dev = any(keyword in prompt_lower for keyword in creative_dev)
        is_rapid_proto = any(keyword in prompt_lower for keyword in rapid_proto)
        # Determine if the prompt is long
        is_long = prompt_length > 500


      
           # Calculate complexity scores for multi-dimensional analysis
        complexity_score = sum([
            is_complex * 3,
            is_innovation * 2,
            needs_reasoning * 2,
            is_quantum_advanced * 4,
            is_research_heavy * 3,
            is_architecture_related * 2,
            is_long * 1
        ])
        
        # Innovation potential score
        innovation_score = sum([
            is_innovation * 3,
            is_quantum_advanced * 4,
            is_startup_related * 2,
            is_dev_related * 2,
            any(keyword in prompt_lower for keyword in ['breakthrough', 'novel', 'paradigm']) * 3
        ])
        
        # Speed requirement score (inverse priority)
        speed_priority = sum([
            is_rapid_proto * 3,
            is_creative_dev * 2,
            is_productivity_related * 2,
            prompt_length < 100 * 2
        ])
        
        # HIERARCHICAL DECISION TREE
        
        # Tier 1: Critical Search & Real-time Requirements
        if needs_search:
            if complexity_score > 8 or is_research_heavy:
                return 'claude', 'claude-sonnet-4-20250514'  # Claude better for complex search synthesis
            return 'openai', 'gpt-4o-mini-search-preview'
        
        # Tier 2: Pure Reasoning & Logic Problems
        if needs_reasoning and not (is_innovation or is_quantum_advanced):
            if complexity_score > 6:
                return 'openai', 'o1'  # Full o1 for complex reasoning
            return 'openai', 'o1-mini'
        
        # Tier 3: Quantum/Hyperdimensional/ASI-level Tasks
        if is_quantum_advanced or innovation_score > 8:
            if complexity_score > 10:
                return 'claude', 'claude-opus-4-20250514'  # Opus for bleeding-edge innovation
            elif is_research_heavy or is_architecture_related:
                return 'openai', 'gpt-4.1'  # 4.1 for quantum research
            return 'claude', 'claude-sonnet-4-20250514'  # Sonnet for quantum prototyping
        
        # Tier 4: High-Stakes Development & Research
        if is_research_heavy and (complexity_score > 6 or is_long):
            if is_startup_related:
                return 'openai', 'gpt-4.1'  # 4.1 for high-stakes startup research
            return 'claude', 'claude-sonnet-4-20250514'  # Claude for deep research analysis
        
        # Tier 5: Innovation & Creative Development
        if is_innovation or innovation_score > 4:
            if is_architecture_related or complexity_score > 8:
                return 'claude', 'claude-opus-4-20250514'  # Opus for innovative architecture
            elif is_startup_related:
                return 'openai', 'gpt-4.1-mini'  # 4.1 Mini for startup innovation balance
            return 'claude', 'claude-sonnet-4-20250514'  # Sonnet for general innovation
        
        # Tier 6: Development & Technical Tasks
        if is_dev_related:
            if complexity_score > 6 or is_architecture_related:
                return 'openai', 'gpt-4.1'  # 4.1 for complex development
            elif speed_priority > 4:
                return 'openai', 'gpt-4.1-mini'  # 4.1 Mini for rapid dev
            return 'claude', 'claude-sonnet-4-20250514'  # Sonnet for balanced dev
        
        # Tier 7: Speed-Optimized Tasks
        if speed_priority > 6 and complexity_score < 4:
            if prompt_length < 50:
                return 'claude', 'claude-3-5-haiku-20241022'  # Ultra-fast for micro tasks
            return 'openai', 'gpt-4.1-mini'  # 4.1 Mini for quick quality work
        
        # Tier 8: Productivity & Workflow
        if is_productivity_related:
            if is_startup_related or complexity_score > 4:
                return 'openai', 'gpt-4.1-mini'  # 4.1 Mini for productive balance
            return 'claude', 'claude-3-5-haiku-20241022'  # Haiku for simple productivity
        
        # Tier 9: Length-Based Fallbacks
        if is_long or complexity_score > 5:
            return 'claude', 'claude-sonnet-4-20250514'  # Sonnet for complex/long content
        elif prompt_length < 100 and complexity_score < 3:
            return 'claude', 'claude-3-5-haiku-20241022'  # Haiku for simple short queries
        
        # Tier 10: Intelligent Default
        # Choose based on subtle indicators
        if any(word in prompt_lower for word in ['help', 'how', 'what', 'explain']) and complexity_score < 4:
            return 'openai', 'gpt-4o-mini'  # Balanced for general help
        elif any(word in prompt_lower for word in ['create', 'build', 'make', 'design']):
            return 'openai', 'gpt-4.1-mini'  # 4.1 Mini for creative tasks
        else:
            return 'claude', 'claude-sonnet-4-20250514'  # Sonnet as premium default
        

def main():
    st.set_page_config(
        page_title="Smart Chat with Memory",
        page_icon="🧠",
        layout="wide"
    )
    
    # Initialize managers
    if 'memory_manager' not in st.session_state:
        st.session_state.memory_manager = MemoryManager()
    
    if 'ai_manager' not in st.session_state:
        st.session_state.ai_manager = AIManager()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.caption("Set up your AI chat environment")
        st.markdown("Configure your AI chat with memory management and model selection.")

        # Model selection
        st.subheader("🤖 AI Model Selection")
        
        # Provider selection
        provider = st.selectbox(
            "Choose AI Provider",
            ["Auto-select", "Claude", "OpenAI"]
        )
        
        # Model selection based on provider
        if provider == "Claude":
            claude_model_keys = list(Config.CLAUDE_MODELS.keys())
            claude_model_names = [f"{Config.CLAUDE_MODELS[key]['name']} ({Config.CLAUDE_MODELS[key]['cost']} cost)" 
                                for key in claude_model_keys]
            selected_claude = st.selectbox("Claude Model", claude_model_names)
            selected_model = claude_model_keys[claude_model_names.index(selected_claude)]
            
            # Show model info
            model_info = Config.CLAUDE_MODELS[selected_model]
            st.info(f"💡 **{model_info['name']}**\n\n**Cost:** {model_info['cost']}\n\n**Best for:** {model_info['best_for']}")
            
        elif provider == "OpenAI":
            openai_model_keys = list(Config.OPENAI_MODELS.keys())
            openai_model_names = [f"{Config.OPENAI_MODELS[key]['name']} ({Config.OPENAI_MODELS[key]['cost']} cost)" 
                                for key in openai_model_keys]
            selected_openai = st.selectbox("OpenAI Model", openai_model_names)
            selected_model = openai_model_keys[openai_model_names.index(selected_openai)]
            
            # Show model info
            model_info = Config.OPENAI_MODELS[selected_model]
            st.info(f"💡 **{model_info['name']}**\n\n**Cost:** {model_info['cost']}\n\n**Best for:** {model_info['best_for']}")
        else:
            selected_model = "auto"
            st.info("🔄 **Auto-select mode**\n\nThe app will intelligently choose the best model based on your query type:\n\n• **Search queries** → GPT-4o Mini Search\n• **Math/Logic** → o1 Mini\n• **Complex analysis** → Claude 4 Sonnet\n• **Quick questions** → Claude 3.5 Haiku")
        
        # Memory settings
        st.subheader("Memory Settings")
        use_memory = st.checkbox("Enable Memory Search", value=True)
        memory_limit = st.slider("Max memories to retrieve", 1, 10, 5)
        
        # Add memory manually
        st.subheader("Add Memory")
        new_memory = st.text_area("Add new memory:")
        memory_category = st.selectbox("Category", ["general", "facts", "preferences", "instructions"])
        
        if st.button("Add Memory") and new_memory:
            st.session_state.memory_manager.add_memory(new_memory, memory_category)
            st.success("Memory added!")
        
        # Clear chat
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    st.title("🧠 Smart Chat with Memory")
    st.caption("RAG-powered chat with Claude & GPT APIs")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if "model" in message:
                    st.caption(f"Model: {message['model']}")
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Display user message
        st.chat_message("user").write(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Search for relevant memories
        context = ""
        relevant_memories = []
        if use_memory:
            relevant_memories = st.session_state.memory_manager.search_memories(prompt, memory_limit)
            if relevant_memories:
                context_parts = []
                for memory in relevant_memories:
                    context_parts.append(f"[{memory['category']}] {memory['content']}")
                context = "Relevant context:\n" + "\n".join(context_parts)
        
        # Choose model and get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if provider == "Claude":
                    response = st.session_state.ai_manager.get_claude_response(prompt, context, selected_model)
                    model_used = Config.CLAUDE_MODELS[selected_model]['name']
                elif provider == "OpenAI":
                    response = st.session_state.ai_manager.get_gpt_response(prompt, context, selected_model)
                    model_used = Config.OPENAI_MODELS[selected_model]['name']
                else:  # Auto-select
                    auto_provider, auto_model = st.session_state.ai_manager.auto_select_model(prompt)
                    if auto_provider == "claude":
                        response = st.session_state.ai_manager.get_claude_response(prompt, context, auto_model)
                        model_used = f"{Config.CLAUDE_MODELS[auto_model]['name']} (auto)"
                    else:
                        response = st.session_state.ai_manager.get_gpt_response(prompt, context, auto_model)
                        model_used = f"{Config.OPENAI_MODELS[auto_model]['name']} (auto)"
                
                st.write(response)
                st.caption(f"🤖 Model: {model_used}")
                
                # Show relevant memories used
                if use_memory and relevant_memories:
                    with st.expander(f"📚 Used {len(relevant_memories)} memories"):
                        for i, memory in enumerate(relevant_memories, 1):
                            st.write(f"**{i}.** [{memory['category']}] {memory['content'][:100]}...")
                            st.caption(f"Similarity: {memory['similarity']:.2f}")
        
        # Save conversation and add to memory
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response, 
            "model": model_used
        })
        
        st.session_state.memory_manager.save_conversation(prompt, response, model_used)
        
        # Auto-add important information to memory
        if len(response) > 100 and not response.startswith("Error") and not response.startswith("API Error"):
            st.session_state.memory_manager.add_memory(
                f"Q: {prompt}\nA: {response[:300]}...", 
                "conversation"
            )
        
        st.rerun()

if __name__ == "__main__":
    main()