"""
Memory management system combining FAISS vector store with conversation buffer
"""
# import torch
# from typing import List
# from datetime import datetime

# from langchain.memory import ConversationBufferMemory
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document

# from config import Config


# class VoiceAgentMemory:
#     """
#     Memory system for voice agent
#     Combines:
#     - FAISS vector store for semantic search of past conversations
#     - Conversation buffer for recent context window
#     """
    
#     def __init__(self, embedding_model: str = Config.EMBEDDING_MODEL):
#         print(f"Initializing memory with embedding model: {embedding_model}")
#         self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
#         self.vector_store = None
#         self.conversation_buffer = ConversationBufferMemory(
#             memory_key="chat_history",
#             return_messages=True
#         )
#         self.documents = []
#         print("Memory initialized!")
        
#     def add_interaction(self, user_input: str, assistant_response: str):
#         """
#         Add interaction to both vector store and conversation buffer
        
#         Args:
#             user_input: User's input text
#             assistant_response: Assistant's response text
#         """
#         # Add to conversation buffer
#         self.conversation_buffer.save_context(
#             {"input": user_input},
#             {"output": assistant_response}
#         )
        
#         # Create document for vector store
#         doc = Document(
#             page_content=f"User: {user_input}\\nAssistant: {assistant_response}",
#             metadata={
#                 "user_input": user_input,
#                 "assistant_response": assistant_response,
#                 "timestamp": datetime.now().isoformat()
#             }
#         )
#         self.documents.append(doc)
        
#         # Update or create vector store
#         if self.vector_store is None:
#             self.vector_store = FAISS.from_documents([doc], self.embeddings)
#         else:
#             self.vector_store.add_documents([doc])
    
#     def search_similar(self, query: str, k: int = Config.VECTOR_STORE_K) -> List[Document]:
#         """
#         Search for similar past conversations
        
#         Args:
#             query: Search query
#             k: Number of results to return
            
#         Returns:
#             List of relevant documents
#         """
#         if self.vector_store is None:
#             return []
#         return self.vector_store.similarity_search(query, k=k)
    
#     def get_recent_history(self, k: int = Config.RECENT_HISTORY_K) -> str:
#         """
#         Get recent conversation history
        
#         Args:
#             k: Number of recent turns to retrieve
            
#         Returns:
#             Formatted string of recent conversations
#         """
#         memory_vars = self.conversation_buffer.load_memory_variables({})
#         messages = memory_vars.get("chat_history", [])
        
#         # Format recent messages
#         recent = messages[-k*2:] if messages else []
#         formatted = []
#         for msg in recent:
#             role = "User" if hasattr(msg, 'type') and msg.type == "human" else "Assistant"
#             formatted.append(f"{role}: {msg.content}")
        
#         return "\\n".join(formatted) if formatted else "No recent history"
    
#     def clear(self):
#         """Clear all memory"""
#         self.vector_store = None
#         self.conversation_buffer.clear()
#         self.documents = []
#         print("Memory cleared!")
    
#     def save_to_disk(self, path: str):
#         """Save vector store to disk"""
#         if self.vector_store:
#             self.vector_store.save_local(path)
#             print(f"Memory saved to {path}")
    
#     def load_from_disk(self, path: str):
#         """Load vector store from disk"""
#         try:
#             self.vector_store = FAISS.load_local(
#                 path, 
#                 self.embeddings,
#                 allow_dangerous_deserialization=True
#             )
#             print(f"Memory loaded from {path}")
#         except Exception as e:
#             print(f"Error loading memory: {e}")

import torch
from typing import List
from datetime import datetime

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from config import Config


class VoiceAgentMemory:
    """
    Memory system for voice agent
    Combines:
    - FAISS vector store for semantic search of past conversations
    - Conversation buffer for recent context window
    """
    
    def __init__(self, embedding_model: str = Config.EMBEDDING_MODEL):
        print(f"Initializing memory with embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = None
        self.conversation_buffer = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.documents = []
        print("Memory initialized!")
        
    def add_interaction(self, user_input: str, assistant_response: str):
        """
        Add interaction to both vector store and conversation buffer
        
        Args:
            user_input: User's input text
            assistant_response: Assistant's response text
        """
        # Add to conversation buffer
        self.conversation_buffer.save_context(
            {"input": user_input},
            {"output": assistant_response}
        )
        
        # Create document for vector store
        doc = Document(
            page_content=f"User: {user_input}\\nAssistant: {assistant_response}",
            metadata={
                "user_input": user_input,
                "assistant_response": assistant_response,
                "timestamp": datetime.now().isoformat()
            }
        )
        self.documents.append(doc)
        
        # Update or create vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents([doc], self.embeddings)
        else:
            self.vector_store.add_documents([doc])
    
    def search_similar(self, query: str, k: int = Config.VECTOR_STORE_K) -> List[Document]:
        """
        Search for similar past conversations
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """  
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)
    
    def get_recent_history(self, k: int = Config.RECENT_HISTORY_K) -> str:
        """
        Get recent conversation history
        
        Args:
            k: Number of recent turns to retrieve
            
        Returns:
            Formatted string of recent conversations
        """
        memory_vars = self.conversation_buffer.load_memory_variables({})
        messages = memory_vars.get("chat_history", [])
        
        # Format recent messages
        recent = messages[-k*2:] if messages else []
        formatted = []
        for msg in recent:
            role = "User" if hasattr(msg, 'type') and msg.type == "human" else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        
        return "\\n".join(formatted) if formatted else "No recent history"
    
    def clear(self):
        """Clear all memory"""
        self.vector_store = None
        self.conversation_buffer.clear()
        self.documents = []
        print("Memory cleared!")
    
    def save_to_disk(self, path: str):
        """Save vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(path)
            print(f"Memory saved to {path}")
    
    def load_from_disk(self, path: str):
        """Load vector store from disk"""
        try:
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Memory loaded from {path}")
        except Exception as e:
            print(f"Error loading memory: {e}")