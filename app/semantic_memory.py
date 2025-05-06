from app.milvus import MilvusCircuitStorage
from sentence_transformers import SentenceTransformer
import datetime

class SemanticMemory:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.vector_db = MilvusCircuitStorage()
        self.model = SentenceTransformer(model_name)
        self.conversation_history = []
        self.vector_dimension = 384
        
    def add_interaction(self, user_query, bot_response):
        interaction = {
            'question': user_query,
            'answer': bot_response,
            'timestamp': datetime.datetime.now()
        }
        self.conversation_history.append(interaction)
        embedding = self.model.encode([user_query[:250]])[0]
        self.vector_db.insert_vector([user_query[:250]], [embedding.reshape(1, -1)])
    
    def last_conversation(self):
        return self.conversation_history[-1] if self.conversation_history else None

        
    def retrieve_relevant_memory(self, query):
        query_vector = self.model.encode([query])[0]
        texts = self.vector_db.search_n_nearest(query_vector.reshape(1, -1), query)
        return texts

    def __del__(self):
        print(f"Memory is being cleared.")
        self.vector_db.close()

