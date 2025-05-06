import numpy as np
from pymilvus import connections, utility
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection

class MilvusCircuitStorage:
    NEAREST_NEIGHBORS = 3
    def __init__(self, collection_id=1):
        self.collection_id = collection_id
        connections.connect("default", host="xxx", port="xxx")
        self._create_collection()
    
    def show_all_vectors(self):
        results = self.collection.query(expr="id>=0",output_fields=["embedding", "id"])
        vectors = [item["embedding"] for item in results]
        ids = [item["id"] for item in results]
        # Convert vectors to a numpy array
        vectors = np.array(vectors)
        print("Vector shape:", vectors.shape)
        print(vectors)


    def _create_collection(self):
        DIMENSIONS = 384
        COLLECTION_NAME = "circuit_embeddings" + str(self.collection_id)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSIONS),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024)
        ]
        self.schema = CollectionSchema(fields, description="Vector collection")
        if utility.has_collection(COLLECTION_NAME):
           utility.drop_collection(COLLECTION_NAME)

        self.collection = Collection(name=COLLECTION_NAME, schema=self.schema)
        self._create_index()
        self._load_in_memory()

    def insert_vector(self, texts, vectors):
        # print(texts, vectors)
        self.collection.insert([vectors[0], texts])

    def _load_in_memory(self):
        self.collection.load()

    def _create_index(self):
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
    
    def search_n_nearest(self, query_vector, query_text):
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=query_vector,
            anns_field="embedding",
            param=search_params,
            limit=MilvusCircuitStorage.NEAREST_NEIGHBORS,
            output_fields=["text"]
        )

        outputs = []
        print(results)
        # print("****")
        for hit in results[0]:
            if (hit.distance > 0.2 )and hit.entity.get("text") != query_text:
                outputs.append(hit.entity.get("text"))
        return outputs


    def delete(self, key):
        self.collection.delete(f"id == {key}")
        self.collection.flush()
    
    def close(self):
        connections.disconnect("default")

#Gold is hovering around $2,400 an ounce - would you recommend adding it to a portfolio at this price?
#With silver at about $30 per ounce, how does it compare to gold as an inflation hedge?

