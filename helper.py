from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, BatchEncoding
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import requests
import conf
import sqlite3
from rank_bm25 import BM25Okapi

class QdrantHelper:
    def __init__(self):
        self.client = QdrantClient(host=conf.QDRANT_HOST, port=conf.QDRANT_PORT)
        self.collection_name = conf.COLLECTION_NAME
        self.model = CLIPModel.from_pretrained(conf.MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(conf.MODEL_NAME)
        self.conn = sqlite3.connect("products.db") 
        
        if not self.client.collection_exists(self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"size": conf.VECTOR_SIZE, "distance": "Cosine"},
            )

    def find_similar_items(self, query_text=None, query_image=None, num_results: int = conf.NUM_RESULTS):
        try:
            if not (query_text or query_image):
                return []

            if query_text and query_image:
                # Process text input
                inputs_text: BatchEncoding = self.processor(text=[query_text], truncation=True, return_tensors="pt")
                with torch.no_grad():
                    text_embeddings = self.model.get_text_features(**inputs_text)

                # Process image input
                image = Image.open(query_image)
                inputs_image = self.processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    image_embeddings = self.model.get_image_features(**inputs_image)

                # Combine text and image embeddings in the unified vector space
                text_weight = 0.5  
                image_weight = 1 - text_weight
                combined_embedding = text_weight * text_embeddings + image_weight * image_embeddings
            
                query_embedding = combined_embedding.squeeze().tolist()

            elif query_text:
                inputs: BatchEncoding = self.processor(text=[query_text], truncation=True, return_tensors="pt")
                with torch.no_grad():
                    text_embeddings = self.model.get_text_features(**inputs)
                    query_embedding = text_embeddings.squeeze().tolist()

            elif query_image:
                image = Image.open(query_image)
                inputs = self.processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    image_embeddings = self.model.get_image_features(**inputs)
                    query_embedding = image_embeddings.squeeze().tolist()

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                append_payload=True,
                limit=num_results,
            )

            return search_result

        except ValueError as e:
            print(f"Error: {e}")
            return []
        
    def combined_search(self, query_text=None, query_image=None, num_results=conf.NUM_RESULTS):
        # Perform text-based search with BM25
        text_search_results = self.perform_text_search(query_text) if query_text else []
        # Perform vector similarity search using Qdrant
        vector_search_results = self.find_similar_items(query_text, query_image, num_results)

        # Combine and rank the results
        combined_results = []
        seen_ids = set()

        # Add text search results with higher priority
        for item in text_search_results:
            item_id = item["id"]
            if item_id not in seen_ids:
                seen_ids.add(item_id)       

                vector_score = next((result.score for result in vector_search_results if result.id == item_id), 0.0)

                combined_results.append((item_id, 0.5 + vector_score * 0.5))

        # Include vector search results not found in text search
        for result in vector_search_results:
            item_id = result.id
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                combined_results.append((item_id, result.score))

        # Sort the combined results by score and return the top N
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return [(item_id, score) for item_id, score in combined_results[:num_results]]

    def add_product_to_vector_store(self, product):
        name = product["name"]
        description = product["description"]
        image_captions = product["image_captions"]
        product_text = f"{name} {description} {image_captions}"
        product_text = product_text.replace("-", "").strip()

        # Embed the product text
        text_inputs: BatchEncoding = self.processor(text=[product_text], truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**text_inputs)

        # Extract image embedding
        image_url = product["image_url"]
        image_inputs = self.processor(images=[Image.open(requests.get(image_url, stream=True).raw)], return_tensors="pt")
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**image_inputs)

        # Combine text and image embeddings in the unified vector space
        text_weight = 0.5  
        image_weight = 1 - text_weight
        combined_embedding = text_weight * text_embeddings + image_weight * image_embeddings


        # Upsert the combined embedding into Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=product["id"],
                    vector=combined_embedding.squeeze().tolist(),
                    payload={"id":product["id"], "text": product_text},
                )
            ],
        )


    def perform_text_search(self, query_text):
        query_text = query_text.lower().replace("'", "''")

        # Fetch all products
        query = """
                SELECT id, name, description, image_captions
                FROM products;
                """

        c = self.conn.cursor()
        c.execute(query)
        results = c.fetchall()

        # Extract text from name, description, and image_captions
        corpus = [f"{row[1]} {row[2]} {row[3]}".lower() for row in results]

        # Tokenize the corpus
        tokenized_corpus = [doc.split() for doc in corpus]

        # Tokenize the query and Create BM25 index
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query_text.split()

        # Calculate BM25 scores
        doc_scores = bm25.get_scores(tokenized_query)

        # Combine scores with product details
        product_details = []
        for idx, row in enumerate(results):
            product_id = row[0]
            product_score = doc_scores[idx]
            if product_score !=0.0:
                product_details.append({
                    'id': product_id,
                    'relevance_score': product_score
                })

        # Sort product details by relevance score in descending order
        product_details.sort(key=lambda x: x['relevance_score'], reverse=True)

        return product_details[:conf.NUM_RESULTS]