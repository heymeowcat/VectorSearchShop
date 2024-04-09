import os
from typing import List
from PIL import Image
from dotenv import load_dotenv
import conf
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from img2vec_pytorch import Img2Vec
import google.generativeai as genai
import requests
from io import BytesIO

load_dotenv()
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model_vision = genai.GenerativeModel('gemini-pro-vision')
except Exception as e:
    print("Error configuring Gemini API: {e}")


class QdrantHelper:
    def __init__(self):
        self.client = QdrantClient(host=conf.QDRANT_HOST, port=conf.QDRANT_PORT)
        self.collection_name = conf.COLLECTION_NAME
        # self.image_model =SentenceTransformer("clip-ViT-L-14")
        self.img2vec = Img2Vec(model=conf.IMAGE_MODEL_NAME, layer_output_size=conf.VECTOR_SIZE_IMAGE)

        if not self.client.collection_exists(self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"size": conf.VECTOR_SIZE, "distance": "Cosine"},
            )

    def find_similar_items_by_text(self, search_term, num_results: int = conf.NUM_RESULTS):
        try:
            query_text_embedding = genai.embed_content(
                model="models/embedding-001",
                content=search_term,
                task_type="retrieval_query",
            )["embedding"]

            padded_query_vector = [0] * 512 + query_text_embedding
            search_result = self.client.search(
                collection_name='products',
                query_vector=padded_query_vector,
                append_payload=True,
                limit=num_results,
            )
            
            filtered_results = [r for r in search_result if "text_embedding" in r.payload]
            return filtered_results
        except (requests.exceptions.RequestException):
            return []

    def find_similar_items_by_image(self, query_image_url: str, num_results: int = conf.NUM_RESULTS) -> List[PointStruct]:
        try:
            response = requests.get(query_image_url)
            response.raise_for_status()
            query_image = Image.open(BytesIO(response.content))
            query_embedding = self.img2vec.get_vec(query_image.convert('RGB')).tolist()
            padded_vector = [0.0] * 768 + query_embedding
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=padded_vector,
                append_payload=True,
                limit=num_results,
            )

            filtered_results = [r for r in results if "image_embedding" in r.payload]

            return filtered_results
        except (requests.exceptions.RequestException):
            return []
        
    def embed_image(self, image_url: str) -> List[float]:
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return self.img2vec.get_vec(image, tensor=False).tolist()
        except (requests.exceptions.RequestException):
            return []

    def add_product_to_vector_store(self, product):
        name = product["name"]
        description = product["description"]
        image_captions = product["image_captions"]
        product_text = f"{name} {description} {image_captions}"
        product_text = product_text.replace("-", "").strip()
        

        # Extract image embedding
        image_url = product["image_url"]
        image_embedding = self.embed_image(image_url)

        # Embed the product text
        result = genai.embed_content(
            model="models/embedding-001",
            content=product_text,
            task_type="retrieval_document",
            title="Qdrant x Gemini",
        )

        # Upsert the product text and image embedding into Qdrant
        self.client.upsert(
            collection_name="products",
            points=[
                PointStruct(
                    id=product["id"],
                    vector=result["embedding"] + image_embedding,
                    payload={"text": product_text, "text_embedding":result["embedding"], "image_embedding": image_embedding},
                )
            ],
        )