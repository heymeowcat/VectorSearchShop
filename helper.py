import os
from typing import List, Union
from PIL import Image
from dotenv import load_dotenv
import conf
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from img2vec_pytorch import Img2Vec
import google.generativeai as genai

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

    def find_similar_items_by_text(self, products, search_term, k=conf.NUM_RESULTS):
        # Generate text embedding for the search term
        query_text_embedding = genai.embed_content(
            model="models/embedding-001",
            content=search_term,
            task_type="retrieval_query",
        )["embedding"]

        padded_query_vector = query_text_embedding + [0.0] * 512 
        search_result = self.client.search(
            collection_name='products',
            query_vector=padded_query_vector,
        )
        
        filtered_texts = [(result.payload["text"], result.score) for result in search_result]

        filtered_products = []

        # Retrieve the actual product objects using the filtered texts and scores
        for text, score in filtered_texts:
            for product in products:
                name = product["name"]
                description = product["description"]
                image_captions = product["image_captions"]
                product_text = f"{name} {description} {image_captions}"
                product_text = product_text.replace("-", "").strip()

                if product_text == text:
                    filtered_products.append((product, score))
                    break

        return filtered_products[:k]


    def find_similar_items_by_image(self, query: Image.Image, num_results: int = conf.NUM_RESULTS) -> List[PointStruct]:
            # query_embedding =  [0.0] * 768 + model.encode(query_image).tolist()
            query_embedding = self.img2vec.get_vec(query.convert('RGB')).tolist()
            padded_vector = [0.0] * 768 + query_embedding
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=padded_vector,
                append_payload=True,
                limit=num_results,
            )

            filtered_results = [r for r in results if "image_embedding" in r.payload]

            return filtered_results


    def embed_image(self, image: Image.Image) -> List[float]:
        # embedding = self.image_model.encode(img_bytes).tolist()
        return self.img2vec.get_vec(image, tensor=False).tolist()

    def update_vector_store(self, products):
        if products:
            texts = []
            image_embeddings = []

            for product in products:
                name = product["name"]
                description = product["description"]
                image_captions = product["image_captions"]
                product_text = f"{name} {description} {image_captions}"
                product_text = product_text.replace("-", "").strip()
                texts.append(product_text)

                # Extract image embedding
                image = product["image"] 
                image_embedding = self.embed_image(image)
                image_embeddings.append(image_embedding)

            # Embed all product texts
            results = [
                genai.embed_content(
                    model="models/embedding-001",
                    content=sentence,
                    task_type="retrieval_document",
                    title="Qdrant x Gemini",
                )
                for sentence in texts
            ]

            # Upsert all product texts and image embeddings into Qdrant
            self.client.upsert(
                collection_name="products",
                points=[
                    PointStruct(
                        id=idx,
                        vector=list(result["embedding"]) + list(image_embedding),
                        payload={"text": text, "image_embedding": image_embedding},
                    )
                    for idx, (result, text, image_embedding) in enumerate(
                        zip(results, texts, image_embeddings)
                    )
                ],
            )

    