from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, BatchEncoding
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import requests
import conf

class QdrantHelper:
    def __init__(self):
        self.client = QdrantClient(host=conf.QDRANT_HOST, port=conf.QDRANT_PORT)
        self.collection_name = conf.COLLECTION_NAME
        self.model = CLIPModel.from_pretrained(conf.MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(conf.MODEL_NAME)

        if not self.client.collection_exists(self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"size": conf.VECTOR_SIZE, "distance": "Cosine"},
            )

    def find_similar_items(self, query_text=None, query_image=None, num_results: int = conf.NUM_RESULTS):
        try:
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

            else:
                raise ValueError("Either text or image must be provided")

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
                    payload={"text": product_text, "text_embedding": text_embeddings.squeeze().tolist(), "image_embedding": image_embeddings.squeeze().tolist()},
                )
            ],
        )