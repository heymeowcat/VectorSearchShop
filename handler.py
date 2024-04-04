from typing import List, Union
from PIL import Image
import conf
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition
from img2vec_pytorch import Img2Vec

class QdrantSearch:
    def __init__(self):
        self.client = QdrantClient(host=conf.QDRANT_HOST, port=conf.QDRANT_PORT)
        self.collection_name = conf.COLLECTION_NAME
        self.text_model = SentenceTransformer(conf.MODEL_NAME)
        self.img2vec = Img2Vec(model=conf.IMAGE_MODEL_NAME, layer_output_size=conf.VECTOR_SIZE_IMAGE)

        if not self.client.collection_exists(self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"size": conf.VECTOR_SIZE, "distance": "Cosine"},
            )

    def find_similar_text(self, query: str, num_results: int = conf.NUM_RESULTS) -> List[PointStruct]:
        query_vector = self.text_model.encode(query).tolist()
        additional_vector = [0.0] * 512
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector+additional_vector,
            append_payload=True,
            limit=num_results,
        )

    def find_similar_image(self, query: Image.Image, num_results: int = conf.NUM_RESULTS) -> List[PointStruct]:
            query_vector = self.img2vec.get_vec(query.convert('RGB')).tolist()
            additional_vector = [0.0] * 384
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=additional_vector+query_vector,
                append_payload=True,
                limit=num_results,
            )

            filtered_results = [r for r in results if "image_embedding" in r.payload]

            return filtered_results

    def embed_text(self, text: str) -> List[float]:
        return self.text_model.encode(text).tolist()

    def embed_image(self, image: Image.Image) -> List[float]:
        return self.img2vec.get_vec(image, tensor=False).tolist()

    def upsert_products(self, products):
        if not products:
            return

        points = []
        for idx, product in enumerate(products):
            name = product["name"]
            description = product["description"]
            image_captions = product["image_captions"]
            product_text = f"{name} {description} {image_captions}"
            product_text = product_text.replace("-", "").strip()
            text_embedding = self.embed_text(product_text)
            image = product["image"]
            image_embedding = self.embed_image(image)

            points.append(
                PointStruct(
                    id=idx,
                    vector=text_embedding+image_embedding,
                    payload={
                        "id":idx,
                        "image_embedding": image_embedding,
                        "text_embedding": text_embedding,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )