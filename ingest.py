import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"
QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "visual_docs"

# Global state
_processor = None
_vlm_model = None
_embedding_model = None
_qdrant_client = None
_markitdown = None

def get_vlm():
    global _processor, _vlm_model
    if _processor is None or _vlm_model is None:
        print(f"Loading Qwen2-VL model on {DEVICE}...")
        _vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID, torch_dtype="auto"
        ).to(DEVICE).eval()
        _processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    return _processor, _vlm_model

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading SentenceTransformer {EMBEDDING_MODEL_ID}...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID, device=DEVICE)
    return _embedding_model

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        print("Initializing Qdrant client...")
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
        try:
            _qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        except Exception:
            print(f"Creating collection {COLLECTION_NAME}...")
            _qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
    return _qdrant_client

def get_markitdown():
    global _markitdown
    if _markitdown is None:
        from markitdown import MarkItDown
        _markitdown = MarkItDown()
    return _markitdown

def process_image(image_path: str) -> str:
    processor, model = get_vlm()
    # Resize image to a maximum of 512x512 to prevent extreme CPU slowness
    from PIL import Image
    temp_path = image_path + ".small.jpg"
    with Image.open(image_path) as img:
        img.thumbnail((512, 512))
        img.convert("RGB").save(temp_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "file://" + os.path.abspath(temp_path)},
                {"type": "text", "text": "Describe this image in detail, including all text, labels, measurements, and structural features."},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(DEVICE)
    
    print("Generating description...")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    description = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return description

def process_document(file_path: str) -> str:
    md = get_markitdown()
    result = md.convert(file_path)
    return result.text_content

def ingest_document(file_path: str):
    print(f"Processing file: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        description = process_image(file_path)
    else:
        description = process_document(file_path)
        
    print(f"Extracted/Generated Text (first 200 chars): {description[:200]}...")
    
    print("Embedding text...")
    embedding_model = get_embedding_model()
    if isinstance(description, dict):
        description = str(description)
    vector = embedding_model.encode(description).tolist()
    
    print("Upserting into Qdrant...")
    qdrant = get_qdrant_client()
    point_id = str(uuid.uuid4())
    
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "image_path": file_path,
                    "description": description
                }
            )
        ]
    )
    print("Ingestion complete.")
    return {"id": point_id, "description": description}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ingest_document(sys.argv[1])
