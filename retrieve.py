from ingest import get_embedding_model, get_qdrant_client, COLLECTION_NAME

def search_documents(query: str, top_k: int = 3):
    embedding_model = get_embedding_model()
    qdrant = get_qdrant_client()
    
    # Embed the query
    query_vector = embedding_model.encode(query).tolist()
    
    # Search in Qdrant using the modern query_points API
    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    
    results = []
    for hit in search_results.points:
        results.append({
            "score": hit.score,
            "image_path": hit.payload.get("image_path"),
            "description": hit.payload.get("description")
        })
        
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        results = search_documents(query)
        for r in results:
            print(f"Score: {r['score']:.4f} | Image: {r['image_path']}")
            print(f"Description: {r['description']}\n")
