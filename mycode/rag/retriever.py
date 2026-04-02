import pickle
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from lib.my_embedding import MyEmbedding

class Retriever:
    def __init__(self, emd_path, chunk_path):
        self.embeddings = np.load(emd_path)
        with open(
            chunk_path,
            mode='rb',
        ) as f:
            self.chunks = pickle.load(f)

    def retrieve(self, query, top_k=5, ):
        embeddinger = MyEmbedding()
        query_embedding = embeddinger.embedding(
            inputs=query,
        )

        scores = cosine_similarity(
            X= query_embedding,
            Y=self.embeddings
        )[0]

        topk_indices = np.argsort(scores)[::-1][:top_k]

        result = []

        for idx in topk_indices:
            result.append(
                {
                    'score': scores[idx],
                    'content': self.chunks[idx]
                }
            )
        return result
    
if __name__ == '__main__':
    my_retriever = Retriever(
        chunk_path='/Users/yqz/project/hello-agents/mycode/rag/chunks.pkl',
        emd_path='/Users/yqz/project/hello-agents/mycode/rag/embeddings.npy',
    )
    result = my_retriever.retrieve(query='怎么用 ollama', top_k=10)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )