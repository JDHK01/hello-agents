import re
import json
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from lib.my_embedding import MyEmbedding
from lib.my_llm import MyLLM
from lib.load_md import MDLoader

class DynamicSenmanticChunker:
    def __init__(self, embedding, tokenizer:AutoTokenizer):
        self.tokenizer = tokenizer
        self.embedding = embedding
    
    def split_text(self, file_name, text, num_cuts=10):
        print('='*20, end=''); print(f'输入', end=''); print('='*20, end=''); print();
        print(text); print()

        # 分句子
        sentences = self.split_into_sentences(text)
        print('='*20, end=''); print(f'分句子', end=''); print('='*20, end=''); print();
        print(json.dumps(sentences, ensure_ascii=False, indent=2)); print()

        # 找边界
        boundaries_indices = self.find_boundaries(sentences=sentences,num_cuts=num_cuts)
        print('='*20, end=''); print(f'边界索引', end=''); print('='*20, end=''); print();
        print(json.dumps([int(x) for x in boundaries_indices], ensure_ascii=False, indent=2)); print()

        # 结合边界和句子划分chunks
        chunks = self.split_chunk(
            sentences=sentences,
            boundary_indices=boundaries_indices
        )
        print('='*20, end=''); print(f'chunks', end=''); print('='*20, end=''); print();
        print(json.dumps(chunks, ensure_ascii=False, indent=2)); print()

        return [
            {
                'name': file_name, 
                'id': i,
                'len': self.token_len(content),
                'content': content
            }
            for i, content in enumerate(chunks)
        ]

    def token_len(self, text):
        return len(self.tokenizer.encode(text))
    
    def split_into_sentences(self, text: str):
        parts = re.findall(r'[^。！？；\n]+[。！？；]?', text)
        return [p.strip() for p in parts if p.strip()]
    
    def find_boundaries(self, sentences, num_cuts):
        embeddings = np.array(self.embedding.embedding(sentences))
        return self.identity_boundaryies(
            gamma_values=self.compute_semantic_discrepancy(embeddings),
            num_cuts=num_cuts
        )

    def compute_semantic_discrepancy(self, embeddings):
        # 相邻位置的余弦相似度
        similirity = np.diag(
            cosine_similarity(embeddings[:-1], embeddings[1:])
        )
        gamma = 1 - similirity
        print('='*20, end=''); print(f'语义差异', end=''); print('='*20, end=''); print();
        print(json.dumps([round(float(x), 4) for x in gamma], ensure_ascii=False, indent=2))
        return gamma
    
    def identity_boundaryies(self, gamma_values, num_cuts):
        # 排序一下 
        boundary_indices = np.argsort(
            gamma_values
        )[-num_cuts:]
        # 确定所有边界
        ### .-.---.--------.
        boundary_indices = [0] + sorted(boundary_indices+1) + [len(gamma_values)+1]
        return boundary_indices

    def split_chunk(self, sentences, boundary_indices):
        # 根据前面拆分的句子和确定的边界, 对 sentence 进行聚簇
        chunks = []
        for i in range(len(boundary_indices)-1):
            start_indice = boundary_indices[i]
            end_indice = boundary_indices[i+1]
            chunk = ''.join(sentences[start_indice:end_indice]).strip()
            chunks.append(chunk)
        return chunks
    
if __name__ == '__main__':
    records = MDLoader().process(
       '/Users/yqz/project/hello-agents/mycode/rag/md'
    )

    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen3-0.6B'
    )

    embeddings = MyEmbedding()

    my_chunker = DynamicSenmanticChunker(
        embedding=embeddings,
        tokenizer=tokenizer,
    )

    for record in records:
        result = my_chunker.split_text(
            file_name=record['source'],
            text = record['content'],
            num_cuts=5
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        