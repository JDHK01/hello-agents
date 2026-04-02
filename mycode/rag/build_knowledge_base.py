import json
import os
from transformers import AutoTokenizer
import pickle
import numpy as np
from lib.load_md import MDLoader
from lib.my_embedding import MyEmbedding
from lib.dynamic_senmantic_chunk import DynamicSenmanticChunker

def build_chunks(directory_path):
    records = MDLoader().process(
        directory_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen3-0.6B'
    )
    embeddings = MyEmbedding()
    my_chunker = DynamicSenmanticChunker(
        embedding=embeddings,
        tokenizer=tokenizer,
    )

    all_chunks = []
    for record in records:
        result = my_chunker.split_text(
            file_name=record['source'],
            text = record['content'],
            num_cuts=10
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        # 平行结构, 避免嵌入
        all_chunks.extend(result)
    print(json.dumps(all_chunks, ensure_ascii=False, indent=2))
    return all_chunks


def build_embedding(all_chunks):
    embeddings = MyEmbedding()
    contents = [record['content'] for record in all_chunks]
    emb_matrix = embeddings.embedding(
        contents
    )
    return np.array(emb_matrix)


if __name__ == '__main__':
    dir_path = '/Users/yqz/project/hello-agents/mycode/rag/md'
    all_chunks = build_chunks(
        dir_path,
    )
    print('-'*50 + '最终结果' + '-'*50)
    print(json.dumps(all_chunks, ensure_ascii=False, indent=2))

    
    emb_matrix = build_embedding(all_chunks)

    print(emb_matrix)

    # 这里保存一下, 两个是一一配对的. 方便后续对照查找
    save_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(save_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)

    with open(os.path.join(save_dir, "embeddings.npy"), "wb") as f:
        np.save(f, emb_matrix)