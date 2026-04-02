from lib.my_llm import MyLLM
from retriever import Retriever
import os
import pickle
import numpy as np
from build_knowledge_base import build_chunks, build_embedding

class RagAgent:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = MyLLM()
    
    def run(self, query, top_k=10):
        prompt = self.build_prompt(
            query, top_k=top_k
        )
        response = self.llm.generate(
            prompt
        )
        return response
    
    def build_prompt(self, query, top_k):
        result = self.retriever.retrieve(
            query=query,
            top_k=top_k
        )
        context = []
        for i, item in enumerate(result):
            chunk = item['content']
            context.append(
                {
                    'source': chunk['name'],
                    'content': chunk['content']
                }
            )
        context_str = '\n'.join(
            f"[{c['source']}] {c['content']}" for c in context
        )
        return [
            {
                'role': 'system',
                'content': '你是一个信息总结助手'
            },
            {
                "role": "user",
                "content": (
                    f"问题：{query}\n\n"
                    f"参考资料如下：\n{context_str}\n"
                    f"请基于参考资料回答问题，并尽量指出依据来自哪一条参考片段。"
                )
            }
        ]
    
if __name__ == '__main__':
    # 编译信息
    dir_path = '/Users/yqz/project/hello-agents/mycode/rag/md'
    all_chunks = build_chunks(
        dir_path,
    )
    emb_matrix = build_embedding(all_chunks)
    save_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(save_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)
    with open(os.path.join(save_dir, "embeddings.npy"), "wb") as f:
        np.save(f, emb_matrix)
    
    # 做检索器
    retriever = Retriever(
        chunk_path='/Users/yqz/project/hello-agents/mycode/rag/chunks.pkl',
        emd_path='/Users/yqz/project/hello-agents/mycode/rag/embeddings.npy',
    )

    # agent启动
    rag_agent = RagAgent(
        retriever=retriever
    )
    rag_agent.run(
        query='ollama 的常见指令'
    )