from pathlib import Path
import os

class MDLoader:
    def __init__(self) -> None:
        pass

    def process(self, directory_path):        
        # 寻找文件夹中的.md文件
        documents = Path(directory_path).rglob('*.md')

        # 读取 documents 中的内容
        records = []
        for document in documents:
            with open(
                file=document,
                mode='r',
                encoding='utf-8'
            ) as f:
                content = f.read()
                # 同时保存文件位置方便检索
                records.append(
                    {
                        'source':  str(document),
                        'content': content
                    }
                )
        return records

if __name__ == '__main__':
    md_loader = MDLoader()
    path = ""
    records = md_loader.process(
        path
    )
    print(records[:2])