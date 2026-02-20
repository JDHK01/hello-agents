from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    
    driver.verify_connectivity()
    print("✅ 连接成功")
    
    result = driver.session().run("RETURN 1 AS n").single()["n"]
    print(f"✅ 查询成功: {result}")
    
    driver.close()

if __name__ == "__main__":
    main()
