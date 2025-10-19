from qdrant_client import QdrantClient

def main():
    print("Hello from capstone!")
    client = QdrantClient(url="http://localhost:6333")

    print(client.get_collections())


if __name__ == "__main__":
    main()
