from mcp.server.fastmcp import FastMCP

import chromadb

# Initialize FastMCP server
mcp = FastMCP("kb")

# Constants
HOST='localhost'
PORT=8000
COLLECTION='brain'

@mcp.tool()
def get_kb(query: str, count: int = 3) -> str:
    """
        Make query to knowledge base
    """
    
    client = chromadb.HttpClient(
        host=HOST,
        port=PORT,
    )
    collection = client.get_collection(COLLECTION)

    results = collection.query(query_texts=[query], n_results=count)
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    query_results = ''

    file_name = ''
    for i, doc in enumerate(documents):
        if file_name != metadatas[i]['file_name']:
            file_name = metadatas[i]['file_name']
            query_results = query_results + "="*len(file_name)
            query_results = query_results + f"{file_name}"
            query_results = query_results + "="*len(file_name)
        else:
            query_results = query_results + "..."
        query_results = query_results + doc

    return query_results
    
if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')