import click

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    # database='brain'
)

collection = client.get_or_create_collection(name="prompts")


@click.group()
def main():
    """Brain manager."""
    pass


@main.command()
def add_prompts():
    import pandas as pd

    df = pd.read_csv("./prompts.csv")
    texts = df["prompt"].tolist()
    print(texts[:1])
    # metadatas = df['act'].tolist()
    # print(metadatas[:1])
    ids = [str(i) for i in range(len(texts))]
    print(ids[:1])

    collection.add(
        documents=texts,
        # metadatas=metadatas,
        ids=ids,
    )


@main.command()
@click.argument("query")
def query_prompts(query):
    results = collection.query(query_texts=[query], n_results=3)
    for row in results["documents"]:
        print(row)
    # print(results)


@main.command()
@click.argument("file_path")
@click.option(
    "--strategy",
    default="character",
    help="Chunking strategy",
    type=click.Choice(["character", "langchain_text_splitter", 'recursive', 'markdown']),
)
def chunk(file_path, strategy):
    from chunking.chunking import Chunking

    chunking = Chunking()
    chunking.from_file(file_path)
    chunks = chunking.split(strategy=strategy, chunk_size=300, chunk_overlap=100, strip_whitespace=True)
    for chunk in chunks:
        print(chunk)
        print("-----")


if __name__ == "__main__":
    main()
