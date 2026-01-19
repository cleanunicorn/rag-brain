import hashlib
import os
from datetime import datetime

import click
import pathspec
from tqdm import tqdm

import chromadb


@click.group()
def main():
    """Brain manager."""
    pass

@main.command()
@click.argument("query")
@click.argument("collection_name")
@click.option("--results", default=3, help="Number of results", type=int)
def query(query, collection_name, results):
    client = chromadb.HttpClient(
        host="localhost",
        port=8000,
    )
    collection = client.get_or_create_collection(name=collection_name)

    results = collection.query(query_texts=[query], n_results=results)
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    file_name = ''
    for i, doc in enumerate(documents):
        if file_name != metadatas[i]['file_name']:
            file_name = metadatas[i]['file_name']
            print("="*len(file_name))
            print(f"{file_name}")
            print("="*len(file_name))
        else:
            print("...")
        print(doc)
    # print(results)


@main.command()
@click.argument("file_path")
@click.option(
    "--strategy",
    default="character",
    help="Chunking strategy",
    type=click.Choice(["character", "endline", "recursive", "markdown", "semantic"]),
)
@click.option("--chunk-size", default=100, help="Chunk size", type=int)
@click.option("--chunk-overlap", default=0, help="Chunk overlap", type=int)
@click.option("--strip-whitespace", is_flag=True, help="Strip whitespace")
def chunk(file_path, strategy, chunk_size, chunk_overlap, strip_whitespace):
    from chunking.chunking import Chunking

    chunking = Chunking()
    chunking.from_file(file_path)
    chunks = chunking.split(
        strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap, strip_whitespace=strip_whitespace
    )
    for chunk in chunks:
        print(chunk)
        print("-----")


def _find_gitignore_files(folder_path):
    """Find all .gitignore files in the directory tree."""
    gitignore_files = []
    for root, dirs, files in os.walk(folder_path):
        if '.gitignore' in files:
            gitignore_files.append(os.path.join(root, '.gitignore'))
    return gitignore_files


def _load_gitignore_patterns(gitignore_files):
    """Load and combine patterns from all .gitignore files."""
    all_patterns = []
    for gitignore_file in gitignore_files:
        try:
            with open(gitignore_file, 'r', encoding='utf-8') as f:
                patterns = f.read().splitlines()
                # Filter out empty lines and comments
                patterns = [p.strip() for p in patterns if p.strip() and not p.strip().startswith('#')]
                all_patterns.extend(patterns)
        except Exception as e:
            click.echo(f"Warning: Could not read {gitignore_file}: {e}")
    return all_patterns


def _should_ignore_file(file_path, folder_path, gitignore_patterns):
    """Check if a file should be ignored based on .gitignore patterns."""
    if not gitignore_patterns:
        return False
    
    # Get relative path from the folder root
    rel_path = os.path.relpath(file_path, folder_path)
    
    # Create pathspec object
    spec = pathspec.PathSpec.from_lines('gitwildmatch', gitignore_patterns)
    
    return spec.match_file(rel_path)


def _discover_files(folder_path, file_extensions, gitignore_patterns):
    """Discover files recursively, respecting .gitignore patterns."""
    discovered_files = []
    extensions = [ext.strip() for ext in file_extensions.split(',')]
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Check if file has the right extension
            if file_ext in extensions:
                # Check if file should be ignored
                if not _should_ignore_file(file_path, folder_path, gitignore_patterns):
                    discovered_files.append(file_path)
    
    return discovered_files


@main.command()
@click.argument("folder_path")
@click.argument("collection_name")
@click.option("--strategy", default="recursive", help="Chunking strategy",
              type=click.Choice(["character", "endline", "recursive", "markdown", "semantic"]))
@click.option("--chunk-size", default=1000, help="Chunk size", type=int)
@click.option("--chunk-overlap", default=200, help="Chunk overlap", type=int)
@click.option(
    "--file-extensions", default=".txt,.md,.py,.pdf", help="Comma-separated file extensions to process"
)
@click.option("--clean", is_flag=True, help="Clean/recreate the collection before adding documents")
@click.option(
    "--refresh",
    is_flag=True,
    help="Only process files that have changed (based on checksum), skip if already processed and unchanged",
)
def rag(folder_path, collection_name, strategy, chunk_size, chunk_overlap, file_extensions, clean, refresh):
    """Load documents from a folder into a ChromaDB collection for RAG.
    """
    from chunking.chunking import Chunking
    
    # Validate folder path
    if not os.path.exists(folder_path):
        click.echo(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    if not os.path.isdir(folder_path):
        click.echo(f"Error: '{folder_path}' is not a directory.")
        return
    
    click.echo(f"Processing folder: {folder_path}")
    click.echo(f"Collection: {collection_name}")
    click.echo(f"Strategy: {strategy}, Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
    
    # Find and load .gitignore patterns
    gitignore_files = _find_gitignore_files(folder_path)
    gitignore_patterns = _load_gitignore_patterns(gitignore_files)
    
    if gitignore_files:
        click.echo(f"Found {len(gitignore_files)} .gitignore file(s)")
    
    # Discover files
    click.echo("Discovering files...")
    files = _discover_files(folder_path, file_extensions, gitignore_patterns)
    
    if not files:
        click.echo("No files found matching the criteria.")
        return
    
    click.echo(f"Found {len(files)} files to process")
    
    # Connect to ChromaDB
    try:
        client = chromadb.HttpClient(
            host="localhost",
            port=8000,
        )
        
        # Handle collection creation/cleaning
        if clean:
            try:
                client.delete_collection(name=collection_name)
                click.echo(f"Deleted existing collection '{collection_name}'")
            except Exception:
                pass  # Collection might not exist
        
        collection = client.get_or_create_collection(name=collection_name)
        click.echo(f"Using collection '{collection_name}'")
        
    except Exception as e:
        click.echo(f"Error connecting to ChromaDB: {e}")
        click.echo("Make sure ChromaDB server is running on localhost:8000")
        return
    
    # Process files
    total_chunks = 0
    processed_files = 0
    skipped_files = []
    
    # If refresh is enabled, fetch existing metadata to compare checksums
    if refresh:
        click.echo("Fetching existing metadata for comparison...")
        existing_metadata = {}
        try:
            # Query collection for all documents
            all_results = collection.get()
            for doc_id, metadata in zip(all_results["ids"], all_results["metadatas"]):
                # Extract file path and checksum from metadata
                file_path = metadata.get("file_path")
                checksum = metadata.get("checksum")
                if file_path and checksum:
                    # Use relative path as key
                    rel_path = os.path.relpath(file_path, folder_path)
                    existing_metadata[rel_path] = checksum
        except Exception as e:
            click.echo(f"Warning: Could not fetch existing metadata: {e}")
            click.echo("Proceeding with full processing.")
    
    with tqdm(files, desc="Processing files") as pbar:
        for file_path in pbar:
            try:
                pbar.set_description(f"Processing {os.path.basename(file_path)}")
                
                # Load and chunk the file
                chunking = Chunking()
                chunking.from_file(file_path)
                
                if not chunking.text.strip():
                    skipped_files.append((file_path, "Empty file"))
                    continue
                
                # Compute checksum for the entire file
                file_hash = hashlib.sha256(chunking.text.encode('utf-8')).hexdigest()
                
                # Determine if file needs processing
                rel_path = os.path.relpath(file_path, folder_path)
                if refresh and rel_path in existing_metadata:
                    if existing_metadata[rel_path] == file_hash:
                        # click.echo(f"✓ Skipping unchanged file: {rel_path}")
                        continue  # Skip processing
                    else:
                        click.echo(f"🔄 File changed, reprocessing: {rel_path}")
                        pass
                
                chunks = chunking.split(
                    strategy=strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    strip_whitespace=True
                )
                
                if not chunks:
                    skipped_files.append((file_path, "No chunks generated"))
                    continue
                
                # Prepare data for ChromaDB
                documents = []
                metadatas = []
                ids = []
                
                file_stats = os.stat(file_path)
                
                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Only add non-empty chunks
                        documents.append(chunk)
                        
                        metadata = {
                            "file_path": file_path,
                            "file_name": os.path.basename(file_path),
                            "relative_path": rel_path,
                            "file_extension": os.path.splitext(file_path)[1].lower(),
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "file_size": file_stats.st_size,
                            "created_at": datetime.now().isoformat(),
                            "chunking_strategy": strategy,
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "checksum": file_hash
                        }
                        metadatas.append(metadata)
                        
                        # Create unique ID using file path and chunk index
                        chunk_id = f"{rel_path}::{i}"
                        ids.append(chunk_id)
                
                # Add to ChromaDB
                if documents:
                    collection.upsert(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    total_chunks += len(documents)
                    processed_files += 1
                
            except Exception as e:
                skipped_files.append((file_path, str(e)))
                continue
    
    # Print summary
    click.echo("\n=== Processing Summary ===")
    click.echo(f"Files processed: {processed_files}")
    click.echo(f"Total chunks added: {total_chunks}")
    click.echo(f"Files skipped: {len(skipped_files)}")
    
    if skipped_files:
        click.echo("\nSkipped files:")
        for file_path, reason in skipped_files[:10]:  # Show first 10
            click.echo(f"  {os.path.relpath(file_path, folder_path)}: {reason}")
        if len(skipped_files) > 10:
            click.echo(f"  ... and {len(skipped_files) - 10} more")
    
    click.echo(f"\nCollection '{collection_name}' is ready for RAG queries!")


if __name__ == "__main__":
    main()
