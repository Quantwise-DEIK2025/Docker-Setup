from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
import torch
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import ColbertReranker
import ollama
import os
import hashlib
import argparse
from tqdm import tqdm
import re, unicodedata
import subprocess


EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
MAX_TOKENS = 2000
OLLAMA_MODEL_NAME = "chunker_full_doc"

# Define model
class DbHandler:
    """
    Convenience class to handle database operations for LanceDB.
    """

    def __init__(self, db_path, embedding_model_name, drop_all_tables=False):
        self.db = lancedb.connect(db_path)
        self.reranker = ColbertReranker()

        self.embedding_model = get_registry().get("huggingface").create(name=embedding_model_name, trust_remote_code=True, device="cuda" if torch.cuda.is_available() else "cpu") 
        
        if drop_all_tables:
            print("Dropping all tables in the database for a fresh start...")
            self.db.drop_all_tables()

    def create_table(self, table_name, recreate_table=False):
        """
        Creates a table in the LanceDB database with the specified schema, optionally overwriting existing table.
        """
        model = DbHandler.create_model_class(embedding_model=self.embedding_model)
        if recreate_table:
            if table_name in self.db.table_names():
                print(f"Table {table_name} already exists. Overwriting...")
            self.db.create_table(table_name, schema=model, mode="overwrite")
        else:
            if table_name in self.db.table_names():
                print(f"Table {table_name} already exists. Preserving existing table...")
            self.db.create_table(table_name, schema=model, exist_ok=True)
    
    def add_to_table(self, table_name, chunks_with_metadata):
        """
        Add chunks to the selected table.
        Args:
            table_name (str): The name of the table where we want to upload the data.
            chunks_with_metadata (list): List of dictionaries of chunks to be added. Each dictionary's format must conform to the model defined in the `MyDocument` class

        """
        table = self.db.open_table(table_name)
        table.add(chunks_with_metadata)  # LanceDB doesn't check for duplicates by default

        # Reindexing is required after adding new data, to avoid duplicate indices and speed up search
        table.create_scalar_index("id", replace=True)  # Index based on the chunk's id, used to manually prevent duplicates
        table.create_fts_index("text", replace=True) # Used by the reranker as well as the hybrid search's BM25 index
        table.wait_for_index(["text_idx"])  # Creating fts index is async and can take some time

    def query_table(self, table_name, prompt, limit=3):
        """
        Queries a specified database table using a prompt and returns the top chunks as a pandas DataFrame.
        Args:
            table_name (str): The name of the table to query.
            prompt (str): The search prompt or query string.
            limit (int, optional): The maximum number of chunks to return. Defaults to 3.
        Returns:
            pandas.DataFrame: A DataFrame containing the top matching chunks from the table.
        Raises:
            Exception: If the table cannot be opened or the query fails.

        """
        table = self.db.open_table(table_name)
        results = table.search(prompt, query_type="hybrid", vector_column_name="vector", fts_columns="text") \
                        .rerank(reranker=self.reranker) \
                        .limit(limit) \
                        .to_pandas()
        return results
    
    def check_existing_documents(self, table_name):
        """
        Checks which documents have already been processed and are present in the specified table.
        Args:
            table_name (str): The name of the table to check.

        Returns:
            set: A set of unique document names already present in the table.
        """
        table = self.db.open_table(table_name)
        results = table.search() \
                        .to_pandas()
        
        unique_documents = set(results['document'])
        return unique_documents
    
    def create_model_class(embedding_model):
        """
        Factory function used for generating a schema when creating new tables.
        Args:
            embedding_model: Embedding model to be used in creation of vectors. Usually provided by calling `get_registry().get().create()` with some parameters
        Returns:
            out (class):
        """
        class MyDocument(LanceModel):
            text: str = embedding_model.SourceField()
            vector: Vector(embedding_model.ndims()) = embedding_model.VectorField()
            original_text: str
            context: str
            document: str
            pages: list[int]
            id: str
        return MyDocument


class SemanticChunker:
    """
    A class to handle chunking of documents into semantic chunks using a hybrid approach.
    It uses Docling for PDF processing, HuggingFace models for embedding and Ollama for context generation.
    """
    def __init__(self, db_path, embedding_model_name=EMBEDDING_MODEL_NAME, max_tokens=MAX_TOKENS, base_table_name="my_table", recreate_base_table=False):
        self.db_handler = DbHandler(db_path, embedding_model_name=embedding_model_name)
        self.db_handler.create_table(base_table_name, recreate_table=recreate_base_table)

        self.converter = DocumentConverter()
        # Tokenizer for chunking
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(embedding_model_name),
            max_tokens=max_tokens
        )
        self.chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True  # Optional, defaults to true
        )

    def clean_docling_chunk_strings(self, chunks):
        cleaned_chunks = []
        
        for chunk in chunks:
            # 1️⃣ Normalize Unicode and replace problematic punctuation
            chunk = unicodedata.normalize("NFKD", chunk).replace("\u00A0", " ")
            chunk = chunk.translate(str.maketrans({
                "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"'
            }))

            # 2️⃣ Remove URLs (massive tokenizers killers)
            chunk = re.sub(r"http\S+", "", chunk)

            # 3️⃣ Normalize whitespace but preserve paragraphs
            chunk = re.sub(r"[ \t]+", " ", chunk)
            chunk = re.sub(r"\n\s*\n", "\n\n", chunk)  # merge single newlines, keep double
            chunk = chunk.strip()

            cleaned_chunks.append(chunk)

        return cleaned_chunks

    def process_document(self, document_path, table_name, recreate_table=False):
        """
        Processes a document by converting, chunking, generating context, and storing results.
        
        This method performs the following steps:

            1. Converts the input document to a Docling Document using the configured converter.
            2. Chunks the document using the configured chunker.
            3. For each chunk, generates additional context using an Ollama model, providing both the full document and the chunk as input.
            4. Prepares each chunk with metadata, including the generated context, original text, document name, page numbers, and a unique identifier.
            5. Stores the processed chunks with metadata in the specified LanceDB table.
        Args:
            document_path (str): Path to the document file to be processed.
            table_name (str): Name of the LanceDB table where the processed chunks will be stored.
            recreate_table (bool, optional): If True, recreates the table before inserting data. Defaults to False.
        Returns:
            None
        """
        ###### ----- 1️⃣ Preliminary chunking and text transformations ---- ######
        # Convert the document to a Docling Document
        doc = self.converter.convert(document_path).document

        # Chunking the document
        chunks = list(self.chunker.chunk(dl_doc=doc))
        chunks_str = [chunk.text for chunk in chunks]
        chunks_str = self.clean_docling_chunk_strings(chunks_str)

        # Free up CUDA memory right after we got the results from Docling, so that Ollama can use the entire GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Prepare chunks with metadata
        chunks_with_metadata = []


        for chunk in tqdm(chunks, desc=f"Processing {document_path.split('/')[-1]}", position=1, leave=False):
            entire_doc = ""

            ###### ----- 2️⃣ Determine sliding window interval ---- ######
            chunk_index = chunks.index(chunk)

            context_length = 16_000 # Reduce window to save memory
            context_length = context_length - 2 * MAX_TOKENS # We need to reserve space for the chunk itself (twice, the context contains the chunk itself)
            total_context_chunk_number = context_length // (MAX_TOKENS*2) # 2x, cuz before and after the chunk

            start_index_original = chunk_index - total_context_chunk_number
            start_index_truncated = max(0, start_index_original) # Avoid index out of bounds

            end_index_original = chunk_index + total_context_chunk_number
            end_index_truncated = min(len(chunks)-1, end_index_original)

            if start_index_original < 0: # We are at the start of the document, so we need to add more chunks at the end
                end_index_truncated = min(len(chunks)-1, end_index_truncated + abs(start_index_original))
            if end_index_original > len(chunks)-1: # We are at the end of the document, so we need to add more chunks at the start
                start_index_truncated = max(0, start_index_truncated - abs(end_index_original - end_index_truncated))

            for i in range(start_index_truncated, end_index_truncated + 1):
                entire_doc += " " + chunks_str[i]

            ###### ----- 3️⃣ Generating context with Ollama ---- ######
            entire_doc = "FULL DOCUMENT:\n" + entire_doc
            ollama_prompt = f"CHUNK:\n{chunks_str[chunk_index]}"
            history = [{'role': 'user', 'content': entire_doc}, {'role': 'user', 'content': ollama_prompt}] # We want the history to only contain the current chunk and surrounding text to get context for the chunk

            response = ollama.chat(
                model=OLLAMA_MODEL_NAME,
                messages=history
            )
            context = response['message']['content']
            text_to_embed = chunks_str[chunk_index] + "\n\n" + context # We put the context AFTER the chunk to not mess up cosine similarity but still benefit keyword search for exact matches

            # Extracting page numbers from metadata
            pages = set( 
                prov.page_no
                for doc_item in chunk.meta.doc_items
                for prov in doc_item.prov
            )
            # Unique ID to avoid duplicates later on
            id = hashlib.sha256(chunks_str[chunk_index].encode()).hexdigest()

            chunks_with_metadata.append({'text': text_to_embed, 'original_text':chunks_str[chunk_index], 'context':context, 'document':document_path.split("/")[-1], 'pages':list(pages), 'id': id})

        ###### ----- 4️⃣ Uploading to LanceDB + clean up GPU memory ---- ######
        # Free up ollama from GPU memory so that Docling can semantically analyze the next doc even if it's like 100 pages
        subprocess.run(["ollama", "stop", OLLAMA_MODEL_NAME], check=True)
        self.db_handler.add_to_table(table_name=table_name, chunks_with_metadata=chunks_with_metadata)
        # Free up CUDA memory again, because the LanceDB embedding model is still in memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def process_directory(self, directory_path, table_name, recreate_table=False):
        """
        Convenience method, does the same thing as `process_document()`, except for every PDF in a directory.

        This method performs the following steps:
            0. Checks all PDF files in the directory to see if they have already been processed, and skips them if so.
            1. Converts the input document to a Docling Document using the configured converter.
            2. Chunks the document using the configured chunker.
            3. For each chunk, generates additional context using an Ollama model, providing both the full document and the chunk as input.
            4. Prepares each chunk with metadata, including the generated context, original text, document name, page numbers, and a unique identifier.
            5. Stores the processed chunks with metadata in the specified LanceDB table.
        Args:
            directory_path (str): The path to the directory where you want to process all PDFs.
            table_name (str): Name of the LanceDB table where the processed chunks will be stored.
            recreate_table (bool, optional): If True, recreates the table before inserting data. Defaults to False.
        Returns:
            None
        """
        pdf_names = set([f for f in os.listdir(directory_path) if f.endswith('.pdf')])
        existing_documents = self.db_handler.check_existing_documents(table_name=table_name)
        print(f"The following documents already exist in the database and will be skipped: {pdf_names & existing_documents}")
        pdf_names = pdf_names - existing_documents  # Process only new documents
        
        for study_name in tqdm(pdf_names, desc="All PDFs", position=0):
            try:
                self.process_document(document_path=f"{directory_path}/{study_name}", table_name=table_name, recreate_table=recreate_table)
            except Exception as e:
                print(f"ERROR! Something unexpected happened when processing document {study_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Chunker Entrypoint")
    parser.add_argument("--db_path", type=str, default="./db", help="Path to LanceDB database")
    parser.add_argument("--input", type=str, required=True, help="Path to a PDF file or directory containing PDFs")
    parser.add_argument("--table", type=str, default="my_table", help="LanceDB table name")
    parser.add_argument("--recreate_table", action="store_true", help="Recreate the table before inserting data")
    args = parser.parse_args()

    

    chunker = SemanticChunker(db_path=args.db_path, base_table_name=args.table, recreate_base_table=args.recreate_table)

    if os.path.isdir(args.input):
        chunker.process_directory(directory_path=args.input, table_name=args.table, recreate_table=args.recreate_table)
    else:
        chunker.process_document(document_path=args.input, table_name=args.table, recreate_table=args.recreate_table)