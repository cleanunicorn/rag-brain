import os

import PyPDF2
from langchain.text_splitter import (
    CharacterTextSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)


class Chunking:
    text = ""

    def from_file(self, file_path):
        """Load text from various file types."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            self.text = self._extract_pdf_text(file_path)
        else:
            # Handle text files (txt, md, py, etc.)
            with open(file_path, "r", encoding="utf-8") as f:
                self.text = f.read()

    def _extract_pdf_text(self, pdf_path):
        """Extract text from PDF file."""
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise ValueError(f"Error reading PDF file {pdf_path}: {str(e)}")
        return text

    def from_text(self, text):
        """Load text directly."""
        self.text = text

    def split(
        self,
        strategy="character",
        chunk_size=100,
        chunk_overlap=0,
        strip_whitespace=False,
    ):
        if strategy == "character":
            return self.split_charater(chunk_size=chunk_size)
        elif strategy == "endline":
            return self.split_endline(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strip_whitespace=strip_whitespace,
            )
        elif strategy == "recursive":
            return self.split_recursive(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strip_whitespace=strip_whitespace,
            )
        elif strategy == "markdown":
            return self.split_markdown(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strip_whitespace=strip_whitespace,
            )
        elif strategy == "semantic":
            return self.split_semantic(
                similarity_threshold=0.8,
                chunk_size=chunk_size,
            )
        else:
            raise ValueError("Unsupported strategy")

    def split_charater(self, chunk_size=100):
        return [
            self.text[i : i + chunk_size] for i in range(0, len(self.text), chunk_size)
        ]

    def split_endline(self, chunk_size=100, chunk_overlap=0, strip_whitespace=False):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strip_whitespace=strip_whitespace,
        )
        texts = text_splitter.create_documents([self.text])
        texts = [doc.page_content for doc in texts]

        return texts

    def split_recursive(self, chunk_size=100, chunk_overlap=0, strip_whitespace=False):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strip_whitespace=strip_whitespace,
        )
        texts = text_splitter.create_documents([self.text])
        texts = [doc.page_content for doc in texts]

        return texts

    def split_markdown(self, chunk_size=100, chunk_overlap=0, strip_whitespace=False):
        text_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strip_whitespace=strip_whitespace,
        )
        texts = text_splitter.create_documents([self.text])
        texts = [doc.page_content for doc in texts]

        return texts

    def split_semantic(self, similarity_threshold=0.8, chunk_size=100):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("./chunking/embeddinggemma-300m")

        # Split into sentences
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self.text)
        sentence_list = [chunk.text for chunk in doc.sents]
        sentences = [{'sentence': s.strip(), 'index': i} for i, s in enumerate(sentence_list)]

        # Add embeddings
        embeddings = model.encode([s['sentence'] for s in sentences])
        for i, sentence in enumerate(sentences):
            sentence['embedding'] = embeddings[i]

        # Calculate distances between sentences
        from sklearn.metrics.pairwise import cosine_similarity
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['embedding']
            embedding_next = sentences[i + 1]['embedding']

            similarity = cosine_similarity([embedding_current], [embedding_next])
            distance = 1 - similarity
            distances.append(distance)

            sentences[i]['distance_to_next'] = distance

        import numpy as np
        breakpoint_threshold_percentile = 95

        breakpoint_distance_threshold = np.percentile(distances, breakpoint_threshold_percentile)
        indices_above_threshold = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

        # Group sentences based on indices_above_threshold
        if not indices_above_threshold:
            # If no breakpoints found, return the whole text as one chunk
            return [self.text]
        
        # Create groups by splitting at the indices
        grouped_texts = []
        start_index = 0
        
        for end_index in indices_above_threshold:
            # Add the chunk from start_index to end_index (inclusive)
            chunk_text = " ".join([sentences[i]['sentence'] for i in range(start_index, end_index + 1)])
            grouped_texts.append(chunk_text)
            start_index = end_index + 1
        
        # Add the remaining sentences as the last chunk
        if start_index < len(sentences):
            chunk_text = " ".join([sentences[i]['sentence'] for i in range(start_index, len(sentences))])
            grouped_texts.append(chunk_text)

        #print(grouped_texts[:3])

        return grouped_texts
