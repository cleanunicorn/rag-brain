from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)


class Chunking:
    text = ""

    def from_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()

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
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer("./embeddinggemma-300m")
        sentences = self.text.split(". ")
        embeddings = model.encode(sentences, convert_to_tensor=True)

        clusters = []
        used_indices = set()

        for i in range(len(sentences)):
            if i in used_indices:
                continue
            cluster = [sentences[i]]
            used_indices.add(i)
            for j in range(i + 1, len(sentences)):
                if j in used_indices:
                    continue
                similarity = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
                if similarity >= similarity_threshold:
                    cluster.append(sentences[j])
                    used_indices.add(j)
            clusters.append(" ".join(cluster))

        # Further split clusters into chunks of specified size
        final_chunks = []
        for cluster in clusters:
            for i in range(0, len(cluster), chunk_size):
                final_chunks.append(cluster[i : i + chunk_size])

        return final_chunks