from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter

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
        elif strategy == "langchain_text_splitter":
            return self.split_langchain_text_splitter(
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
        else:
            raise ValueError("Unsupported strategy")

    def split_charater(self, chunk_size=100):
        return [
            self.text[i : i + chunk_size] for i in range(0, len(self.text), chunk_size)
        ]

    def split_langchain_text_splitter(
        self, chunk_size=100, chunk_overlap=0, strip_whitespace=False
    ):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strip_whitespace=strip_whitespace,
        )
        texts = text_splitter.create_documents([self.text])
        texts = [doc.page_content for doc in texts]

        return texts

    def split_recursive(
        self, chunk_size=100, chunk_overlap=0, strip_whitespace=False
    ):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strip_whitespace=strip_whitespace,
        )
        texts = text_splitter.create_documents([self.text])
        texts = [doc.page_content for doc in texts]

        return texts
    
    def split_markdown(
        self, chunk_size=100, chunk_overlap=0, strip_whitespace=False
    ):
        text_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strip_whitespace=strip_whitespace,
        )
        texts = text_splitter.create_documents([self.text])
        texts = [doc.page_content for doc in texts]

        return texts