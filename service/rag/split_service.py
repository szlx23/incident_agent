import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class DocumentSplitService:
    def __init__(self, chunk_max_size: int, chunk_overlap: int, chunk_min_size: int):
        self.chunk_size = chunk_max_size
        self.chunk_overlap = chunk_overlap
        self.chunk_min_size = chunk_min_size

        # 1. markdown 标题分割器
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2")
            ]
        )

        # 2. 递归字符分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def _merge_small_chunk(self, documents: List[Document]) -> List[Document]:
        merged_docs = []
        current_doc = None

        for doc in documents:
            doc_size = len(doc.page_content)

            if current_doc is None:
                current_doc = doc
            elif doc_size < self.chunk_min_size and len(current_doc.page_content) < self.chunk_size:
                current_doc.page_content += "\n\n" + doc.page_content
            else:
                merged_docs.append(current_doc)
                current_doc = doc

        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs

    def split_markdown(self, content: str, file_path: str) -> List[Document]:
        md_docs = self.markdown_splitter.split_text(content)
        md_docs_after_header_split = self.text_splitter.split_documents(md_docs)
        final_docs = self._merge_small_chunk(md_docs_after_header_split)

        for doc in final_docs:
            doc.metadata["_source"] = file_path
            doc.metadata["_extension"] = ".md"
            doc.metadata["_file_name"] = Path(file_path).name

        logging.info(f"Markdown 分割完成: {file_path} -> {len(final_docs)} 个分片")
        return final_docs
