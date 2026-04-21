import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


class SplitService:
    def __init__(self, chunk_max_size: int, chunk_overlap: int, chunk_min_size: int):
        self.chunk_size = chunk_max_size
        self.chunk_overlap = chunk_overlap
        self.chunk_min_size = chunk_min_size

        # 1) 先按标题切分，尽量保留段落级语义边界。
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2")
            ]
        )

        # 2) 再按长度递归切分，确保单块文本不会超过模型可用上下文。
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
            # 当前块过小且前一块仍有容量时进行合并，降低检索碎片化带来的召回噪音。
            elif doc_size < self.chunk_min_size and len(current_doc.page_content) < self.chunk_size:
                current_doc.page_content += "\n\n" + doc.page_content
            else:
                merged_docs.append(current_doc)
                current_doc = doc

        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs

    def split_markdown(self, content: str, file_path: str) -> List[Document]:
        # 先按标题切，再按长度切，平衡语义完整性与 embedding 成本。
        md_docs = self.markdown_splitter.split_text(content)
        md_docs_after_header_split = self.text_splitter.split_documents(md_docs)
        final_docs = self._merge_small_chunk(md_docs_after_header_split)
        file_ext = Path(file_path).suffix.lower() or ".txt"

        for doc in final_docs:
            # 补充统一元数据，便于按来源过滤、删除和结果展示。
            doc.metadata["_source"] = file_path
            # extension 取真实文件后缀，避免 txt 被误标为 md。
            doc.metadata["_extension"] = file_ext
            doc.metadata["_file_name"] = Path(file_path).name

        logging.info(f"Markdown 分割完成: {file_path} -> {len(final_docs)} 个分片")
        return final_docs
