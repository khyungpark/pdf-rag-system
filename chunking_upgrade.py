"""
========================================
🚀 Semantic Chunking 업그레이드 모듈
========================================

포함 기능:
- Semantic Chunking (의미 단위 분할)
- Hierarchical Chunking (계층적 분할)
- Markdown Header Splitter
- 적응형 청킹

사용법:
    from chunking_upgrade import SemanticChunker, HierarchicalChunker
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# 1. Semantic Chunker (의미 기반 분할)
# ============================================================

class SemanticChunker:
    """
    의미 기반 청킹: 임베딩 유사도로 문장 경계 결정

    문장 간 의미 변화가 큰 지점에서 청크를 나눔
    """

    def __init__(
        self,
        embedding_model=None,
        breakpoint_threshold: float = 0.3,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000
    ):
        """
        Args:
            embedding_model: SentenceTransformer 모델 (None이면 기본 모델 사용)
            breakpoint_threshold: 청크 분할 임계값 (0~1, 높을수록 적은 분할)
            min_chunk_size: 최소 청크 크기 (글자수)
            max_chunk_size: 최대 청크 크기 (글자수)
        """
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # 임베딩 모델 로드
        if embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("   🤖 Semantic Chunker 모델 로드 중...")
                self.model = SentenceTransformer(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                print("   ✅ 모델 로드 완료")
            except ImportError:
                print("   ⚠️ sentence-transformers가 필요합니다.")
                self.model = None
        else:
            self.model = embedding_model

    def _split_sentences(self, text: str) -> List[str]:
        """문장 분리 (한국어/영어 지원)"""
        # 문장 종결 패턴
        pattern = r'(?<=[.!?。！？])\s+|(?<=[.!?。！？])(?=[가-힣A-Z])'
        sentences = re.split(pattern, text)

        # 빈 문장 제거 및 정리
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_similarities(self, sentences: List[str]) -> np.ndarray:
        """인접 문장 간 유사도 계산"""
        if self.model is None or len(sentences) < 2:
            return np.array([])

        # 문장 임베딩
        embeddings = self.model.encode(sentences)

        # 인접 문장 간 코사인 유사도
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]) + 1e-8
            )
            similarities.append(sim)

        return np.array(similarities)

    def _find_breakpoints(self, similarities: np.ndarray) -> List[int]:
        """유사도 기반 분할점 찾기"""
        if len(similarities) == 0:
            return []

        # 유사도 변화량 계산 (유사도가 급격히 떨어지는 지점)
        threshold = np.percentile(similarities, self.breakpoint_threshold * 100)

        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i + 1)  # 다음 문장부터 새 청크

        return breakpoints

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        텍스트를 의미 단위로 청킹

        Args:
            text: 분할할 텍스트
            metadata: 청크에 추가할 메타데이터

        Returns:
            청크 리스트 [{text, metadata, sentences}]
        """
        metadata = metadata or {}

        # 문장 분리
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        # 임베딩 모델이 없으면 고정 크기 폴백
        if self.model is None:
            return self._fallback_chunk(text, metadata)

        # 유사도 계산 및 분할점 찾기
        similarities = self._calculate_similarities(sentences)
        breakpoints = self._find_breakpoints(similarities)

        # 청크 생성
        chunks = []
        start_idx = 0

        for bp in breakpoints + [len(sentences)]:
            chunk_sentences = sentences[start_idx:bp]
            chunk_text = " ".join(chunk_sentences)

            # 최소/최대 크기 확인
            if len(chunk_text) < self.min_chunk_size and chunks:
                # 이전 청크에 합치기
                chunks[-1]["text"] += " " + chunk_text
                chunks[-1]["sentences"].extend(chunk_sentences)
            elif len(chunk_text) > self.max_chunk_size:
                # 강제 분할
                sub_chunks = self._force_split(chunk_text, chunk_sentences, metadata)
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    "text": chunk_text,
                    "sentences": chunk_sentences,
                    "metadata": metadata.copy()
                })

            start_idx = bp

        return chunks

    def _force_split(
        self,
        text: str,
        sentences: List[str],
        metadata: Dict
    ) -> List[Dict[str, Any]]:
        """최대 크기 초과 시 강제 분할"""
        chunks = []
        current_text = ""
        current_sentences = []

        for sentence in sentences:
            if len(current_text) + len(sentence) > self.max_chunk_size:
                if current_text:
                    chunks.append({
                        "text": current_text.strip(),
                        "sentences": current_sentences,
                        "metadata": metadata.copy()
                    })
                current_text = sentence
                current_sentences = [sentence]
            else:
                current_text += " " + sentence
                current_sentences.append(sentence)

        if current_text:
            chunks.append({
                "text": current_text.strip(),
                "sentences": current_sentences,
                "metadata": metadata.copy()
            })

        return chunks

    def _fallback_chunk(self, text: str, metadata: Dict) -> List[Dict[str, Any]]:
        """폴백: 고정 크기 청킹"""
        chunks = []
        sentences = self._split_sentences(text)

        current_text = ""
        current_sentences = []

        for sentence in sentences:
            if len(current_text) + len(sentence) > self.max_chunk_size:
                if current_text:
                    chunks.append({
                        "text": current_text.strip(),
                        "sentences": current_sentences,
                        "metadata": metadata.copy()
                    })
                current_text = sentence
                current_sentences = [sentence]
            else:
                current_text += " " + sentence
                current_sentences.append(sentence)

        if current_text:
            chunks.append({
                "text": current_text.strip(),
                "sentences": current_sentences,
                "metadata": metadata.copy()
            })

        return chunks


# ============================================================
# 2. Hierarchical Chunker (계층적 분할)
# ============================================================

class HierarchicalChunker:
    """
    계층적 청킹: 작은 청크로 검색, 큰 청크로 컨텍스트 제공

    Parent-Child 관계로 청크 관리
    """

    def __init__(
        self,
        child_chunk_size: int = 200,
        parent_chunk_size: int = 1000,
        overlap: int = 50
    ):
        """
        Args:
            child_chunk_size: 작은 청크 크기 (검색용)
            parent_chunk_size: 큰 청크 크기 (컨텍스트용)
            overlap: 청크 간 겹침
        """
        self.child_chunk_size = child_chunk_size
        self.parent_chunk_size = parent_chunk_size
        self.overlap = overlap

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> Tuple[List[Dict], List[Dict], Dict[int, int]]:
        """
        계층적 청킹 실행

        Args:
            text: 분할할 텍스트
            metadata: 메타데이터

        Returns:
            (child_chunks, parent_chunks, child_to_parent_map)
        """
        metadata = metadata or {}

        # Parent 청크 생성
        parent_chunks = self._create_chunks(
            text, self.parent_chunk_size, self.overlap, metadata, "parent"
        )

        # Child 청크 생성
        child_chunks = []
        child_to_parent = {}

        for parent_idx, parent in enumerate(parent_chunks):
            children = self._create_chunks(
                parent["text"],
                self.child_chunk_size,
                self.overlap // 2,
                {**metadata, "parent_idx": parent_idx},
                "child"
            )

            for child in children:
                child_to_parent[len(child_chunks)] = parent_idx
                child_chunks.append(child)

        return child_chunks, parent_chunks, child_to_parent

    def _create_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        metadata: Dict,
        level: str
    ) -> List[Dict]:
        """청크 생성 헬퍼"""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # 문장 경계 찾기
            if end < len(text):
                # 마침표, 물음표, 느낌표 뒤에서 자르기
                for i in range(end, max(start + chunk_size // 2, start), -1):
                    if text[i-1] in '.!?。！？':
                        end = i
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "level": level,
                    "start": start,
                    "end": end,
                    "metadata": metadata.copy()
                })

            start = max(end - overlap, start + 1)

        return chunks

    def get_parent_context(
        self,
        child_idx: int,
        child_to_parent: Dict[int, int],
        parent_chunks: List[Dict]
    ) -> str:
        """Child 청크의 Parent 컨텍스트 반환"""
        parent_idx = child_to_parent.get(child_idx)
        if parent_idx is not None and parent_idx < len(parent_chunks):
            return parent_chunks[parent_idx]["text"]
        return ""


# ============================================================
# 3. Markdown Header Splitter
# ============================================================

class MarkdownSplitter:
    """
    마크다운 헤더 기반 분할

    문서 구조를 유지하면서 청킹
    """

    def __init__(
        self,
        headers_to_split: List[str] = None,
        min_chunk_size: int = 100
    ):
        """
        Args:
            headers_to_split: 분할할 헤더 레벨 ["#", "##", "###"]
            min_chunk_size: 최소 청크 크기
        """
        self.headers = headers_to_split or ["#", "##", "###"]
        self.min_chunk_size = min_chunk_size

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        마크다운 텍스트를 헤더 기준으로 분할

        Args:
            text: 마크다운 텍스트
            metadata: 메타데이터

        Returns:
            청크 리스트 [{text, header_path, metadata}]
        """
        metadata = metadata or {}
        chunks = []

        # 헤더 패턴
        header_pattern = r'^(#{1,6})\s+(.+)$'

        lines = text.split('\n')
        current_chunk = []
        current_headers = {}  # level -> header text

        for line in lines:
            match = re.match(header_pattern, line)

            if match:
                level = len(match.group(1))
                header_text = match.group(2).strip()

                # 이전 청크 저장
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append({
                            "text": chunk_text,
                            "headers": current_headers.copy(),
                            "header_path": " > ".join(
                                current_headers.get(i, "")
                                for i in sorted(current_headers.keys())
                                if current_headers.get(i)
                            ),
                            "metadata": metadata.copy()
                        })
                    current_chunk = []

                # 헤더 업데이트
                current_headers[level] = header_text
                # 하위 레벨 헤더 제거
                for l in list(current_headers.keys()):
                    if l > level:
                        del current_headers[l]

            current_chunk.append(line)

        # 마지막 청크
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "headers": current_headers.copy(),
                    "header_path": " > ".join(
                        current_headers.get(i, "")
                        for i in sorted(current_headers.keys())
                        if current_headers.get(i)
                    ),
                    "metadata": metadata.copy()
                })

        return chunks


# ============================================================
# 4. 적응형 청커 (Adaptive Chunker)
# ============================================================

class AdaptiveChunker:
    """
    적응형 청킹: 컨텐츠 유형에 따라 자동으로 최적의 전략 선택
    """

    def __init__(self, embedding_model=None):
        self.semantic_chunker = SemanticChunker(embedding_model)
        self.hierarchical_chunker = HierarchicalChunker()
        self.markdown_splitter = MarkdownSplitter()

    def detect_content_type(self, text: str) -> str:
        """컨텐츠 유형 감지"""
        # 마크다운 헤더 체크
        if re.search(r'^#{1,6}\s+', text, re.MULTILINE):
            return "markdown"

        # 표 체크
        if re.search(r'\|.*\|.*\|', text):
            return "table"

        # 코드 블록 체크
        if re.search(r'```', text):
            return "code"

        # 일반 텍스트
        return "text"

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        force_strategy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        적응형 청킹 실행

        Args:
            text: 분할할 텍스트
            metadata: 메타데이터
            force_strategy: 강제 전략 ("semantic", "hierarchical", "markdown")

        Returns:
            청크 리스트
        """
        metadata = metadata or {}

        if force_strategy:
            strategy = force_strategy
        else:
            content_type = self.detect_content_type(text)

            if content_type == "markdown":
                strategy = "markdown"
            elif content_type == "table":
                strategy = "hierarchical"  # 표는 컨텍스트가 중요
            else:
                strategy = "semantic"

        # 전략별 청킹
        if strategy == "markdown":
            chunks = self.markdown_splitter.chunk(text, metadata)
        elif strategy == "hierarchical":
            child_chunks, _, _ = self.hierarchical_chunker.chunk(text, metadata)
            chunks = child_chunks
        else:
            chunks = self.semantic_chunker.chunk(text, metadata)

        # 청크 ID 추가
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = i
            chunk["strategy"] = strategy

        return chunks


# ============================================================
# 기존 content_to_chunks 업그레이드 버전
# ============================================================

def content_to_chunks_v2(
    pages_content: List[Dict],
    strategy: str = "semantic",
    embedding_model=None,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    페이지 컨텐츠를 향상된 방식으로 청킹

    Args:
        pages_content: PDF 처리 결과
        strategy: 청킹 전략 ("semantic", "adaptive", "hierarchical", "fixed")
        embedding_model: 임베딩 모델 (semantic 전략용)
        chunk_size: 고정 청크 크기 (fixed 전략용)
        overlap: 청크 오버랩 (fixed 전략용)

    Returns:
        청크 리스트
    """
    all_chunks = []

    # 청커 초기화
    if strategy == "semantic":
        chunker = SemanticChunker(
            embedding_model,
            max_chunk_size=chunk_size * 2
        )
    elif strategy == "adaptive":
        chunker = AdaptiveChunker(embedding_model)
    elif strategy == "hierarchical":
        chunker = HierarchicalChunker(
            child_chunk_size=chunk_size // 2,
            parent_chunk_size=chunk_size * 2
        )
    else:
        chunker = None  # 기존 방식

    for page_data in pages_content:
        page_num = page_data["page"]
        source = page_data.get("source", "unknown")

        # 모든 컨텐츠 합치기
        combined_parts = []

        if page_data.get("text"):
            combined_parts.append(page_data["text"])

        for table in page_data.get("tables", []):
            combined_parts.append(f"\n[표]\n{table['content']}\n")

        for img in page_data.get("images", []):
            combined_parts.append(f"\n[이미지 설명]\n{img['description']}\n")

        full_text = "\n\n".join(combined_parts)

        if not full_text.strip():
            continue

        metadata = {"page": page_num, "source": source}

        # 전략별 청킹
        if strategy == "hierarchical" and chunker:
            child_chunks, parent_chunks, mapping = chunker.chunk(full_text, metadata)
            for chunk in child_chunks:
                chunk["chunk_id"] = len(all_chunks)
                chunk["text"] = chunk.get("text", "")
                chunk["page"] = page_num
                chunk["source"] = source
                all_chunks.append(chunk)

        elif chunker:
            chunks = chunker.chunk(full_text, metadata)
            for chunk in chunks:
                chunk["chunk_id"] = len(all_chunks)
                chunk["text"] = chunk.get("text", "")
                chunk["page"] = page_num
                chunk["source"] = source
                all_chunks.append(chunk)

        else:
            # 기존 방식 (fixed)
            chunks = _fixed_chunk(full_text, chunk_size, overlap, page_num, source)
            for chunk in chunks:
                chunk["chunk_id"] = len(all_chunks)
                all_chunks.append(chunk)

    return all_chunks


def _fixed_chunk(
    text: str,
    chunk_size: int,
    overlap: int,
    page_num: int,
    source: str
) -> List[Dict]:
    """기존 고정 크기 청킹 (호환성 유지)"""
    import re
    chunks = []
    sentence_endings = re.compile(r'[.!?]\s+|[.!?]$|\n\n')

    if len(text) <= chunk_size:
        return [{
            "text": text.strip(),
            "page": page_num,
            "source": source
        }]

    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            match = sentence_endings.search(text, end)
            if match and match.end() - end < 100:
                end = match.end()

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "page": page_num,
                "source": source
            })

        start = max(end - overlap, start + 1)

    return chunks


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    print("🧪 Chunking 업그레이드 모듈 테스트\n")

    test_text = """
    임상시험의 개요

    임상시험은 새로운 의약품이나 치료법의 안전성과 효과를 평가하는 과학적 연구입니다.
    이 과정은 여러 단계로 나뉘어 진행되며, 각 단계마다 목적과 방법이 다릅니다.

    1상 임상시험

    1상 임상시험은 새로운 치료법을 처음으로 사람에게 시험하는 단계입니다.
    주로 건강한 자원자를 대상으로 하며, 안전성과 적절한 용량을 확인합니다.
    일반적으로 20-100명의 참여자가 포함됩니다.

    2상 임상시험

    2상 임상시험은 치료법의 효능을 처음으로 평가하는 단계입니다.
    실제 환자를 대상으로 하며, 부작용도 면밀히 관찰합니다.
    100-300명의 참여자가 포함되는 것이 일반적입니다.

    3상 임상시험

    3상 임상시험은 대규모로 진행되어 통계적 유의성을 확보합니다.
    수천 명의 환자가 참여하며, 기존 치료법과 비교 연구도 진행됩니다.
    이 단계를 통과해야 의약품 승인을 받을 수 있습니다.
    """

    # 1. Semantic Chunker 테스트
    print("1️⃣ Semantic Chunker 테스트")
    try:
        semantic = SemanticChunker(breakpoint_threshold=0.5)
        chunks = semantic.chunk(test_text)
        print(f"   생성된 청크 수: {len(chunks)}")
        for i, chunk in enumerate(chunks[:2]):
            print(f"   청크 {i+1}: {chunk['text'][:80]}...")
    except Exception as e:
        print(f"   ⚠️ 오류: {e}")

    # 2. Hierarchical Chunker 테스트
    print("\n2️⃣ Hierarchical Chunker 테스트")
    hier = HierarchicalChunker(child_chunk_size=200, parent_chunk_size=500)
    child_chunks, parent_chunks, mapping = hier.chunk(test_text)
    print(f"   Parent 청크: {len(parent_chunks)}개")
    print(f"   Child 청크: {len(child_chunks)}개")

    # 3. Markdown Splitter 테스트
    print("\n3️⃣ Markdown Splitter 테스트")
    md_text = """# 임상시험

## 1상 임상시험
안전성을 평가합니다.

## 2상 임상시험
효능을 평가합니다.

### 2a상
탐색적 연구입니다.

### 2b상
확증적 연구입니다.

## 3상 임상시험
대규모 연구입니다.
"""
    md_splitter = MarkdownSplitter()
    md_chunks = md_splitter.chunk(md_text)
    print(f"   생성된 청크 수: {len(md_chunks)}")
    for chunk in md_chunks[:3]:
        print(f"   [{chunk['header_path']}] {chunk['text'][:50]}...")

    # 4. Adaptive Chunker 테스트
    print("\n4️⃣ Adaptive Chunker 테스트")
    adaptive = AdaptiveChunker()
    print(f"   텍스트 유형: {adaptive.detect_content_type(test_text)}")
    print(f"   마크다운 유형: {adaptive.detect_content_type(md_text)}")

    print("\n✅ 테스트 완료!")
