"""
========================================
🚀 RAG 검색 업그레이드 모듈
========================================

포함 기능:
- Hybrid Search (BM25 + Dense)
- Cross-Encoder Reranking
- MMR (Maximal Marginal Relevance)
- 검색 품질 평가

사용법:
    from retrieval_upgrade import HybridRetriever, Reranker, RAGEvaluator
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# 1. Hybrid Search (BM25 + Dense)
# ============================================================

class BM25Retriever:
    """BM25 기반 키워드 검색"""

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        """
        Args:
            documents: 문서 리스트
            k1: 용어 빈도 포화 파라미터
            b: 문서 길이 정규화 파라미터
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avgdl = 0
        self.doc_freqs = []  # 각 문서의 단어 빈도
        self.idf = {}  # 역문서 빈도
        self.vocab = set()

        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """간단한 토큰화 (한국어/영어 모두 지원)"""
        import re
        # 한국어와 영어 모두 공백 기준 + 특수문자 제거
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _build_index(self):
        """BM25 인덱스 구축"""
        doc_count = len(self.documents)

        # 문서별 단어 빈도 계산
        for doc in self.documents:
            tokens = self._tokenize(doc)
            self.doc_len.append(len(tokens))

            # 단어 빈도 계산
            freq = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
                self.vocab.add(token)
            self.doc_freqs.append(freq)

        # 평균 문서 길이
        self.avgdl = sum(self.doc_len) / doc_count if doc_count > 0 else 0

        # IDF 계산
        df = {}  # 문서 빈도
        for freq in self.doc_freqs:
            for word in freq.keys():
                df[word] = df.get(word, 0) + 1

        for word, freq in df.items():
            # IDF with smoothing
            self.idf[word] = np.log((doc_count - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query: str) -> np.ndarray:
        """쿼리에 대한 모든 문서의 BM25 점수 계산"""
        query_tokens = self._tokenize(query)
        scores = np.zeros(len(self.documents))

        for i, doc_freq in enumerate(self.doc_freqs):
            doc_len = self.doc_len[i]

            for token in query_tokens:
                if token not in doc_freq:
                    continue

                freq = doc_freq[token]
                idf = self.idf.get(token, 0)

                # BM25 공식
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[i] += idf * (numerator / denominator)

        return scores

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """상위 k개 문서 반환"""
        scores = self.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]


class HybridRetriever:
    """
    Hybrid Search: BM25 (키워드) + Dense (의미) 검색 결합

    Reciprocal Rank Fusion (RRF) 사용
    """

    def __init__(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60
    ):
        """
        Args:
            documents: 문서 리스트
            embeddings: 문서 임베딩 (numpy array)
            dense_weight: Dense 검색 가중치
            sparse_weight: Sparse(BM25) 검색 가중치
            rrf_k: RRF 상수 (기본값 60)
        """
        self.documents = documents
        self.embeddings = embeddings
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

        # BM25 인덱스 구축
        print("   📊 BM25 인덱스 구축 중...")
        self.bm25 = BM25Retriever(documents)
        print("   ✅ Hybrid Retriever 초기화 완료")

    def _cosine_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """코사인 유사도 계산"""
        # 정규화
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)

        return np.dot(doc_norms, query_norm.flatten())

    def _reciprocal_rank_fusion(
        self,
        dense_ranking: List[int],
        sparse_ranking: List[int]
    ) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion으로 두 랭킹 결합

        RRF(d) = Σ 1 / (k + rank(d))
        """
        scores = {}

        # Dense 랭킹 점수
        for rank, doc_idx in enumerate(dense_ranking):
            if doc_idx not in scores:
                scores[doc_idx] = 0
            scores[doc_idx] += self.dense_weight / (self.rrf_k + rank + 1)

        # Sparse 랭킹 점수
        for rank, doc_idx in enumerate(sparse_ranking):
            if doc_idx not in scores:
                scores[doc_idx] = 0
            scores[doc_idx] += self.sparse_weight / (self.rrf_k + rank + 1)

        # 점수 기준 정렬
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        fetch_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Hybrid Search 실행

        Args:
            query: 검색 쿼리 텍스트
            query_embedding: 쿼리 임베딩
            top_k: 최종 반환할 문서 수
            fetch_k: 각 검색에서 가져올 후보 수

        Returns:
            검색 결과 리스트 [{doc_idx, document, score, method}]
        """
        # 1. Dense 검색 (의미 기반)
        dense_scores = self._cosine_similarity(query_embedding)
        dense_ranking = np.argsort(dense_scores)[::-1][:fetch_k].tolist()

        # 2. Sparse 검색 (키워드 기반)
        sparse_results = self.bm25.search(query, top_k=fetch_k)
        sparse_ranking = [idx for idx, _ in sparse_results]

        # 3. RRF로 결합
        fused_results = self._reciprocal_rank_fusion(dense_ranking, sparse_ranking)

        # 4. 결과 구성
        results = []
        for doc_idx, score in fused_results[:top_k]:
            results.append({
                "doc_idx": doc_idx,
                "document": self.documents[doc_idx],
                "score": score,
                "dense_score": float(dense_scores[doc_idx]),
                "in_dense": doc_idx in dense_ranking[:top_k],
                "in_sparse": doc_idx in sparse_ranking[:top_k]
            })

        return results


# ============================================================
# 2. Cross-Encoder Reranking
# ============================================================

class Reranker:
    """Cross-Encoder 기반 Reranker"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: Cross-Encoder 모델명
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """모델 로드"""
        try:
            from sentence_transformers import CrossEncoder
            print(f"   🤖 Reranker 모델 로드 중: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            print("   ✅ Reranker 초기화 완료")
        except ImportError:
            print("   ⚠️ sentence-transformers가 필요합니다.")
            print("      pip install sentence-transformers")
            self.model = None

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, str, float]]:
        """
        문서 재순위화

        Args:
            query: 검색 쿼리
            documents: 재순위화할 문서 리스트
            top_k: 반환할 상위 문서 수 (None이면 전체)

        Returns:
            [(원래_인덱스, 문서, 점수)] 리스트
        """
        if self.model is None:
            # 모델이 없으면 원래 순서 반환
            return [(i, doc, 1.0 - i * 0.1) for i, doc in enumerate(documents)]

        # Cross-Encoder 점수 계산
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)

        # 점수 기준 정렬
        scored_docs = list(zip(range(len(documents)), documents, scores))
        scored_docs.sort(key=lambda x: x[2], reverse=True)

        if top_k:
            scored_docs = scored_docs[:top_k]

        return scored_docs


# ============================================================
# 3. MMR (Maximal Marginal Relevance)
# ============================================================

def mmr_search(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    documents: List[str],
    top_k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5
) -> List[Dict[str, Any]]:
    """
    MMR (Maximal Marginal Relevance) 검색

    다양성과 관련성의 균형을 맞춤

    Args:
        query_embedding: 쿼리 임베딩
        doc_embeddings: 문서 임베딩들
        documents: 문서 리스트
        top_k: 최종 반환할 문서 수
        fetch_k: 초기 후보 문서 수
        lambda_mult: 0=최대 다양성, 1=최대 관련성

    Returns:
        선택된 문서 리스트
    """
    # 쿼리와 모든 문서 간 유사도
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)
    similarities = np.dot(doc_norms, query_norm.flatten())

    # 상위 fetch_k개 후보 선택
    candidate_indices = np.argsort(similarities)[::-1][:fetch_k]

    selected = []
    selected_embeddings = []

    while len(selected) < top_k and len(candidate_indices) > 0:
        # 각 후보에 대해 MMR 점수 계산
        mmr_scores = []

        for idx in candidate_indices:
            if idx in selected:
                continue

            # 관련성 (쿼리와의 유사도)
            relevance = similarities[idx]

            # 다양성 (이미 선택된 문서들과의 최대 유사도)
            if selected_embeddings:
                selected_emb = np.array(selected_embeddings)
                doc_emb = doc_norms[idx]
                max_sim = np.max(np.dot(selected_emb, doc_emb))
            else:
                max_sim = 0

            # MMR 점수
            mmr = lambda_mult * relevance - (1 - lambda_mult) * max_sim
            mmr_scores.append((idx, mmr))

        if not mmr_scores:
            break

        # 최고 MMR 점수 문서 선택
        best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
        selected.append(best_idx)
        selected_embeddings.append(doc_norms[best_idx])
        candidate_indices = [i for i in candidate_indices if i != best_idx]

    # 결과 구성
    results = []
    for idx in selected:
        results.append({
            "doc_idx": idx,
            "document": documents[idx],
            "similarity": float(similarities[idx])
        })

    return results


# ============================================================
# 4. 검색 품질 평가
# ============================================================

@dataclass
class EvaluationResult:
    """평가 결과"""
    precision_at_k: float
    recall_at_k: float
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain


class RAGEvaluator:
    """RAG 시스템 평가기"""

    def __init__(self):
        self.results = []

    def evaluate_retrieval(
        self,
        retrieved_docs: List[int],
        relevant_docs: List[int],
        k: int = 5
    ) -> Dict[str, float]:
        """
        검색 품질 평가

        Args:
            retrieved_docs: 검색된 문서 인덱스 리스트
            relevant_docs: 관련 문서 인덱스 리스트 (정답)
            k: 평가할 상위 k개

        Returns:
            평가 메트릭 딕셔너리
        """
        retrieved_set = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)

        # Precision@K
        hits = len(retrieved_set & relevant_set)
        precision = hits / k if k > 0 else 0

        # Recall@K
        recall = hits / len(relevant_set) if relevant_set else 0

        # MRR (Mean Reciprocal Rank)
        mrr = 0
        for i, doc_idx in enumerate(retrieved_docs[:k]):
            if doc_idx in relevant_set:
                mrr = 1 / (i + 1)
                break

        # NDCG@K
        dcg = 0
        for i, doc_idx in enumerate(retrieved_docs[:k]):
            if doc_idx in relevant_set:
                dcg += 1 / np.log2(i + 2)

        idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevant_set))))
        ndcg = dcg / idcg if idcg > 0 else 0

        return {
            "precision@k": precision,
            "recall@k": recall,
            "mrr": mrr,
            "ndcg@k": ndcg
        }

    def evaluate_groundedness(
        self,
        answer: str,
        context: str,
        llm_client=None
    ) -> Dict[str, Any]:
        """
        답변의 근거성 평가 (LLM 기반)

        Args:
            answer: 생성된 답변
            context: 참조한 컨텍스트
            llm_client: LLM 클라이언트 (선택)

        Returns:
            근거성 평가 결과
        """
        # 간단한 휴리스틱 평가 (LLM 없이)
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        # 답변 단어 중 컨텍스트에 있는 비율
        overlap = len(answer_words & context_words)
        coverage = overlap / len(answer_words) if answer_words else 0

        return {
            "word_overlap": overlap,
            "coverage_ratio": coverage,
            "is_grounded": coverage > 0.3  # 30% 이상이면 근거 있음으로 판단
        }

    def log_result(self, query: str, metrics: Dict[str, float]):
        """평가 결과 기록"""
        self.results.append({
            "query": query,
            **metrics
        })

    def get_summary(self) -> Dict[str, float]:
        """평가 결과 요약"""
        if not self.results:
            return {}

        summary = {}
        metrics = ["precision@k", "recall@k", "mrr", "ndcg@k"]

        for metric in metrics:
            values = [r.get(metric, 0) for r in self.results if metric in r]
            if values:
                summary[f"avg_{metric}"] = np.mean(values)

        return summary


# ============================================================
# 통합 검색 클래스
# ============================================================

class EnhancedRetriever:
    """
    향상된 검색기: Hybrid Search + Reranking + MMR
    """

    def __init__(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict]] = None,
        use_hybrid: bool = True,
        use_reranking: bool = True,
        use_mmr: bool = False,
        dense_weight: float = 0.7,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Args:
            documents: 문서 리스트
            embeddings: 문서 임베딩
            metadatas: 문서 메타데이터 (선택)
            use_hybrid: Hybrid Search 사용 여부
            use_reranking: Reranking 사용 여부
            use_mmr: MMR 사용 여부
            dense_weight: Dense 검색 가중치
            reranker_model: Reranker 모델명
        """
        self.documents = documents
        self.embeddings = embeddings
        self.metadatas = metadatas or [{}] * len(documents)
        self.use_hybrid = use_hybrid
        self.use_reranking = use_reranking
        self.use_mmr = use_mmr

        print("\n🔧 Enhanced Retriever 초기화")

        # Hybrid Retriever
        if use_hybrid:
            self.hybrid = HybridRetriever(
                documents, embeddings,
                dense_weight=dense_weight,
                sparse_weight=1-dense_weight
            )
        else:
            self.hybrid = None

        # Reranker
        if use_reranking:
            self.reranker = Reranker(reranker_model)
        else:
            self.reranker = None

        # Evaluator
        self.evaluator = RAGEvaluator()

        print("✅ Enhanced Retriever 준비 완료\n")

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        fetch_k: int = 20,
        mmr_lambda: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        향상된 검색 실행

        Args:
            query: 검색 쿼리
            query_embedding: 쿼리 임베딩
            top_k: 최종 반환할 문서 수
            fetch_k: 후보 문서 수
            mmr_lambda: MMR 다양성 파라미터

        Returns:
            검색 결과 리스트
        """
        # 1단계: 초기 검색
        if self.use_hybrid and self.hybrid:
            initial_results = self.hybrid.search(
                query, query_embedding,
                top_k=fetch_k, fetch_k=fetch_k * 2
            )
        else:
            # Dense 검색만
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            doc_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(doc_norms, query_norm.flatten())
            top_indices = np.argsort(similarities)[::-1][:fetch_k]

            initial_results = [
                {
                    "doc_idx": idx,
                    "document": self.documents[idx],
                    "score": float(similarities[idx])
                }
                for idx in top_indices
            ]

        # 2단계: MMR (다양성 확보)
        if self.use_mmr:
            candidate_indices = [r["doc_idx"] for r in initial_results]
            candidate_embeddings = self.embeddings[candidate_indices]
            candidate_docs = [r["document"] for r in initial_results]

            mmr_results = mmr_search(
                query_embedding,
                candidate_embeddings,
                candidate_docs,
                top_k=fetch_k,
                fetch_k=len(candidate_docs),
                lambda_mult=mmr_lambda
            )

            # 원래 인덱스로 매핑
            initial_results = [
                {
                    "doc_idx": candidate_indices[r["doc_idx"]],
                    "document": r["document"],
                    "score": r["similarity"]
                }
                for r in mmr_results
            ]

        # 3단계: Reranking
        if self.use_reranking and self.reranker and self.reranker.model:
            docs_to_rerank = [r["document"] for r in initial_results]
            reranked = self.reranker.rerank(query, docs_to_rerank, top_k=top_k)

            final_results = []
            for orig_idx, doc, score in reranked:
                result = initial_results[orig_idx].copy()
                result["rerank_score"] = float(score)
                result["final_score"] = float(score)
                final_results.append(result)
        else:
            final_results = initial_results[:top_k]
            for r in final_results:
                r["final_score"] = r.get("score", 0)

        # 메타데이터 추가
        for r in final_results:
            r["metadata"] = self.metadatas[r["doc_idx"]]

        return final_results


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    print("🧪 검색 업그레이드 모듈 테스트\n")

    # 테스트 데이터
    documents = [
        "임상시험은 새로운 의약품이나 치료법의 안전성과 효과를 확인하는 연구입니다.",
        "1상 임상시험은 소규모로 진행되며 안전성을 주로 평가합니다.",
        "2상 임상시험은 효능과 부작용을 평가하는 단계입니다.",
        "3상 임상시험은 대규모로 진행되어 통계적 유의성을 확보합니다.",
        "임상시험 참여자의 동의서 작성은 필수적인 절차입니다.",
        "위약 대조군은 임상시험의 효과를 비교하기 위해 사용됩니다.",
        "무작위 배정은 임상시험의 편향을 줄이는 중요한 방법입니다.",
        "이중맹검은 연구자와 참여자 모두 치료 내용을 모르게 하는 방법입니다."
    ]

    # 임베딩 생성 (테스트용 랜덤)
    np.random.seed(42)
    embeddings = np.random.randn(len(documents), 384)

    # BM25 테스트
    print("1️⃣ BM25 테스트")
    bm25 = BM25Retriever(documents)
    results = bm25.search("임상시험 안전성", top_k=3)
    for idx, score in results:
        print(f"   [{score:.3f}] {documents[idx][:50]}...")

    # Hybrid Retriever 테스트
    print("\n2️⃣ Hybrid Retriever 테스트")
    hybrid = HybridRetriever(documents, embeddings)
    query_emb = np.random.randn(384)
    results = hybrid.search("임상시험 안전성", query_emb, top_k=3)
    for r in results:
        print(f"   [{r['score']:.3f}] {r['document'][:50]}...")

    # Evaluator 테스트
    print("\n3️⃣ Evaluator 테스트")
    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate_retrieval(
        retrieved_docs=[0, 1, 2, 3, 4],
        relevant_docs=[0, 1, 5],
        k=5
    )
    print(f"   Precision@5: {metrics['precision@k']:.3f}")
    print(f"   Recall@5: {metrics['recall@k']:.3f}")
    print(f"   MRR: {metrics['mrr']:.3f}")
    print(f"   NDCG@5: {metrics['ndcg@k']:.3f}")

    print("\n✅ 테스트 완료!")
