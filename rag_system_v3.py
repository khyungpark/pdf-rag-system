"""
========================================
🚀 RAG 시스템 v3 - 업그레이드 버전
========================================

업그레이드 내용:
1. Hybrid Search (BM25 + Dense)
2. Cross-Encoder Reranking
3. Semantic Chunking
4. 검색 품질 평가

사용법:
    python rag_system_v3.py              # 기존 DB로 검색
    python rag_system_v3.py --rebuild    # DB 재구축 (Semantic Chunking 적용)
    python rag_system_v3.py --evaluate   # 검색 품질 평가 모드
"""

import os
import sys
import argparse
import tarfile
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

# ============================================================
# 필요한 라이브러리
# ============================================================
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    from pdf_processor import EnhancedPDFProcessor
    from llm_client import get_llm_client
    from retrieval_upgrade import EnhancedRetriever, RAGEvaluator
    from chunking_upgrade import content_to_chunks_v2, SemanticChunker
except ImportError as e:
    print("❌ 필요한 패키지가 설치되지 않았습니다!")
    print("   아래 명령어를 먼저 실행해주세요:")
    print()
    print("   pip install -r requirements.txt")
    print()
    print(f"   (에러 상세: {e})")
    sys.exit(1)


# ============================================================
# 설정값
# ============================================================

# PDF 파일 경로
PDF_PATH = "./pdfs"

# 청킹 설정
CHUNK_STRATEGY = "semantic"  # "semantic", "adaptive", "hierarchical", "fixed"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# 임베딩 모델
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 데이터베이스 경로
DB_PATH = "./chroma_db"

# 검색 설정
TOP_K = 5
FETCH_K = 20  # 초기 후보 수

# 업그레이드 기능 설정
USE_HYBRID = True       # Hybrid Search 사용
USE_RERANKING = True    # Reranking 사용
USE_MMR = False         # MMR 사용 (다양성 vs 관련성)
DENSE_WEIGHT = 0.7      # Dense 검색 가중치 (나머지는 BM25)

# Reranker 모델
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 백업 설정
BACKUP_DIR = "./backups"
MAX_BACKUPS = 5


# ============================================================
# 백업 유틸리티
# ============================================================

def backup_database(db_path: str = DB_PATH, backup_dir: str = BACKUP_DIR) -> str | None:
    """
    DB 변경 전 자동 백업 생성

    Returns:
        str: 백업 파일 경로 (성공 시)
        None: 백업 실패 또는 DB가 없는 경우
    """
    if not os.path.exists(db_path):
        print("   ℹ️ 기존 DB가 없어 백업 생략")
        return None

    os.makedirs(backup_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"chroma_db_backup_{timestamp}.tar.gz"
    backup_path = os.path.join(backup_dir, backup_filename)

    try:
        print(f"   💾 DB 백업 생성 중...")
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(db_path, arcname=os.path.basename(db_path))

        backup_size = os.path.getsize(backup_path) / (1024 * 1024)
        print(f"   ✅ 백업 완료: {backup_filename} ({backup_size:.1f}MB)")

        cleanup_old_backups(backup_dir)
        return backup_path
    except Exception as e:
        print(f"   ⚠️ 백업 실패: {e}")
        return None


def cleanup_old_backups(backup_dir: str = BACKUP_DIR, max_backups: int = MAX_BACKUPS):
    """오래된 백업 파일 정리 (최신 N개만 유지)"""
    if not os.path.exists(backup_dir):
        return

    backups = sorted([
        f for f in os.listdir(backup_dir)
        if f.startswith("chroma_db_backup_") and f.endswith(".tar.gz")
    ], reverse=True)

    for old_backup in backups[max_backups:]:
        old_path = os.path.join(backup_dir, old_backup)
        try:
            os.remove(old_path)
            print(f"   🗑️ 오래된 백업 삭제: {old_backup}")
        except Exception:
            pass


# ============================================================
# RAG 시스템 클래스
# ============================================================

class RAGSystemV3:
    """업그레이드된 RAG 시스템"""

    def __init__(
        self,
        db_path: str = DB_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        use_hybrid: bool = USE_HYBRID,
        use_reranking: bool = USE_RERANKING,
        use_mmr: bool = USE_MMR
    ):
        self.db_path = db_path
        self.use_hybrid = use_hybrid
        self.use_reranking = use_reranking
        self.use_mmr = use_mmr

        print("\n" + "="*60)
        print("🚀 RAG 시스템 v3 초기화")
        print("="*60)

        # 임베딩 모델 로드
        print(f"\n📦 임베딩 모델 로드: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("   ✅ 모델 로드 완료")

        # ChromaDB 연결
        print(f"\n💾 데이터베이스 연결: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)

        try:
            self.collection = self.client.get_collection("rag_documents")
            count = self.collection.count()
            print(f"   ✅ 컬렉션 로드 완료 ({count:,}개 청크)")
        except Exception:
            print("   ⚠️ 컬렉션이 없습니다. --rebuild 옵션으로 생성하세요.")
            self.collection = None

        # Enhanced Retriever 초기화
        self.retriever = None
        if self.collection:
            self._init_enhanced_retriever()

        # LLM 클라이언트
        self.llm = get_llm_client()
        if self.llm.is_available():
            print("\n🤖 LLM 클라이언트 연결됨")
        else:
            print("\n💡 LLM 클라이언트 없음 (ANTHROPIC_API_KEY 설정 필요)")

        # 평가기
        self.evaluator = RAGEvaluator()

        print("\n" + "="*60)
        print("✅ 초기화 완료!")
        print("="*60)
        self._print_config()

    def _print_config(self):
        """설정 출력"""
        print(f"\n📊 현재 설정:")
        print(f"   • Hybrid Search: {'✅' if self.use_hybrid else '❌'}")
        print(f"   • Reranking: {'✅' if self.use_reranking else '❌'}")
        print(f"   • MMR: {'✅' if self.use_mmr else '❌'}")
        if self.use_hybrid:
            print(f"   • Dense 가중치: {DENSE_WEIGHT}")
            print(f"   • BM25 가중치: {1-DENSE_WEIGHT}")

    def _init_enhanced_retriever(self):
        """Enhanced Retriever 초기화"""
        print("\n🔧 Enhanced Retriever 초기화 중...")

        # 모든 문서와 임베딩 가져오기
        all_data = self.collection.get(include=["documents", "embeddings", "metadatas"])

        documents = all_data["documents"]
        metadatas = all_data["metadatas"]

        # 임베딩이 없으면 생성
        if all_data["embeddings"]:
            embeddings = np.array(all_data["embeddings"])
        else:
            print("   임베딩 재생성 중...")
            embeddings = self.embedding_model.encode(documents, show_progress_bar=True)

        # Enhanced Retriever 생성
        self.retriever = EnhancedRetriever(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            use_hybrid=self.use_hybrid,
            use_reranking=self.use_reranking,
            use_mmr=self.use_mmr,
            dense_weight=DENSE_WEIGHT,
            reranker_model=RERANKER_MODEL
        )

    def rebuild_database(
        self,
        pdf_path: str = PDF_PATH,
        chunk_strategy: str = CHUNK_STRATEGY,
        extract_images: bool = True,
        extract_tables: bool = True
    ):
        """데이터베이스 재구축 (업그레이드된 청킹 적용)"""
        print("\n" + "="*60)
        print("🔄 데이터베이스 재구축")
        print("="*60)
        print(f"   • PDF 경로: {pdf_path}")
        print(f"   • 청킹 전략: {chunk_strategy}")

        # 기존 DB 백업
        print("\n📦 기존 DB 백업")
        backup_database(self.db_path)

        # API 키 확인
        has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
        if not has_api_key:
            print("   ⚠️ ANTHROPIC_API_KEY 없음 - 이미지 설명 비활성화")
            extract_images = False

        # PDF 처리
        print("\n📄 PDF 처리 중...")
        processor = EnhancedPDFProcessor()

        if os.path.isfile(pdf_path) and pdf_path.lower().endswith('.pdf'):
            content = processor.process_pdf(
                pdf_path,
                extract_images=extract_images,
                extract_tables=extract_tables
            )
        elif os.path.isdir(pdf_path):
            content = processor.process_folder(
                pdf_path,
                extract_images=extract_images,
                extract_tables=extract_tables
            )
        else:
            print(f"❌ 유효하지 않은 경로: {pdf_path}")
            return

        if not content:
            print("❌ 처리된 컨텐츠가 없습니다.")
            return

        # 업그레이드된 청킹
        print(f"\n✂️ {chunk_strategy} 청킹 적용 중...")
        chunks = content_to_chunks_v2(
            content,
            strategy=chunk_strategy,
            embedding_model=self.embedding_model if chunk_strategy == "semantic" else None,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )
        print(f"   ✅ {len(chunks):,}개 청크 생성")

        # 임베딩 생성
        print("\n🔢 임베딩 생성 중...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        print(f"   ✅ 벡터 변환 완료 (차원: {embeddings.shape[1]})")

        # 기존 컬렉션 삭제
        try:
            self.client.delete_collection("rag_documents")
            print("\n🗑️ 기존 컬렉션 삭제")
        except Exception:
            pass

        # 새 컬렉션 생성
        self.collection = self.client.create_collection(
            name="rag_documents",
            metadata={
                "hnsw:space": "cosine",
                "chunk_strategy": chunk_strategy,
                "version": "v3"
            }
        )

        # 데이터 저장
        print("\n💾 데이터베이스 저장 중...")
        ids = [f"chunk_{c['chunk_id']}" for c in chunks]
        metadatas = [
            {
                "page": c["page"],
                "source": c.get("source", "unknown"),
                "strategy": c.get("strategy", chunk_strategy)
            }
            for c in chunks
        ]

        BATCH_SIZE = 5000
        for i in range(0, len(chunks), BATCH_SIZE):
            end = min(i + BATCH_SIZE, len(chunks))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end].tolist(),
                documents=texts[i:end],
                metadatas=metadatas[i:end]
            )
            print(f"   📦 배치 {i//BATCH_SIZE + 1}: {end - i:,}개 저장")

        print(f"\n✅ 데이터베이스 재구축 완료! ({len(chunks):,}개 청크)")

        # Enhanced Retriever 재초기화
        self._init_enhanced_retriever()

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        fetch_k: int = FETCH_K
    ) -> List[Dict[str, Any]]:
        """향상된 검색 실행"""
        if not self.retriever:
            print("❌ Retriever가 초기화되지 않았습니다.")
            return []

        # 쿼리 임베딩
        query_embedding = self.embedding_model.encode([query])[0]

        # Enhanced Retriever로 검색
        results = self.retriever.search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            fetch_k=fetch_k
        )

        return results

    def generate_answer(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """검색 결과로 답변 생성"""
        if not self.llm.is_available():
            return {
                "answer": None,
                "error": "LLM 클라이언트가 설정되지 않았습니다.",
                "sources": []
            }

        # 컨텍스트 구성
        context_parts = []
        sources = []

        for i, r in enumerate(results):
            context_parts.append(f"[문서 {i+1}]\n{r['document']}")
            sources.append({
                "source": r["metadata"].get("source", "unknown"),
                "page": r["metadata"].get("page", 0),
                "score": r.get("final_score", 0)
            })

        context = "\n\n---\n\n".join(context_parts)

        # LLM으로 답변 생성
        try:
            response = self.llm.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": f"""다음 문서 내용을 바탕으로 질문에 답변해주세요.
답변은 문서 내용에 근거해야 합니다. 문서에 없는 내용은 답변하지 마세요.

## 문서 내용
{context}

## 질문
{query}

## 답변"""
                    }
                ]
            )

            return {
                "answer": response.content[0].text,
                "error": None,
                "sources": sources
            }

        except Exception as e:
            return {
                "answer": None,
                "error": str(e),
                "sources": sources
            }

    def interactive_mode(self):
        """대화형 검색 모드"""
        print("\n" + "="*60)
        print("💬 대화형 RAG 검색 모드 (v3)")
        print("="*60)
        print("질문을 입력하면 관련 문서를 찾아 답변합니다.")
        print()
        print("📌 특수 명령어:")
        print("   /config  - 현재 설정 확인")
        print("   /hybrid  - Hybrid Search 토글")
        print("   /rerank  - Reranking 토글")
        print("   /eval    - 최근 검색 품질 평가")
        print("   q, quit  - 종료")
        print("="*60)

        while True:
            print()
            query = input("❓ 질문: ").strip()

            # 특수 명령어 처리
            if query.lower() in ["q", "quit", "exit", "종료"]:
                print("\n👋 프로그램을 종료합니다.")
                break

            if query == "/config":
                self._print_config()
                continue

            if query == "/hybrid":
                self.use_hybrid = not self.use_hybrid
                self._init_enhanced_retriever()
                print(f"   Hybrid Search: {'✅ 활성화' if self.use_hybrid else '❌ 비활성화'}")
                continue

            if query == "/rerank":
                self.use_reranking = not self.use_reranking
                self._init_enhanced_retriever()
                print(f"   Reranking: {'✅ 활성화' if self.use_reranking else '❌ 비활성화'}")
                continue

            if query == "/eval":
                summary = self.evaluator.get_summary()
                if summary:
                    print("\n📊 검색 품질 평가 요약:")
                    for k, v in summary.items():
                        print(f"   {k}: {v:.3f}")
                else:
                    print("   아직 평가 데이터가 없습니다.")
                continue

            if not query:
                print("   ⚠️ 질문을 입력해주세요.")
                continue

            # 검색 실행
            print(f"\n🔍 검색 중...")
            results = self.search(query)

            if not results:
                print("   검색 결과가 없습니다.")
                continue

            # 결과 출력
            self._print_results(query, results)

            # LLM 답변 생성
            if self.llm.is_available():
                print(f"\n{'='*60}")
                print("🤖 AI 답변 생성 중...")
                print("="*60)

                answer_result = self.generate_answer(query, results)

                if answer_result["error"]:
                    print(f"\n❌ 오류: {answer_result['error']}")
                else:
                    print(f"\n{answer_result['answer']}")

                    if answer_result["sources"]:
                        print(f"\n{'─'*40}")
                        print("📚 참고 문서:")
                        for src in answer_result["sources"][:3]:
                            print(f"   • {src['source']} (p.{src['page']})")

    def _print_results(self, query: str, results: List[Dict]):
        """검색 결과 출력"""
        print(f"\n{'='*60}")
        print(f"🔍 질문: {query}")
        print(f"{'='*60}")

        for i, r in enumerate(results):
            score = r.get("final_score", r.get("score", 0))
            meta = r.get("metadata", {})

            print(f"\n📌 결과 {i+1} (점수: {score:.3f})")
            print(f"   📄 출처: {meta.get('source', 'unknown')}, 페이지: {meta.get('page', '?')}")

            # 검색 방법 표시
            methods = []
            if r.get("in_dense"):
                methods.append("Dense")
            if r.get("in_sparse"):
                methods.append("BM25")
            if r.get("rerank_score"):
                methods.append(f"Reranked({r['rerank_score']:.2f})")
            if methods:
                print(f"   🔧 검색: {', '.join(methods)}")

            print(f"   {'-'*50}")
            preview = r["document"][:300].replace("\n", "\n   ")
            print(f"   {preview}")
            if len(r["document"]) > 300:
                print(f"   ... (총 {len(r['document'])}자)")


# ============================================================
# 메인 실행
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RAG 시스템 v3")
    parser.add_argument("--rebuild", action="store_true", help="데이터베이스 재구축")
    parser.add_argument("--pdf-path", default=PDF_PATH, help="PDF 폴더 경로")
    parser.add_argument("--strategy", default=CHUNK_STRATEGY,
                        choices=["semantic", "adaptive", "hierarchical", "fixed"],
                        help="청킹 전략")
    parser.add_argument("--no-hybrid", action="store_true", help="Hybrid Search 비활성화")
    parser.add_argument("--no-rerank", action="store_true", help="Reranking 비활성화")
    parser.add_argument("--evaluate", action="store_true", help="평가 모드")
    args = parser.parse_args()

    # 시스템 초기화
    rag = RAGSystemV3(
        use_hybrid=not args.no_hybrid,
        use_reranking=not args.no_rerank
    )

    # 재구축 모드
    if args.rebuild:
        rag.rebuild_database(
            pdf_path=args.pdf_path,
            chunk_strategy=args.strategy
        )

    # 컬렉션 확인
    if not rag.collection:
        print("\n❌ 데이터베이스가 없습니다.")
        print("   --rebuild 옵션으로 데이터베이스를 생성하세요.")
        print(f"   예: python rag_system_v3.py --rebuild --pdf-path {PDF_PATH}")
        return

    # 대화형 모드
    rag.interactive_mode()


if __name__ == "__main__":
    main()
