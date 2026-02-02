"""
========================================
📚 RAG 시스템 v2 - 향상된 문서 질의응답
========================================

개선사항:
- 여러 PDF 파일 지원 (폴더 기반)
- 표(Table) 추출
- 이미지 설명 생성 (Claude Vision)

사용법:
    python rag_system_v2.py              # 현재 폴더의 모든 PDF 처리
    python rag_system_v2.py ./docs       # 특정 폴더의 PDF 처리
"""

import os
import sys

# ============================================================
# 필요한 라이브러리 불러오기
# ============================================================
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    from pdf_processor import EnhancedPDFProcessor, content_to_chunks
    from llm_client import get_llm_client
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

# PDF 파일 경로 (폴더 또는 단일 파일)
PDF_PATH = "."  # 현재 폴더

# 텍스트를 자르는 크기 (글자 수)
CHUNK_SIZE = 500

# 청크 간 겹치는 부분 (글자 수)
CHUNK_OVERLAP = 100

# 임베딩 모델
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 데이터베이스 저장 폴더
DB_PATH = "./chroma_db"

# 검색 시 가져올 결과 수
TOP_K = 5

# 이미지 처리 여부 (ANTHROPIC_API_KEY 필요)
EXTRACT_IMAGES = True

# 표 추출 여부
EXTRACT_TABLES = True


# ============================================================
# 벡터 데이터베이스 구축
# ============================================================

def build_vector_database(chunks):
    """청크를 벡터 데이터베이스에 저장"""
    print(f"\n🤖 임베딩 모델 로드 중: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("   ✅ 모델 로드 완료")

    print(f"\n🔢 {len(chunks)}개 청크를 벡터로 변환 중...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"   ✅ 벡터 변환 완료 (벡터 차원: {embeddings.shape[1]})")

    print(f"\n💾 벡터 데이터베이스 저장 중: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)

    # 기존 컬렉션 삭제
    try:
        client.delete_collection("rag_documents")
    except Exception:
        pass

    collection = client.create_collection(
        name="rag_documents",
        metadata={"description": "RAG 문서 데이터베이스", "hnsw:space": "cosine"}
    )

    # 메타데이터 구성
    metadatas = []
    for chunk in chunks:
        metadatas.append({
            "page": chunk["page"],
            "source": chunk.get("source", "unknown"),
            "chunk_id": chunk["chunk_id"]
        })

    # 배치 크기 제한 (ChromaDB max: 5461)
    BATCH_SIZE = 5000
    ids = [f"chunk_{chunk['chunk_id']}" for chunk in chunks]
    embeddings_list = embeddings.tolist()

    for i in range(0, len(chunks), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(chunks))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings_list[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"   📦 배치 {i//BATCH_SIZE + 1}: {end - i}개 청크 저장")

    print(f"   ✅ 데이터베이스 저장 완료! ({len(chunks)}개 청크)")
    print(f"   📁 저장 위치: {os.path.abspath(DB_PATH)}")

    return model, collection


# ============================================================
# 검색 함수
# ============================================================

def search(query, model, collection, top_k=TOP_K):
    """질문을 받아 관련 문서 검색"""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results


def print_search_results(query, results):
    """검색 결과 출력"""
    print(f"\n{'='*60}")
    print(f"🔍 질문: {query}")
    print(f"{'='*60}")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        similarity = round(1 - dist, 2)

        print(f"\n📌 결과 {i+1} (유사도: {similarity})")
        print(f"   📄 출처: {meta.get('source', 'unknown')}, 페이지: {meta['page']}")
        print(f"   {'-'*50}")
        preview = doc[:300].replace("\n", "\n   ")
        print(f"   {preview}")
        if len(doc) > 300:
            print(f"   ... (총 {len(doc)}자)")

    print(f"\n{'='*60}")


# ============================================================
# 대화형 인터페이스
# ============================================================

def interactive_mode(model, collection, use_ai=True):
    """대화형 검색 모드"""
    # LLM 클라이언트 초기화
    llm = get_llm_client()
    ai_available = llm.is_available() and use_ai

    print("\n" + "="*60)
    print("💬 대화형 RAG 검색 모드 (v2)")
    print("="*60)
    print("질문을 입력하면 관련 문서 내용을 찾아드립니다.")
    print("표와 이미지 설명도 검색됩니다.")
    if ai_available:
        print("🤖 AI 답변 기능이 활성화되어 있습니다.")
    else:
        print("💡 AI 답변을 받으려면 ANTHROPIC_API_KEY를 설정하세요.")
    print("종료하려면 'q' 또는 'quit'를 입력하세요.")
    print("="*60)

    while True:
        print()
        query = input("❓ 질문: ").strip()

        if query.lower() in ["q", "quit", "exit", "종료"]:
            print("\n👋 프로그램을 종료합니다.")
            break

        if not query:
            print("   ⚠️ 질문을 입력해주세요.")
            continue

        results = search(query, model, collection, top_k=TOP_K)
        print_search_results(query, results)

        # AI 답변 생성 또는 프롬프트 표시
        if ai_available:
            print(f"\n{'='*60}")
            print("🤖 AI 답변 생성 중...")
            print("="*60)

            answer_result = llm.generate_answer(query, results)

            if answer_result["error"]:
                print(f"\n❌ 오류: {answer_result['error']}")
            else:
                print(f"\n{answer_result['answer']}")

                # 출처 표시
                if answer_result["sources"]:
                    print(f"\n{'─'*40}")
                    print("📚 참고 문서:")
                    for src in answer_result["sources"][:3]:
                        print(f"   • {src['source']} (p.{src['page']}, 유사도: {src['similarity']})")
        else:
            # 기존 LLM 프롬프트 표시
            context = "\n\n---\n\n".join(results["documents"][0])
            print(f"\n💡 [팁] ChatGPT/Claude에 아래 프롬프트를 붙여넣으세요:")
            print(f"\n   다음 문서 내용을 바탕으로 '{query}'에 대해 답변해주세요:")
            print(f"   ---")
            print(f"   {context[:300]}...")


# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    print()
    print("🚀" + "="*58)
    print("   RAG 시스템 v2 - 향상된 문서 질의응답")
    print("="*60)

    # 경로 결정
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = PDF_PATH

    # API 키 확인
    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    if not has_api_key:
        print("\n⚠️ ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        print("   이미지 설명 기능이 비활성화됩니다.")
        print("   설정하려면: export ANTHROPIC_API_KEY='your-key'")

    # PDF 처리기 초기화
    processor = EnhancedPDFProcessor()

    # PDF 처리
    if os.path.isfile(path) and path.lower().endswith('.pdf'):
        # 단일 파일
        content = processor.process_pdf(
            path,
            extract_images=EXTRACT_IMAGES and has_api_key,
            extract_tables=EXTRACT_TABLES
        )
    elif os.path.isdir(path):
        # 폴더
        content = processor.process_folder(
            path,
            extract_images=EXTRACT_IMAGES and has_api_key,
            extract_tables=EXTRACT_TABLES
        )
    else:
        print(f"❌ 유효하지 않은 경로: {path}")
        sys.exit(1)

    if not content:
        print("❌ 처리된 컨텐츠가 없습니다.")
        sys.exit(1)

    # 청크 생성
    print(f"\n✂️ 청크 분할 중...")
    chunks = content_to_chunks(content, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"   ✅ 총 {len(chunks)}개 청크 생성 완료")

    # 벡터 DB 구축
    model, collection = build_vector_database(chunks)

    # 통계 출력
    sources = set(c.get("source", "unknown") for c in chunks)
    print(f"\n📊 데이터베이스 통계")
    print(f"   📚 PDF 파일: {len(sources)}개")
    print(f"   📄 총 청크: {len(chunks)}개")

    # 대화형 검색
    interactive_mode(model, collection)
