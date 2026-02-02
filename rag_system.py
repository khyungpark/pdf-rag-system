"""
========================================
📚 RAG 시스템 - 임상시험 문서 질의응답
========================================

이 프로그램은 PDF 문서를 읽어서 AI가 질문에 답변할 수 있도록
벡터 데이터베이스를 구축하는 프로그램입니다.

[RAG란?]
RAG = Retrieval-Augmented Generation (검색 증강 생성)
→ 문서에서 관련 내용을 "검색"한 후, AI가 그 내용을 바탕으로 "답변"하는 방식

[작동 순서]
1단계: PDF에서 텍스트 추출
2단계: 텍스트를 작은 조각(청크)으로 분할
3단계: 각 조각을 숫자 벡터로 변환 (임베딩)
4단계: 벡터 데이터베이스(ChromaDB)에 저장
5단계: 질문하면 관련 조각을 찾아서 답변

사용법:
    python rag_system.py
"""

import os
import sys

# ============================================================
# 1단계: 필요한 라이브러리 불러오기
# ============================================================
try:
    import fitz  # PyMuPDF - PDF 읽기용
    import chromadb  # 벡터 데이터베이스
    from sentence_transformers import SentenceTransformer  # 임베딩 모델
except ImportError as e:
    print("❌ 필요한 패키지가 설치되지 않았습니다!")
    print("   아래 명령어를 먼저 실행해주세요:")
    print()
    print("   pip install -r requirements.txt")
    print()
    print(f"   (에러 상세: {e})")
    sys.exit(1)


# ============================================================
# 2단계: 설정값 (필요에 따라 수정하세요)
# ============================================================

# PDF 파일 경로 - 여기에 본인의 PDF 파일 경로를 넣으세요
PDF_PATH = "14__임상시험의_개요_주완석.pdf"

# 텍스트를 자르는 크기 (글자 수)
# - 너무 작으면: 맥락을 잃어버림
# - 너무 크면: 검색 정확도가 떨어짐
# - 보통 500~1000이 적당합니다
CHUNK_SIZE = 500

# 청크 간 겹치는 부분 (글자 수)
# - 문장이 잘리는 것을 방지하기 위해 앞뒤로 겹치게 합니다
CHUNK_OVERLAP = 100

# 임베딩 모델 이름 (한국어 지원 모델)
# - 이 모델이 텍스트를 숫자 벡터로 바꿔줍니다
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 데이터베이스 저장 폴더
DB_PATH = "./chroma_db"

# 검색 시 가져올 결과 수
TOP_K = 3


# ============================================================
# 3단계: PDF에서 텍스트 추출하기
# ============================================================

def extract_text_from_pdf(pdf_path):
    """
    PDF 파일에서 텍스트를 추출합니다.
    
    매개변수:
        pdf_path: PDF 파일의 경로 (예: "문서.pdf")
    
    반환값:
        페이지별 텍스트 리스트 [{"page": 1, "text": "..."}, ...]
    """
    print(f"\n📄 PDF 읽는 중: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ 파일을 찾을 수 없습니다: {pdf_path}")
        print(f"   현재 폴더: {os.getcwd()}")
        print(f"   폴더 내 파일들: {os.listdir('.')}")
        sys.exit(1)
    
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        # 빈 페이지는 건너뛰기
        if text.strip():
            pages.append({
                "page": page_num + 1,
                "text": text.strip()
            })
    
    doc.close()
    print(f"   ✅ 총 {len(pages)}페이지에서 텍스트 추출 완료")
    return pages


# ============================================================
# 4단계: 텍스트를 청크(조각)로 분할하기
# ============================================================

def split_into_chunks(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    긴 텍스트를 작은 조각(청크)으로 나눕니다.

    왜 나누나요?
    → AI가 한 번에 처리할 수 있는 텍스트 양에 한계가 있고,
      작은 조각으로 나눠야 "관련 부분만" 정확하게 찾을 수 있습니다.

    개선: 문장 경계(. ! ? 등)에서 분할하여 문맥 유지

    매개변수:
        pages: extract_text_from_pdf()의 결과
        chunk_size: 한 조각의 최대 글자 수
        overlap: 조각 간 겹치는 글자 수
    """
    import re

    print(f"\n✂️  텍스트 분할 중 (청크 크기: {chunk_size}자, 겹침: {overlap}자)")

    chunks = []

    def find_sentence_boundary(text, position, direction="forward"):
        """문장 경계를 찾습니다."""
        sentence_endings = re.compile(r'[.!?]\s+|[.!?]$|\n\n')

        if direction == "forward":
            match = sentence_endings.search(text, position)
            if match and match.end() - position < 100:  # 100자 이내에서 문장 끝 찾기
                return match.end()
        else:  # backward
            # 역방향으로 문장 끝 찾기
            text_before = text[:position]
            matches = list(sentence_endings.finditer(text_before))
            if matches:
                last_match = matches[-1]
                if position - last_match.end() < 100:
                    return last_match.end()
        return position

    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]

        # 텍스트가 청크 크기보다 작으면 그대로 사용
        if len(text) <= chunk_size:
            chunks.append({
                "text": text.strip(),
                "page": page_num,
                "chunk_id": len(chunks)
            })
        else:
            # 문장 경계를 고려하여 청크 분할
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))

                # 문장 경계에서 분할 시도
                if end < len(text):
                    end = find_sentence_boundary(text, end, "forward")

                chunk_text = text[start:end].strip()

                if chunk_text:  # 빈 청크는 건너뛰기
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num,
                        "chunk_id": len(chunks)
                    })

                # 다음 시작점 = 현재 끝 - 겹침 (문장 경계 고려)
                next_start = end - overlap
                if next_start > start:
                    next_start = find_sentence_boundary(text, next_start, "backward")
                start = max(next_start, end - overlap, start + 1)  # 무한루프 방지

    print(f"   ✅ 총 {len(chunks)}개 청크 생성 완료")
    
    # 처음 3개 청크 미리보기
    print("\n   [청크 미리보기]")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk["text"][:80].replace("\n", " ")
        print(f"   청크 {i}: (p.{chunk['page']}) {preview}...")
    
    return chunks


# ============================================================
# 5단계: 임베딩 모델 로드 & 벡터 데이터베이스 구축
# ============================================================

def build_vector_database(chunks):
    """
    텍스트 청크를 벡터로 변환하여 데이터베이스에 저장합니다.
    
    [임베딩이란?]
    텍스트를 숫자 배열(벡터)로 바꾸는 것입니다.
    예: "임상시험" → [0.12, -0.34, 0.56, ...]
    
    비슷한 의미의 텍스트는 비슷한 벡터를 가지게 되어,
    나중에 질문과 비슷한 내용을 수학적으로 찾을 수 있습니다.
    
    [ChromaDB란?]
    이런 벡터들을 저장하고 검색할 수 있는 데이터베이스입니다.
    SQLite처럼 파일로 저장되어 서버 없이도 사용 가능합니다.
    """
    # --- 임베딩 모델 로드 ---
    print(f"\n🤖 임베딩 모델 로드 중: {EMBEDDING_MODEL}")
    print("   (처음 실행 시 모델 다운로드에 몇 분 걸릴 수 있습니다)")
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("   ✅ 모델 로드 완료")
    
    # --- 텍스트를 벡터로 변환 ---
    print(f"\n🔢 {len(chunks)}개 청크를 벡터로 변환 중...")
    
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print(f"   ✅ 벡터 변환 완료 (벡터 차원: {embeddings.shape[1]})")
    
    # --- ChromaDB에 저장 ---
    print(f"\n💾 벡터 데이터베이스 저장 중: {DB_PATH}")
    
    # 기존 DB가 있으면 삭제 후 새로 생성
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 컬렉션(=테이블) 생성
    # 같은 이름이 있으면 삭제 후 재생성
    try:
        client.delete_collection("clinical_trial_docs")
    except Exception:
        # 컬렉션이 존재하지 않을 때 발생하는 예외 (NotFoundError) - 무시
        pass
    
    collection = client.create_collection(
        name="clinical_trial_docs",
        metadata={"description": "임상시험 문서 RAG 데이터베이스", "hnsw:space": "cosine"}
    )
    
    # 데이터 추가
    collection.add(
        ids=[f"chunk_{chunk['chunk_id']}" for chunk in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"page": chunk["page"], "chunk_id": chunk["chunk_id"]} for chunk in chunks]
    )
    
    print(f"   ✅ 데이터베이스 저장 완료! ({len(chunks)}개 청크 저장됨)")
    print(f"   📁 저장 위치: {os.path.abspath(DB_PATH)}")
    
    return model, collection


# ============================================================
# 6단계: 질문하고 답변 받기 (검색)
# ============================================================

def search(query, model, collection, top_k=TOP_K):
    """
    질문을 받아서 관련 문서 조각을 찾아줍니다.
    
    작동 방식:
    1. 질문을 벡터로 변환
    2. 데이터베이스에서 가장 비슷한 벡터를 가진 청크 찾기
    3. 찾은 청크 반환
    
    매개변수:
        query: 질문 문자열
        model: 임베딩 모델
        collection: ChromaDB 컬렉션
        top_k: 가져올 결과 수
    """
    # 질문을 벡터로 변환
    query_embedding = model.encode([query]).tolist()
    
    # 데이터베이스에서 검색
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    return results


def print_search_results(query, results):
    """검색 결과를 보기 좋게 출력합니다."""
    print(f"\n{'='*60}")
    print(f"🔍 질문: {query}")
    print(f"{'='*60}")
    
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        # 코사인 거리를 유사도로 변환 (코사인 거리 = 1 - 코사인 유사도)
        similarity = round(1 - dist, 2)  # 코사인 유사도 (0~1)
        
        print(f"\n📌 결과 {i+1} (유사도: {similarity}, 페이지: {meta['page']})")
        print(f"   {'-'*50}")
        # 텍스트 미리보기 (300자까지)
        preview = doc[:300].replace("\n", "\n   ")
        print(f"   {preview}")
        if len(doc) > 300:
            print(f"   ... (총 {len(doc)}자)")
    
    print(f"\n{'='*60}")


# ============================================================
# 7단계: 대화형 인터페이스
# ============================================================

def interactive_mode(model, collection):
    """
    사용자가 질문을 입력하면 관련 문서를 찾아주는 대화형 모드입니다.
    'quit' 또는 'q'를 입력하면 종료됩니다.
    """
    print("\n" + "="*60)
    print("💬 대화형 RAG 검색 모드")
    print("="*60)
    print("질문을 입력하면 관련 문서 내용을 찾아드립니다.")
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
        
        # 검색된 내용을 LLM에 전달할 수 있는 프롬프트 생성
        context = "\n\n---\n\n".join(results["documents"][0])
        print(f"\n💡 [팁] 아래 텍스트를 ChatGPT/Claude에 붙여넣고 질문하면")
        print(f"   문서 기반 답변을 받을 수 있습니다:")
        print(f"\n   다음 문서 내용을 바탕으로 답변해주세요:")
        print(f"   ---")
        print(f"   {context[:200]}...")


# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    print()
    print("🚀" + "="*58)
    print("   RAG 시스템 - 임상시험 문서 질의응답")
    print("="*60)
    
    # --- Step 1: PDF 텍스트 추출 ---
    pages = extract_text_from_pdf(PDF_PATH)
    
    # --- Step 2: 청크 분할 ---
    chunks = split_into_chunks(pages)
    
    # --- Step 3: 벡터 DB 구축 ---
    model, collection = build_vector_database(chunks)
    
    # --- Step 4: 대화형 검색 ---
    interactive_mode(model, collection)
