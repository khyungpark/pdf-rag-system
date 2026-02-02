"""
RAG 시스템 대시보드
Streamlit 기반 웹 인터페이스
"""

import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

# ============================================================
# 설정
# ============================================================
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_K = 5

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="RAG 문서 검색 시스템",
    page_icon="📚",
    layout="wide"
)

# ============================================================
# 세션 상태 초기화
# ============================================================
if "model" not in st.session_state:
    st.session_state.model = None
if "collection" not in st.session_state:
    st.session_state.collection = None
if "search_history" not in st.session_state:
    st.session_state.search_history = []


# ============================================================
# 유틸리티 함수
# ============================================================
@st.cache_resource
def load_model():
    """임베딩 모델 로드 (캐시됨)"""
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_collection():
    """ChromaDB 컬렉션 로드 (캐시됨)"""
    if not os.path.exists(DB_PATH):
        return None
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        return client.get_collection("clinical_trial_docs")
    except Exception:
        return None


def search(query, model, collection, top_k):
    """벡터 검색 수행"""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results


def extract_text_from_pdf(pdf_file):
    """업로드된 PDF에서 텍스트 추출"""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages.append({
                "page": page_num + 1,
                "text": text.strip()
            })
    doc.close()
    return pages


def split_into_chunks(pages, chunk_size=500, overlap=100):
    """텍스트를 청크로 분할"""
    import re
    chunks = []
    sentence_endings = re.compile(r'[.!?]\s+|[.!?]$|\n\n')

    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]

        if len(text) <= chunk_size:
            chunks.append({
                "text": text.strip(),
                "page": page_num,
                "chunk_id": len(chunks)
            })
        else:
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
                        "chunk_id": len(chunks)
                    })

                next_start = end - overlap
                start = max(next_start, start + 1)

    return chunks


def build_database(chunks, model):
    """벡터 데이터베이스 구축"""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection("clinical_trial_docs")
    except Exception:
        pass

    collection = client.create_collection(
        name="clinical_trial_docs",
        metadata={"description": "RAG 문서 데이터베이스", "hnsw:space": "cosine"}
    )

    collection.add(
        ids=[f"chunk_{chunk['chunk_id']}" for chunk in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"page": chunk["page"], "chunk_id": chunk["chunk_id"]} for chunk in chunks]
    )

    return collection


# ============================================================
# 메인 UI
# ============================================================
st.title("📚 RAG 문서 검색 시스템")
st.markdown("PDF 문서에서 관련 내용을 AI로 검색합니다")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")

    top_k = st.slider("검색 결과 수", 1, 10, DEFAULT_TOP_K)

    st.divider()

    st.header("📄 PDF 업로드")
    uploaded_file = st.file_uploader("새 PDF 파일 업로드", type="pdf")

    if uploaded_file:
        if st.button("📥 DB 구축", type="primary"):
            with st.spinner("PDF 처리 중..."):
                # 텍스트 추출
                pages = extract_text_from_pdf(uploaded_file)
                st.info(f"✅ {len(pages)}페이지 추출 완료")

                # 청크 분할
                chunks = split_into_chunks(pages)
                st.info(f"✅ {len(chunks)}개 청크 생성")

                # 모델 로드
                model = load_model()

                # DB 구축
                collection = build_database(chunks, model)
                st.session_state.collection = collection

                st.success("✅ 데이터베이스 구축 완료!")
                st.rerun()

    st.divider()

    # DB 상태
    st.header("💾 DB 상태")
    collection = load_collection()
    if collection:
        count = collection.count()
        st.success(f"✅ 연결됨 ({count:,}개 청크)")
    else:
        st.warning("⚠️ DB 없음 - PDF를 먼저 업로드하세요")

# 메인 영역
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 질문하기")

    query = st.text_input(
        "질문을 입력하세요",
        placeholder="예: 임상시험 1상이란?",
        label_visibility="collapsed"
    )

    search_button = st.button("검색", type="primary", use_container_width=True)

# 검색 실행
if search_button and query:
    collection = load_collection()

    if not collection:
        st.error("❌ 먼저 PDF를 업로드하여 데이터베이스를 구축하세요")
    else:
        model = load_model()

        with st.spinner("검색 중..."):
            results = search(query, model, collection, top_k)

        # 검색 기록 저장
        st.session_state.search_history.append({
            "query": query,
            "results_count": len(results["documents"][0])
        })

        st.header("📋 검색 결과")

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            similarity = round(1 - dist, 2)

            # 결과 카드
            with st.container():
                col_score, col_page = st.columns([1, 1])
                with col_score:
                    # 유사도에 따른 색상
                    if similarity >= 0.8:
                        st.success(f"유사도: {similarity}")
                    elif similarity >= 0.6:
                        st.warning(f"유사도: {similarity}")
                    else:
                        st.error(f"유사도: {similarity}")
                with col_page:
                    st.info(f"📄 페이지: {meta['page']}")

                st.markdown(f"**결과 {i+1}**")
                st.text_area(
                    f"내용_{i}",
                    doc[:500] + ("..." if len(doc) > 500 else ""),
                    height=150,
                    label_visibility="collapsed"
                )
                st.divider()

        # LLM 프롬프트 생성
        st.header("💡 LLM에 전달할 프롬프트")
        context = "\n\n---\n\n".join(documents)
        prompt = f"""다음 문서 내용을 바탕으로 질문에 답변해주세요.

[문서 내용]
{context}

[질문]
{query}

[답변]"""

        st.code(prompt, language=None)
        st.button("📋 복사", help="위 텍스트를 복사해서 ChatGPT/Claude에 붙여넣으세요")

with col2:
    st.header("📊 통계")

    collection = load_collection()
    if collection:
        count = collection.count()

        st.metric("총 청크 수", f"{count:,}")
        st.metric("검색 횟수", len(st.session_state.search_history))

        if st.session_state.search_history:
            st.subheader("🕐 최근 검색")
            for item in reversed(st.session_state.search_history[-5:]):
                st.text(f"• {item['query'][:30]}...")
    else:
        st.info("DB를 먼저 구축하세요")

# 푸터
st.divider()
st.caption("RAG 시스템 대시보드 | Streamlit + ChromaDB + Sentence-Transformers")
