"""
RAG 시스템 대시보드 v2
- 여러 PDF 파일 지원
- 표/이미지 추출 옵션
- Claude Vision 연동
- Claude LLM 답변 생성
"""

import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from pdf_processor import EnhancedPDFProcessor, content_to_chunks
from llm_client import get_llm_client

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
    page_title="RAG 문서 검색 시스템 v2",
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
if "processing" not in st.session_state:
    st.session_state.processing = False


# ============================================================
# 유틸리티 함수
# ============================================================
@st.cache_resource
def load_model():
    """임베딩 모델 로드"""
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_collection():
    """ChromaDB 컬렉션 로드"""
    if not os.path.exists(DB_PATH):
        return None
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        return client.get_collection("rag_documents")
    except Exception:
        try:
            return client.get_collection("clinical_trial_docs")
        except Exception:
            return None


def search(query, model, collection, top_k):
    """벡터 검색"""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results


def build_database(chunks, model, progress_callback=None):
    """벡터 데이터베이스 구축"""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection("rag_documents")
    except Exception:
        pass

    collection = client.create_collection(
        name="rag_documents",
        metadata={"description": "RAG 문서 데이터베이스", "hnsw:space": "cosine"}
    )

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

    return collection


# ============================================================
# 메인 UI
# ============================================================
st.title("📚 RAG 문서 검색 시스템 v2")
st.markdown("PDF 문서의 **텍스트, 표, 이미지**를 AI로 검색합니다")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")

    top_k = st.slider("검색 결과 수", 1, 10, DEFAULT_TOP_K)

    # AI 답변 옵션
    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    use_ai_answer = st.checkbox(
        "🤖 AI 답변 생성",
        value=has_api_key,
        disabled=not has_api_key,
        help="Claude가 검색 결과를 바탕으로 답변을 생성합니다"
    )

    st.divider()

    # API 키 상태
    st.header("🔑 API 설정")
    if has_api_key:
        st.success("✅ Anthropic API 연결됨")
    else:
        st.warning("⚠️ API 키 없음 - 이미지 설명 비활성화")
        api_key = st.text_input("Anthropic API Key", type="password")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            st.success("✅ API 키 설정됨")
            st.rerun()

    st.divider()

    # PDF 업로드
    st.header("📄 PDF 업로드")
    uploaded_files = st.file_uploader(
        "PDF 파일 선택 (여러 개 가능)",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        # 추출 옵션
        extract_tables = st.checkbox("📊 표 추출", value=True)
        extract_images = st.checkbox(
            "🖼️ 이미지 설명 (Claude Vision)",
            value=has_api_key,
            disabled=not has_api_key
        )

        if st.button("📥 DB 구축", type="primary", use_container_width=True):
            st.session_state.processing = True

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                processor = EnhancedPDFProcessor()
                all_content = []

                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"처리 중: {uploaded_file.name}")
                    progress_bar.progress((i + 1) / (len(uploaded_files) + 2))

                    # 임시 파일로 저장
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # PDF 처리
                    content = processor.process_pdf(
                        temp_path,
                        extract_images=extract_images,
                        extract_tables=extract_tables
                    )
                    all_content.extend(content)

                    # 임시 파일 삭제
                    os.remove(temp_path)

                # 청크 생성
                status_text.text("청크 생성 중...")
                chunks = content_to_chunks(all_content)

                # 모델 로드
                status_text.text("임베딩 모델 로드 중...")
                progress_bar.progress(0.8)
                model = load_model()

                # DB 구축
                status_text.text("벡터 DB 구축 중...")
                collection = build_database(chunks, model)
                progress_bar.progress(1.0)

                # 캐시 갱신을 위해 rerun
                st.cache_resource.clear()

                # 통계
                total_tables = sum(len(p.get("tables", [])) for p in all_content)
                total_images = sum(len(p.get("images", [])) for p in all_content)

                status_text.empty()
                st.success(f"""✅ 완료!
                - {len(uploaded_files)}개 PDF
                - {len(chunks)}개 청크
                - {total_tables}개 표
                - {total_images}개 이미지""")

            except Exception as e:
                st.error(f"❌ 오류: {e}")
            finally:
                st.session_state.processing = False

    st.divider()

    # DB 상태
    st.header("💾 DB 상태")
    collection = load_collection()
    if collection:
        try:
            count = collection.count()
            st.success(f"✅ 연결됨 ({count:,}개 청크)")
        except Exception:
            st.warning("⚠️ DB 재구축 필요")
            collection = None
    else:
        st.warning("⚠️ DB 없음")

# 메인 영역
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 질문하기")

    query = st.text_input(
        "질문을 입력하세요",
        placeholder="예: 임상시험 1상의 목적은?",
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

        # 검색 기록
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

            with st.container():
                # 메타 정보
                col_score, col_source, col_page = st.columns([1, 1, 1])
                with col_score:
                    if similarity >= 0.8:
                        st.success(f"유사도: {similarity}")
                    elif similarity >= 0.6:
                        st.warning(f"유사도: {similarity}")
                    else:
                        st.error(f"유사도: {similarity}")
                with col_source:
                    st.info(f"📄 {meta.get('source', 'unknown')}")
                with col_page:
                    st.info(f"페이지: {meta['page']}")

                # 컨텐츠 타입 표시
                content_type = "텍스트"
                if "[표]" in doc:
                    content_type = "📊 표 포함"
                elif "[이미지 설명]" in doc:
                    content_type = "🖼️ 이미지 설명"

                st.markdown(f"**결과 {i+1}** - {content_type}")
                st.text_area(
                    f"내용_{i}",
                    doc[:600] + ("..." if len(doc) > 600 else ""),
                    height=150,
                    label_visibility="collapsed"
                )
                st.divider()

        # AI 답변 또는 LLM 프롬프트
        if use_ai_answer and has_api_key:
            st.header("🤖 AI 답변")
            with st.spinner("Claude가 답변을 생성하는 중..."):
                llm = get_llm_client()
                answer_result = llm.generate_answer(query, results)

            if answer_result["error"]:
                st.error(f"❌ {answer_result['error']}")
            else:
                st.markdown(answer_result["answer"])

                # 출처 표시
                if answer_result["sources"]:
                    st.divider()
                    st.caption("📚 참고 문서:")
                    for src in answer_result["sources"][:3]:
                        st.caption(f"  • {src['source']} (p.{src['page']}, 유사도: {src['similarity']})")
        else:
            # 기존 LLM 프롬프트 표시
            st.header("💡 LLM 프롬프트")
            context = "\n\n---\n\n".join(documents)
            prompt = f"""다음 문서 내용을 바탕으로 질문에 답변해주세요.

[문서 내용]
{context}

[질문]
{query}

[답변]"""

            st.code(prompt, language=None)
            st.caption("💡 위 프롬프트를 ChatGPT/Claude에 붙여넣어 답변을 받으세요")

with col2:
    st.header("📊 통계")

    collection = load_collection()
    if collection:
        try:
            count = collection.count()
            st.metric("총 청크 수", f"{count:,}")
            st.metric("검색 횟수", len(st.session_state.search_history))

            if st.session_state.search_history:
                st.subheader("🕐 최근 검색")
                for item in reversed(st.session_state.search_history[-5:]):
                    st.text(f"• {item['query'][:25]}...")
        except Exception:
            st.info("DB를 재구축하세요")
    else:
        st.info("DB를 먼저 구축하세요")

# 푸터
st.divider()
st.caption("RAG 시스템 v2 | 텍스트 + 표 + 이미지 | Streamlit + ChromaDB + Claude Vision")
