"""
========================================
🚀 RAG 시스템 대시보드 v3
========================================

업그레이드 기능:
- Hybrid Search (BM25 + Dense)
- Cross-Encoder Reranking
- Semantic Chunking
- 검색 품질 평가

실행: streamlit run dashboard_v3.py
"""

import os
import sys
import numpy as np
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# 로컬 모듈
from pdf_processor import EnhancedPDFProcessor
from llm_client import get_llm_client
from retrieval_upgrade import EnhancedRetriever, RAGEvaluator
from chunking_upgrade import content_to_chunks_v2

# ============================================================
# 설정
# ============================================================
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K = 5

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="RAG 시스템 v3",
    page_icon="🚀",
    layout="wide"
)

# ============================================================
# 세션 상태 초기화
# ============================================================
if "model" not in st.session_state:
    st.session_state.model = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "evaluator" not in st.session_state:
    st.session_state.evaluator = RAGEvaluator()


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
        return None


def init_enhanced_retriever(collection, model, use_hybrid, use_reranking, use_mmr):
    """Enhanced Retriever 초기화"""
    if not collection:
        return None

    all_data = collection.get(include=["documents", "embeddings", "metadatas"])

    documents = all_data["documents"]
    metadatas = all_data["metadatas"]

    # 임베딩 처리
    if all_data["embeddings"]:
        embeddings = np.array(all_data["embeddings"])
    else:
        embeddings = model.encode(documents, show_progress_bar=True)

    return EnhancedRetriever(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        use_hybrid=use_hybrid,
        use_reranking=use_reranking,
        use_mmr=use_mmr,
        dense_weight=0.7,
        reranker_model=RERANKER_MODEL
    )


def build_database_v3(chunks, model, chunk_strategy):
    """v3 벡터 데이터베이스 구축"""
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection("rag_documents")
    except Exception:
        pass

    collection = client.create_collection(
        name="rag_documents",
        metadata={
            "hnsw:space": "cosine",
            "chunk_strategy": chunk_strategy,
            "version": "v3"
        }
    )

    metadatas = []
    for chunk in chunks:
        metadatas.append({
            "page": chunk.get("page", 0),
            "source": chunk.get("source", "unknown"),
            "strategy": chunk.get("strategy", chunk_strategy)
        })

    BATCH_SIZE = 5000
    ids = [f"chunk_{i}" for i in range(len(chunks))]
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
st.title("🚀 RAG 시스템 v3")
st.markdown("**Hybrid Search** + **Reranking** + **Semantic Chunking**")

# 사이드바
with st.sidebar:
    st.header("⚙️ v3 설정")

    # 검색 업그레이드 옵션
    st.subheader("🔍 검색 설정")

    use_hybrid = st.toggle("Hybrid Search (BM25 + Dense)", value=True)
    if use_hybrid:
        dense_weight = st.slider("Dense 가중치", 0.0, 1.0, 0.7, 0.1)
        st.caption(f"BM25 가중치: {1-dense_weight:.1f}")

    use_reranking = st.toggle("Cross-Encoder Reranking", value=True)
    use_mmr = st.toggle("MMR (다양성 확보)", value=False)

    top_k = st.slider("검색 결과 수", 1, 10, DEFAULT_TOP_K)

    st.divider()

    # AI 답변 옵션
    st.subheader("🤖 AI 답변")
    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    use_ai_answer = st.checkbox(
        "Claude 답변 생성",
        value=has_api_key,
        disabled=not has_api_key
    )

    if not has_api_key:
        api_key = st.text_input("Anthropic API Key", type="password")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            st.success("✅ API 키 설정됨")
            st.rerun()

    st.divider()

    # PDF 업로드
    st.header("📄 PDF 업로드")
    uploaded_files = st.file_uploader(
        "PDF 파일 선택",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        # 청킹 전략 선택
        st.subheader("✂️ 청킹 전략")
        chunk_strategy = st.selectbox(
            "전략 선택",
            ["semantic", "adaptive", "hierarchical", "fixed"],
            index=0,
            help="semantic: 의미 단위 분할 (권장)\nadaptive: 컨텐츠 유형별 자동 선택\nhierarchical: 계층적 분할\nfixed: 고정 크기"
        )

        extract_tables = st.checkbox("📊 표 추출", value=True)
        extract_images = st.checkbox(
            "🖼️ 이미지 설명",
            value=has_api_key,
            disabled=not has_api_key
        )

        if st.button("📥 DB 구축", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                processor = EnhancedPDFProcessor()
                all_content = []

                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"처리 중: {uploaded_file.name}")
                    progress_bar.progress((i + 1) / (len(uploaded_files) + 3))

                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    content = processor.process_pdf(
                        temp_path,
                        extract_images=extract_images,
                        extract_tables=extract_tables
                    )
                    all_content.extend(content)
                    os.remove(temp_path)

                # v3 청킹
                status_text.text(f"{chunk_strategy} 청킹 중...")
                progress_bar.progress(0.6)
                model = load_model()

                chunks = content_to_chunks_v2(
                    all_content,
                    strategy=chunk_strategy,
                    embedding_model=model if chunk_strategy == "semantic" else None
                )

                # DB 구축
                status_text.text("벡터 DB 구축 중...")
                progress_bar.progress(0.8)
                collection = build_database_v3(chunks, model, chunk_strategy)
                progress_bar.progress(1.0)

                # 캐시 클리어
                st.cache_resource.clear()
                st.session_state.retriever = None

                total_tables = sum(len(p.get("tables", [])) for p in all_content)
                total_images = sum(len(p.get("images", [])) for p in all_content)

                status_text.empty()
                st.success(f"""✅ 완료!
                - {len(uploaded_files)}개 PDF
                - {len(chunks)}개 청크 ({chunk_strategy})
                - {total_tables}개 표
                - {total_images}개 이미지""")

            except Exception as e:
                st.error(f"❌ 오류: {e}")

    st.divider()

    # DB 상태
    st.header("💾 DB 상태")
    collection = load_collection()
    if collection:
        try:
            count = collection.count()
            meta = collection.metadata or {}
            version = meta.get("version", "v2")
            strategy = meta.get("chunk_strategy", "fixed")

            st.success(f"✅ 연결됨")
            st.caption(f"• {count:,}개 청크")
            st.caption(f"• 버전: {version}")
            st.caption(f"• 청킹: {strategy}")
        except Exception:
            st.warning("⚠️ DB 재구축 필요")
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

        # Enhanced Retriever 초기화 (필요 시)
        with st.spinner("검색 준비 중..."):
            retriever = init_enhanced_retriever(
                collection, model,
                use_hybrid, use_reranking, use_mmr
            )

        if not retriever:
            st.error("❌ Retriever 초기화 실패")
        else:
            # 검색 실행
            with st.spinner("🔍 검색 중..."):
                query_embedding = model.encode([query])[0]
                results = retriever.search(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    fetch_k=top_k * 4
                )

            # 검색 기록
            st.session_state.search_history.append({
                "query": query,
                "results_count": len(results),
                "hybrid": use_hybrid,
                "reranking": use_reranking
            })

            st.header("📋 검색 결과")

            # 검색 방법 표시
            methods = []
            if use_hybrid:
                methods.append("🔄 Hybrid")
            if use_reranking:
                methods.append("📊 Reranking")
            if use_mmr:
                methods.append("🎯 MMR")

            if methods:
                st.caption(f"적용된 기능: {' | '.join(methods)}")

            for i, r in enumerate(results):
                score = r.get("final_score", r.get("score", 0))
                meta = r.get("metadata", {})

                with st.container():
                    # 메타 정보
                    col_score, col_source, col_method = st.columns([1, 1, 1])

                    with col_score:
                        if score >= 0.5:
                            st.success(f"점수: {score:.3f}")
                        elif score >= 0.3:
                            st.warning(f"점수: {score:.3f}")
                        else:
                            st.error(f"점수: {score:.3f}")

                    with col_source:
                        st.info(f"📄 {meta.get('source', 'unknown')}")

                    with col_method:
                        search_methods = []
                        if r.get("in_dense"):
                            search_methods.append("Dense")
                        if r.get("in_sparse"):
                            search_methods.append("BM25")
                        if r.get("rerank_score"):
                            search_methods.append(f"Rerank")
                        st.info(f"🔧 {', '.join(search_methods) or 'Dense'}")

                    # 컨텐츠 타입
                    doc = r["document"]
                    content_type = "텍스트"
                    if "[표]" in doc:
                        content_type = "📊 표 포함"
                    elif "[이미지 설명]" in doc:
                        content_type = "🖼️ 이미지 설명"

                    st.markdown(f"**결과 {i+1}** - {content_type} | p.{meta.get('page', '?')}")
                    st.text_area(
                        f"내용_{i}",
                        doc[:600] + ("..." if len(doc) > 600 else ""),
                        height=120,
                        label_visibility="collapsed"
                    )
                    st.divider()

            # AI 답변
            if use_ai_answer and has_api_key:
                st.header("🤖 AI 답변")

                with st.spinner("Claude가 답변을 생성하는 중..."):
                    llm = get_llm_client()

                    # 컨텍스트 구성
                    context_parts = []
                    for i, r in enumerate(results):
                        context_parts.append(f"[문서 {i+1}]\n{r['document']}")
                    context = "\n\n---\n\n".join(context_parts)

                    try:
                        response = llm.client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=1024,
                            messages=[{
                                "role": "user",
                                "content": f"""다음 문서 내용을 바탕으로 질문에 답변해주세요.

## 문서 내용
{context}

## 질문
{query}

## 답변"""
                            }]
                        )

                        st.markdown(response.content[0].text)

                        # 출처
                        st.divider()
                        st.caption("📚 참고 문서:")
                        for r in results[:3]:
                            meta = r.get("metadata", {})
                            st.caption(f"  • {meta.get('source', 'unknown')} (p.{meta.get('page', '?')})")

                    except Exception as e:
                        st.error(f"❌ 오류: {e}")
            else:
                # LLM 프롬프트
                st.header("💡 LLM 프롬프트")
                context = "\n\n---\n\n".join([r["document"] for r in results])
                prompt = f"""다음 문서 내용을 바탕으로 질문에 답변해주세요.

[문서 내용]
{context}

[질문]
{query}

[답변]"""
                st.code(prompt, language=None)

with col2:
    st.header("📊 통계")

    collection = load_collection()
    if collection:
        try:
            count = collection.count()
            st.metric("총 청크 수", f"{count:,}")
            st.metric("검색 횟수", len(st.session_state.search_history))

            # 현재 설정
            st.subheader("⚙️ 현재 설정")
            st.caption(f"• Hybrid: {'✅' if use_hybrid else '❌'}")
            st.caption(f"• Reranking: {'✅' if use_reranking else '❌'}")
            st.caption(f"• MMR: {'✅' if use_mmr else '❌'}")

            if st.session_state.search_history:
                st.subheader("🕐 최근 검색")
                for item in reversed(st.session_state.search_history[-5:]):
                    methods = []
                    if item.get("hybrid"):
                        methods.append("H")
                    if item.get("reranking"):
                        methods.append("R")
                    method_str = f"[{','.join(methods)}]" if methods else ""
                    st.text(f"• {item['query'][:20]}... {method_str}")
        except Exception:
            st.info("DB를 재구축하세요")
    else:
        st.info("DB를 먼저 구축하세요")

# 푸터
st.divider()
st.caption("🚀 RAG 시스템 v3 | Hybrid Search + Reranking + Semantic Chunking")
