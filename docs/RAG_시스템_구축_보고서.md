# RAG 시스템 구축 보고서

**프로젝트명**: PDF 문서 RAG (Retrieval-Augmented Generation) 시스템
**작성일**: 2026-02-02
**버전**: v2.0

---

## 1. 프로젝트 개요

### 1.1 목적
PDF 문서를 AI가 검색할 수 있는 벡터 데이터베이스로 구축하여, 자연어 질문에 대해 관련 문서 내용을 검색하는 RAG 시스템 개발

### 1.2 주요 기능
| 기능 | 설명 | 구현 상태 |
|------|------|----------|
| PDF 텍스트 추출 | PyMuPDF를 이용한 텍스트 추출 | ✅ 완료 |
| 표(Table) 추출 | pdfplumber를 이용한 표 마크다운 변환 | ✅ 완료 |
| 이미지 설명 생성 | Claude Vision API로 이미지 내용 설명 | ✅ 완료 |
| 벡터 검색 | ChromaDB + Sentence-Transformers | ✅ 완료 |
| 웹 대시보드 | Streamlit 기반 UI | ✅ 완료 |
| 여러 PDF 지원 | 폴더 내 다중 PDF 처리 | ✅ 완료 |

---

## 2. 작업 진행 과정

### 2.1 Phase 1: 초기 플랜 검토 및 개선 (v1.0)

**시작 상태**: 기본 RAG 시스템 코드 존재 (`rag_system.py`)

**수행한 개선 작업**:

| 항목 | 이전 | 이후 |
|------|------|------|
| 의존성 | langchain 포함 (미사용) | 필수 패키지만 유지 |
| 청크 분할 | 글자 수 기준 단순 분할 | 문장 경계 기반 분할 |
| 유사도 계산 | L2 거리 기반 | 코사인 유사도 |
| 예외 처리 | bare except | 명시적 Exception 처리 |

**코드 변경 내역**:
```python
# 청크 분할 개선 (rag_system.py:117-190)
def split_into_chunks(pages, chunk_size, overlap):
    # 문장 경계(. ! ?)에서 분할하여 문맥 유지
    sentence_endings = re.compile(r'[.!?]\s+|[.!?]$|\n\n')
    ...

# 코사인 유사도 설정 (rag_system.py:220-222)
collection = client.create_collection(
    metadata={"hnsw:space": "cosine"}
)
```

---

### 2.2 Phase 2: 웹 대시보드 구축

**생성 파일**: `dashboard.py`

**주요 기능**:
- 질문 입력 및 검색
- PDF 업로드 및 DB 구축
- 검색 결과 시각화 (유사도 점수, 페이지 정보)
- LLM 프롬프트 자동 생성

**기술 스택**:
- Streamlit 1.53.1
- 세션 상태 관리 (`st.session_state`)
- 리소스 캐싱 (`@st.cache_resource`)

---

### 2.3 Phase 3: 향상된 PDF 처리기 개발 (v2.0)

**사용자 요구사항**:
1. 여러 PDF 파일 지원
2. 표(Table) 추출
3. 이미지/그림 처리

**구현 방식 선택**:
- 표 추출: pdfplumber
- 이미지 설명: Claude Vision API (Anthropic)

**생성 파일**:

| 파일 | 역할 |
|------|------|
| `pdf_processor.py` | 향상된 PDF 처리기 (표+이미지) |
| `rag_system_v2.py` | CLI 버전 v2 |
| `dashboard_v2.py` | 웹 대시보드 v2 |

**핵심 클래스**: `EnhancedPDFProcessor`
```python
class EnhancedPDFProcessor:
    def process_pdf(self, pdf_path, extract_images=True, extract_tables=True):
        # 1. 텍스트 추출 (PyMuPDF)
        # 2. 표 추출 (pdfplumber) → 마크다운 변환
        # 3. 이미지 추출 → Claude Vision API로 설명 생성
        ...

    def process_folder(self, folder_path):
        # 폴더 내 모든 PDF 일괄 처리
        ...
```

---

### 2.4 Phase 4: API 연동 및 설정

**Anthropic API 설정**:
- `.env` 파일 생성 (API 키 저장)
- `.gitignore` 추가 (보안)

**연결 테스트 결과**: ✅ 성공
```
✅ API 연결 성공!
응답: 안녕하세요, 연결이 정상적으로 작동하고 있습니다.
```

---

## 3. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG 시스템 v2                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   PDF 파일   │───▶│ PDF 처리기  │───▶│   청크 분할  │     │
│  │  (다중 지원) │    │ (표+이미지) │    │ (문장 경계) │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                            │                   │            │
│                            ▼                   ▼            │
│                    ┌─────────────┐    ┌─────────────┐       │
│                    │Claude Vision│    │  Embedding  │       │
│                    │  (이미지)   │    │   Model     │       │
│                    └─────────────┘    └─────────────┘       │
│                            │                   │            │
│                            └─────────┬─────────┘            │
│                                      ▼                      │
│                            ┌─────────────────┐              │
│                            │   ChromaDB      │              │
│                            │ (벡터 데이터베이스)│              │
│                            └─────────────────┘              │
│                                      │                      │
│                     ┌────────────────┼────────────────┐     │
│                     ▼                ▼                ▼     │
│              ┌───────────┐   ┌───────────┐   ┌───────────┐  │
│              │    CLI    │   │ Dashboard │   │   API     │  │
│              │  (터미널)  │   │   (웹)    │   │  (확장)   │  │
│              └───────────┘   └───────────┘   └───────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 테스트 결과

### 4.1 PDF 처리 테스트

**테스트 파일**: `14__임상시험의_개요_주완석.pdf` (123페이지)

| 항목 | 결과 |
|------|------|
| 페이지 추출 | 123페이지 |
| 청크 생성 | 4,374개 |
| 표 추출 | 143개 |
| 처리 시간 | ~2분 |

### 4.2 검색 테스트

**질문**: "임상시험 1상이란?"

| 순위 | 유사도 | 페이지 |
|------|--------|--------|
| 1 | 0.78 | 62 |
| 2 | 0.78 | 62 |
| 3 | 0.78 | 32 |

---

## 5. 파일 구조

```
rag_project/
├── rag_system.py        # CLI 버전 v1 (텍스트만)
├── rag_system_v2.py     # CLI 버전 v2 (표+이미지)
├── pdf_processor.py     # 향상된 PDF 처리기
├── dashboard.py         # 웹 대시보드 v1
├── dashboard_v2.py      # 웹 대시보드 v2
├── requirements.txt     # 패키지 목록
├── .env                 # API 키 (Git 제외)
├── .gitignore           # Git 제외 파일
├── README.md            # 사용 가이드
├── chroma_db/           # 벡터 데이터베이스
├── venv/                # Python 가상환경
└── docs/
    └── RAG_시스템_구축_보고서.md  # 이 보고서
```

---

## 6. 사용된 기술 스택

| 분류 | 기술 | 버전 | 용도 |
|------|------|------|------|
| PDF 처리 | PyMuPDF | ≥1.24.0 | 텍스트/이미지 추출 |
| PDF 처리 | pdfplumber | ≥0.10.0 | 표 추출 |
| 벡터 DB | ChromaDB | ≥0.5.0 | 벡터 저장/검색 |
| 임베딩 | Sentence-Transformers | ≥3.0.0 | 텍스트 벡터화 |
| 웹 UI | Streamlit | ≥1.30.0 | 대시보드 |
| AI API | Anthropic | ≥0.30.0 | 이미지 설명 |

**임베딩 모델**: `paraphrase-multilingual-MiniLM-L12-v2`
- 다국어 지원 (한국어 포함)
- 벡터 차원: 384

---

## 7. 실행 방법

### 7.1 웹 대시보드
```bash
cd /Users/parkkyu-hyung/Documents/rag_project
source venv/bin/activate
export $(cat .env | xargs)
streamlit run dashboard_v2.py
```
→ http://localhost:8501

### 7.2 CLI 버전
```bash
source venv/bin/activate
export $(cat .env | xargs)
python rag_system_v2.py           # 현재 폴더 PDF
python rag_system_v2.py ./docs    # 특정 폴더
```

---

## 8. 향후 개선 사항

| 우선순위 | 기능 | 설명 |
|----------|------|------|
| 높음 | LLM 통합 | 검색 결과를 Claude/GPT에 자동 전달하여 답변 생성 |
| 중간 | 증분 업데이트 | 기존 DB에 새 PDF 추가 (전체 재구축 없이) |
| 중간 | 메타데이터 필터 | 특정 문서/페이지 범위 검색 |
| 낮음 | 하이브리드 검색 | 벡터 검색 + 키워드 검색 결합 |

---

## 9. 결론

PDF 문서를 벡터 데이터베이스로 변환하여 자연어 검색이 가능한 RAG 시스템을 성공적으로 구축했습니다. 특히 표와 이미지까지 처리할 수 있는 향상된 버전(v2)을 개발하여, 다양한 형태의 PDF 문서에서 정보를 추출할 수 있게 되었습니다.

---

**작성**: Claude Code (bkit)
**검토 필요**: 실제 운영 환경 배포 전 성능 테스트 권장
