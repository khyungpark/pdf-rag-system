# PDF RAG 시스템 - 개발 로그

Claude Code와 함께 진행한 개발 작업 기록입니다.

---

## 2026-02-02 (Day 1)

### 1. RAG 시스템 플랜 검토 및 개선 요청

```
지금 파일은 클로드에서 PDF 파일을 RAG 데이터베이스 구축에 관한 플랜인데
이것을 검토하고 실행에 대하여 평가해 줘
```

**Claude 작업:**
- 기존 `rag_system.py`, `requirements.txt`, `README.md` 검토
- 4가지 개선점 도출:
  - 불필요한 langchain 의존성 제거
  - 청크 분할을 문장 경계 기반으로 개선
  - L2 거리 → 코사인 유사도로 변경
  - bare except → 명시적 Exception 처리

---

### 2. 개선점 적용 및 실행

```
개선점 적용하고 실행해줘
```

**Claude 작업:**
- `requirements.txt` - langchain 제거, 필수 패키지만 유지
- `rag_system.py` - 문장 경계 기반 청크 분할, 코사인 유사도 적용
- ChromaDB NotFoundError 발생 → exception 타입 수정으로 해결

---

### 3. 웹 대시보드 생성

```
지금 대쉬보드를 만들어 줄수 있나
```

**Claude 작업:**
- `dashboard.py` 생성 - Streamlit 기반 웹 UI
- 기능: 질문 입력, PDF 업로드, 검색 결과 시각화, LLM 프롬프트 생성
- 포트 충돌 해결 (기존 프로세스 종료)
- http://localhost:8501 에서 실행

---

### 4. PDF 표/이미지 추출 기능 추가

```
PDF파일을 더 RAG 데이터베이스에 추가하고 싶고
특히 표 그림이 포함된 pdf파일을 데이터베이스에 추가할 수 있어?
```

**Claude 작업:**
- 표 추출 방식 선택: pdfplumber (마크다운 변환)
- 이미지 설명 방식 선택: Claude Vision API
- `pdf_processor.py` 생성 - EnhancedPDFProcessor 클래스
- `rag_system_v2.py` 생성 - CLI v2
- `dashboard_v2.py` 생성 - 웹 대시보드 v2

---

### 5. Anthropic API 키 설정

```
claude api key 받았는데 설정해 주세요
```

**Claude 작업:**
- `.env` 파일 생성 (API 키 저장)
- `.gitignore` 추가 (보안)
- API 연결 테스트 성공

---

### 6. 배치 크기 오류 해결

```
오류: ValueError: Batch size of 6405 is greater than max batch size of 5461
```

**Claude 작업:**
- ChromaDB 배치 크기 제한 확인 (max 5461)
- BATCH_SIZE = 5000으로 배치 처리 로직 추가
- `rag_system_v2.py`, `dashboard_v2.py` 모두 수정

---

### 7. 자동 DB 구축 스크립트 생성

```
DB에 저장할 수 있는 최대 파일은 몇개까지 가능하며,
내가 하나하나 업로드하는게 초기에는 시간이 너무 많이 걸리니
특정한 폴더에 DB에 올릴 파일을 넣으면 자동으로 해줄수 있나요?
```

**Claude 작업:**
- `pdfs/` 폴더 생성
- `build_db.py` 생성 - 자동 배치 처리 스크립트
- 옵션: `--with-images`, `--append`, `--no-tables`
- 파일 수 제한 없음 (디스크/메모리만큼)

---

### 8. DB 구축 및 확장 (8개 → 19개 PDF)

```
지금 PDFS 폴더에 7개의 파일을 옮겼는데 DB 생성해 주세요
```

**Claude 작업:**
- 첫 구축: 8개 PDF → 26,331개 청크
- 확장: 19개 PDF → 79,820개 청크
- 약 400개 표 추출

---

### 9. LLM 답변 생성 기능 추가

```
현재까지 RAG 시스템 구축에서 자연어로 관련 질문을 하면 RAG DB에서
유사한 내용 찾아서 다시 llm에 질문할 프롬프트를 만들어 주는거 까지인데
LLM에 연결해서 답변까지 받을려면 어떻게 해야 될까요?
```

**Claude 작업:**
- `llm_client.py` 생성 - RAGAnswerGenerator 클래스
- `dashboard_v2.py` - "🤖 AI 답변 생성" 체크박스 추가
- `rag_system_v2.py` - 대화형 모드에 자동 답변 생성

---

### 10. 프롬프트 개선 (추론 기반 답변)

```
프롬프트를 수정해서 질문이 DB 검색 한계로 구체적인 답변을 하지 못할때
ai가 추론해서 최선의 답변을 도출하고 맨 마지막에
전문가나 식약처 상담이 필요하다는 문구를 넣어서 프롬프트를 다시 만들수 있을까요?
```

**Claude 작업:**
- `llm_client.py` 프롬프트 대폭 개선:
  - 의약품/바이오의약품 규제과학 전문가 역할 부여
  - 문서 기반 답변 + 전문 지식 추론 보완
  - "답변 드릴 수 없습니다" 금지
  - 마지막에 전문가 상담 권고 문구 자동 추가

---

### 11. DB 백업 및 GitHub 업로드

```
현재까지 구축된 DB 백업파일 만들어 주세요
```

**Claude 작업:**
- `chroma_db_backup_20260202_230314.tar.gz` 생성 (95MB)

```
현재까지 생성된 작업 파일 github에 새로운 레포지토리 만들어서 저장해 주세요
```

**Claude 작업:**
- `.gitignore` 업데이트 (PDF, 백업 파일 제외)
- GitHub 레포지토리 생성: https://github.com/khyungpark/pdf-rag-system
- Initial commit 및 v3.0 업데이트 푸시

---

### 12. 보고서 v3.0 업데이트

```
지금까지 작업내용을 업데이트해서 다시 RAG 시스템 구축 보고서를 만들어 주세요
```

**Claude 작업:**
- `docs/RAG_시스템_구축_보고서.md` v3.0으로 업데이트
- LLM 답변 생성, 자동 DB 구축, GitHub 연동 내용 추가
- 19개 PDF, 79,820개 청크 현황 반영

---

## 커밋 히스토리

| 날짜 | 커밋 | 설명 |
|------|------|------|
| 02/02 | `0367226` | Initial commit: PDF RAG System v2 |
| 02/02 | `1cef8ea` | Update to v3.0: LLM answer generation and improved prompts |

---

## 기술 스택

- **PDF 처리**: PyMuPDF, pdfplumber
- **벡터 DB**: ChromaDB
- **임베딩**: Sentence-Transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **LLM**: Claude Sonnet 4 (Anthropic API)
- **웹 UI**: Streamlit
- **버전 관리**: GitHub

---

## 주요 기능

1. **PDF 텍스트/표/이미지 추출**
   - PyMuPDF로 텍스트 및 이미지 추출
   - pdfplumber로 표를 마크다운으로 변환
   - Claude Vision으로 이미지 설명 생성 (선택)

2. **벡터 검색**
   - 문장 경계 기반 청크 분할
   - 코사인 유사도 기반 검색
   - 79,820개 청크 저장

3. **LLM 답변 생성**
   - 검색 결과 기반 Claude 자동 답변
   - 규제과학 전문가 수준 프롬프트
   - 전문가 상담 권고 자동 추가

4. **자동 DB 구축**
   - 폴더 기반 일괄 처리
   - 증분 업데이트 지원 (--append)

5. **웹 대시보드**
   - Streamlit 기반 UI
   - PDF 업로드, 검색, AI 답변 통합
