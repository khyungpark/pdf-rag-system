#!/usr/bin/env python3
"""
RAG 데이터베이스 자동 구축 스크립트

사용법:
    1. pdfs/ 폴더에 PDF 파일들을 넣으세요
    2. python build_db.py 실행
    3. 완료!

옵션:
    python build_db.py                    # pdfs/ 폴더의 모든 PDF 처리
    python build_db.py ./my_folder        # 특정 폴더의 PDF 처리
    python build_db.py --with-images      # 이미지 설명 포함 (API 키 필요)
"""

import os
import sys
import argparse
import tarfile
import shutil
from datetime import datetime

# ============================================================
# 설정
# ============================================================
DEFAULT_PDF_FOLDER = "./pdfs"
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 5000
BACKUP_DIR = "./backups"
MAX_BACKUPS = 5  # 최대 백업 유지 개수


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

    # 백업 디렉토리 생성
    os.makedirs(backup_dir, exist_ok=True)

    # 백업 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"chroma_db_backup_{timestamp}.tar.gz"
    backup_path = os.path.join(backup_dir, backup_filename)

    try:
        print(f"   💾 DB 백업 생성 중...")
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(db_path, arcname=os.path.basename(db_path))

        backup_size = os.path.getsize(backup_path) / (1024 * 1024)  # MB
        print(f"   ✅ 백업 완료: {backup_filename} ({backup_size:.1f}MB)")

        # 오래된 백업 정리
        cleanup_old_backups(backup_dir)

        return backup_path
    except Exception as e:
        print(f"   ⚠️ 백업 실패: {e}")
        return None


def cleanup_old_backups(backup_dir: str = BACKUP_DIR, max_backups: int = MAX_BACKUPS):
    """오래된 백업 파일 정리 (최신 N개만 유지)"""
    if not os.path.exists(backup_dir):
        return

    # 백업 파일 목록 (타임스탬프 기준 정렬)
    backups = sorted([
        f for f in os.listdir(backup_dir)
        if f.startswith("chroma_db_backup_") and f.endswith(".tar.gz")
    ], reverse=True)

    # 오래된 백업 삭제
    for old_backup in backups[max_backups:]:
        old_path = os.path.join(backup_dir, old_backup)
        try:
            os.remove(old_path)
            print(f"   🗑️ 오래된 백업 삭제: {old_backup}")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="RAG 데이터베이스 자동 구축")
    parser.add_argument("folder", nargs="?", default=DEFAULT_PDF_FOLDER,
                        help=f"PDF 폴더 경로 (기본값: {DEFAULT_PDF_FOLDER})")
    parser.add_argument("--with-images", action="store_true",
                        help="이미지 설명 생성 (ANTHROPIC_API_KEY 필요)")
    parser.add_argument("--no-tables", action="store_true",
                        help="표 추출 비활성화")
    parser.add_argument("--append", action="store_true",
                        help="기존 DB에 추가 (삭제하지 않음)")
    args = parser.parse_args()

    # 폴더 확인
    if not os.path.exists(args.folder):
        print(f"❌ 폴더가 없습니다: {args.folder}")
        print(f"   폴더를 생성하고 PDF 파일을 넣어주세요.")
        os.makedirs(args.folder, exist_ok=True)
        print(f"   ✅ 폴더 생성됨: {args.folder}")
        return

    # PDF 파일 목록
    pdf_files = [f for f in os.listdir(args.folder) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"❌ PDF 파일이 없습니다: {args.folder}")
        print(f"   PDF 파일을 폴더에 넣고 다시 실행하세요.")
        return

    print()
    print("🚀" + "="*58)
    print("   RAG 데이터베이스 자동 구축")
    print("="*60)
    print(f"📁 폴더: {os.path.abspath(args.folder)}")
    print(f"📚 PDF 파일: {len(pdf_files)}개")
    print(f"📊 표 추출: {'✅' if not args.no_tables else '❌'}")
    print(f"🖼️ 이미지 설명: {'✅' if args.with_images else '❌'}")
    print(f"📦 모드: {'추가' if args.append else '새로 구축'}")
    print("="*60)

    # 라이브러리 로드
    print("\n📦 라이브러리 로드 중...")
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        from pdf_processor import EnhancedPDFProcessor, content_to_chunks
    except ImportError as e:
        print(f"❌ 필요한 패키지가 없습니다: {e}")
        print("   pip install -r requirements.txt 를 실행하세요.")
        return

    # API 키 확인
    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    if args.with_images and not has_api_key:
        print("⚠️ ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        print("   이미지 설명 기능이 비활성화됩니다.")
        args.with_images = False

    # PDF 처리
    print("\n" + "="*60)
    print("📄 PDF 처리 시작")
    print("="*60)

    processor = EnhancedPDFProcessor()
    all_content = []
    failed_files = []

    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(args.folder, pdf_file)
        print(f"\n[{i}/{len(pdf_files)}] {pdf_file}")

        try:
            content = processor.process_pdf(
                pdf_path,
                extract_images=args.with_images,
                extract_tables=not args.no_tables
            )
            all_content.extend(content)
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            failed_files.append(pdf_file)

    if not all_content:
        print("\n❌ 처리된 컨텐츠가 없습니다.")
        return

    # 청크 생성
    print("\n" + "="*60)
    print("✂️ 청크 분할")
    print("="*60)

    chunks = content_to_chunks(all_content, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"   ✅ {len(chunks):,}개 청크 생성")

    # 임베딩
    print("\n" + "="*60)
    print("🤖 임베딩 생성")
    print("="*60)

    print(f"   모델: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    print(f"   {len(texts):,}개 텍스트 벡터 변환 중...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"   ✅ 벡터 변환 완료 (차원: {embeddings.shape[1]})")

    # DB 백업 (append 모드가 아닐 때만)
    if not args.append:
        print("\n" + "="*60)
        print("📦 기존 DB 백업")
        print("="*60)
        backup_database()

    # DB 저장
    print("\n" + "="*60)
    print("💾 데이터베이스 저장")
    print("="*60)

    client = chromadb.PersistentClient(path=DB_PATH)

    if not args.append:
        try:
            client.delete_collection("rag_documents")
            print("   🗑️ 기존 컬렉션 삭제")
        except Exception:
            pass

    try:
        collection = client.get_collection("rag_documents")
        existing_count = collection.count()
        print(f"   📂 기존 컬렉션 사용 ({existing_count:,}개 청크)")
    except Exception:
        collection = client.create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        print("   📂 새 컬렉션 생성")

    # 배치 저장
    ids = [f"chunk_{c['chunk_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}" for c in chunks]
    metadatas = [{"page": c["page"], "source": c.get("source", "unknown")} for c in chunks]
    embeddings_list = embeddings.tolist()

    for i in range(0, len(chunks), BATCH_SIZE):
        end = min(i + BATCH_SIZE, len(chunks))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings_list[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"   📦 배치 {i//BATCH_SIZE + 1}: {end - i:,}개 저장")

    # 완료 보고
    print("\n" + "="*60)
    print("✅ 완료!")
    print("="*60)

    final_count = collection.count()
    print(f"   📚 처리된 PDF: {len(pdf_files) - len(failed_files)}개")
    print(f"   📄 총 청크: {final_count:,}개")
    print(f"   📁 DB 위치: {os.path.abspath(DB_PATH)}")

    if failed_files:
        print(f"\n   ⚠️ 실패한 파일: {len(failed_files)}개")
        for f in failed_files:
            print(f"      - {f}")

    print("\n💡 대시보드에서 검색: http://localhost:8501")
    print()


if __name__ == "__main__":
    main()
