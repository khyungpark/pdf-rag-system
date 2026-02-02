"""
향상된 PDF 처리기
- 여러 PDF 파일 지원
- 표(Table) 추출
- 이미지 추출 및 Claude Vision으로 설명 생성
"""

import os
import base64
import fitz  # PyMuPDF
import pdfplumber
from anthropic import Anthropic

# ============================================================
# 설정
# ============================================================
IMAGE_MIN_SIZE = 100  # 최소 이미지 크기 (픽셀)
IMAGE_DESCRIPTION_PROMPT = """이 이미지를 분석하고 상세히 설명해주세요.
- 그래프/차트인 경우: 축, 데이터 포인트, 트렌드를 설명
- 다이어그램인 경우: 구성요소와 관계를 설명
- 사진인 경우: 보이는 내용을 설명
- 수식/공식인 경우: 수식을 텍스트로 변환

한국어로 답변해주세요. 간결하면서도 핵심 정보를 포함해주세요."""


class EnhancedPDFProcessor:
    """향상된 PDF 처리기"""

    def __init__(self, anthropic_api_key=None):
        """
        Args:
            anthropic_api_key: Anthropic API 키 (없으면 환경변수에서 읽음)
        """
        self.api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        if self.api_key:
            self.client = Anthropic(api_key=self.api_key)

    def process_pdf(self, pdf_path, extract_images=True, extract_tables=True):
        """
        PDF 파일을 처리하여 텍스트, 표, 이미지 설명을 추출합니다.

        Args:
            pdf_path: PDF 파일 경로
            extract_images: 이미지 추출 및 설명 생성 여부
            extract_tables: 표 추출 여부

        Returns:
            페이지별 컨텐츠 리스트
        """
        print(f"\n📄 PDF 처리 중: {os.path.basename(pdf_path)}")

        pages_content = []

        # PyMuPDF로 텍스트와 이미지 추출
        doc = fitz.open(pdf_path)

        # pdfplumber로 표 추출
        plumber_pdf = pdfplumber.open(pdf_path) if extract_tables else None

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_data = {
                "page": page_num + 1,
                "source": os.path.basename(pdf_path),
                "text": "",
                "tables": [],
                "images": []
            }

            # 1. 텍스트 추출
            text = page.get_text().strip()
            if text:
                page_data["text"] = text

            # 2. 표 추출 (pdfplumber)
            if extract_tables and plumber_pdf:
                try:
                    plumber_page = plumber_pdf.pages[page_num]
                    tables = plumber_page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:  # 헤더 + 데이터
                            table_md = self._table_to_markdown(table)
                            if table_md:
                                page_data["tables"].append({
                                    "table_id": table_idx,
                                    "content": table_md
                                })
                except Exception as e:
                    print(f"   ⚠️ 표 추출 오류 (p.{page_num+1}): {e}")

            # 3. 이미지 추출 및 설명 생성
            if extract_images and self.client:
                try:
                    images = page.get_images(full=True)
                    for img_idx, img_info in enumerate(images):
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)

                        if base_image:
                            image_bytes = base_image["image"]
                            width = base_image.get("width", 0)
                            height = base_image.get("height", 0)

                            # 너무 작은 이미지는 건너뛰기
                            if width < IMAGE_MIN_SIZE or height < IMAGE_MIN_SIZE:
                                continue

                            # Claude Vision으로 이미지 설명 생성
                            description = self._describe_image(
                                image_bytes,
                                base_image.get("ext", "png")
                            )

                            if description:
                                page_data["images"].append({
                                    "image_id": img_idx,
                                    "size": f"{width}x{height}",
                                    "description": description
                                })
                except Exception as e:
                    print(f"   ⚠️ 이미지 처리 오류 (p.{page_num+1}): {e}")

            # 컨텐츠가 있는 페이지만 추가
            if page_data["text"] or page_data["tables"] or page_data["images"]:
                pages_content.append(page_data)

            # 진행 상황 표시
            if (page_num + 1) % 10 == 0:
                print(f"   📖 {page_num + 1}/{len(doc)} 페이지 처리 완료")

        doc.close()
        if plumber_pdf:
            plumber_pdf.close()

        # 통계 출력
        total_tables = sum(len(p["tables"]) for p in pages_content)
        total_images = sum(len(p["images"]) for p in pages_content)
        print(f"   ✅ 완료: {len(pages_content)}페이지, {total_tables}개 표, {total_images}개 이미지")

        return pages_content

    def process_folder(self, folder_path, **kwargs):
        """
        폴더 내 모든 PDF 파일을 처리합니다.

        Args:
            folder_path: PDF 파일이 있는 폴더 경로
            **kwargs: process_pdf에 전달할 추가 인자

        Returns:
            모든 PDF의 페이지 컨텐츠 리스트
        """
        all_content = []
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

        if not pdf_files:
            print(f"❌ PDF 파일이 없습니다: {folder_path}")
            return all_content

        print(f"\n📁 폴더 처리: {folder_path}")
        print(f"   📚 {len(pdf_files)}개 PDF 파일 발견")

        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            try:
                content = self.process_pdf(pdf_path, **kwargs)
                all_content.extend(content)
            except Exception as e:
                print(f"   ❌ 오류 ({pdf_file}): {e}")

        return all_content

    def _table_to_markdown(self, table):
        """표를 마크다운 형식으로 변환"""
        if not table or len(table) < 1:
            return ""

        # None 값 처리
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell) if cell else "" for cell in row]
            cleaned_table.append(cleaned_row)

        if not cleaned_table:
            return ""

        # 마크다운 테이블 생성
        lines = []

        # 헤더
        header = cleaned_table[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        # 데이터 행
        for row in cleaned_table[1:]:
            # 열 개수 맞추기
            while len(row) < len(header):
                row.append("")
            lines.append("| " + " | ".join(row[:len(header)]) + " |")

        return "\n".join(lines)

    def _describe_image(self, image_bytes, ext="png"):
        """Claude Vision API로 이미지 설명 생성"""
        if not self.client:
            return None

        try:
            # 이미지를 base64로 인코딩
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # MIME 타입 결정
            mime_types = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "webp": "image/webp"
            }
            media_type = mime_types.get(ext.lower(), "image/png")

            # Claude API 호출
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": IMAGE_DESCRIPTION_PROMPT
                            }
                        ]
                    }
                ]
            )

            return message.content[0].text

        except Exception as e:
            print(f"      ⚠️ 이미지 설명 생성 오류: {e}")
            return None


def content_to_chunks(pages_content, chunk_size=500, overlap=100):
    """
    페이지 컨텐츠를 청크로 변환합니다.

    Args:
        pages_content: process_pdf/process_folder의 결과
        chunk_size: 청크 크기
        overlap: 청크 간 겹침

    Returns:
        청크 리스트
    """
    import re
    chunks = []
    sentence_endings = re.compile(r'[.!?]\s+|[.!?]$|\n\n')

    for page_data in pages_content:
        page_num = page_data["page"]
        source = page_data.get("source", "unknown")

        # 모든 컨텐츠를 하나의 텍스트로 합침
        combined_text = []

        # 텍스트
        if page_data["text"]:
            combined_text.append(page_data["text"])

        # 표
        for table in page_data.get("tables", []):
            combined_text.append(f"\n[표]\n{table['content']}\n")

        # 이미지 설명
        for img in page_data.get("images", []):
            combined_text.append(f"\n[이미지 설명]\n{img['description']}\n")

        full_text = "\n\n".join(combined_text)

        if not full_text.strip():
            continue

        # 청크 분할
        if len(full_text) <= chunk_size:
            chunks.append({
                "text": full_text.strip(),
                "page": page_num,
                "source": source,
                "chunk_id": len(chunks)
            })
        else:
            start = 0
            while start < len(full_text):
                end = min(start + chunk_size, len(full_text))

                # 문장 경계 찾기
                if end < len(full_text):
                    match = sentence_endings.search(full_text, end)
                    if match and match.end() - end < 100:
                        end = match.end()

                chunk_text = full_text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num,
                        "source": source,
                        "chunk_id": len(chunks)
                    })

                start = max(end - overlap, start + 1)

    return chunks


# ============================================================
# 테스트 코드
# ============================================================
if __name__ == "__main__":
    import sys

    # API 키 확인
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️ ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   이미지 설명 기능은 비활성화됩니다.")
        print()

    # 테스트 실행
    processor = EnhancedPDFProcessor()

    # 현재 폴더의 PDF 처리
    pdf_files = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("❌ 현재 폴더에 PDF 파일이 없습니다.")
        sys.exit(1)

    print(f"📚 {len(pdf_files)}개 PDF 파일 발견")

    all_content = []
    for pdf_file in pdf_files:
        # API 키가 없으면 이미지 추출 건너뛰기
        content = processor.process_pdf(
            pdf_file,
            extract_images=bool(processor.client),
            extract_tables=True
        )
        all_content.extend(content)

    # 청크 생성
    chunks = content_to_chunks(all_content)

    print(f"\n📊 결과 요약")
    print(f"   총 페이지: {len(all_content)}")
    print(f"   총 청크: {len(chunks)}")

    # 샘플 출력
    if chunks:
        print(f"\n📝 샘플 청크 (첫 번째):")
        print(f"   출처: {chunks[0]['source']}, 페이지: {chunks[0]['page']}")
        print(f"   내용: {chunks[0]['text'][:200]}...")
