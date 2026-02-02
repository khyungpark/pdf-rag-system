"""
LLM 클라이언트 - RAG 검색 결과를 바탕으로 답변 생성
"""

import os
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class RAGAnswerGenerator:
    """RAG 검색 결과를 바탕으로 LLM 답변을 생성하는 클래스"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None

        if HAS_ANTHROPIC and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)

    def is_available(self) -> bool:
        """LLM 사용 가능 여부"""
        return self.client is not None

    def generate_answer(
        self,
        query: str,
        search_results: dict,
        max_tokens: int = 2000,
        temperature: float = 0.3
    ) -> dict:
        """
        검색 결과를 바탕으로 답변 생성

        Args:
            query: 사용자 질문
            search_results: ChromaDB 검색 결과
            max_tokens: 최대 토큰 수
            temperature: 창의성 조절 (0: 정확, 1: 창의적)

        Returns:
            dict: {"answer": str, "sources": list, "error": str or None}
        """
        if not self.is_available():
            return {
                "answer": None,
                "sources": [],
                "error": "API 키가 설정되지 않았습니다. ANTHROPIC_API_KEY를 설정하세요."
            }

        # 검색 결과에서 문서와 메타데이터 추출
        documents = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]
        distances = search_results.get("distances", [[]])[0]

        if not documents:
            return {
                "answer": "검색 결과가 없습니다.",
                "sources": [],
                "error": None
            }

        # 컨텍스트 구성
        context_parts = []
        sources = []

        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            similarity = round(1 - dist, 2)
            source_info = {
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", 0),
                "similarity": similarity
            }
            sources.append(source_info)

            context_parts.append(f"[문서 {i+1}] (출처: {source_info['source']}, 페이지: {source_info['page']}, 유사도: {similarity})\n{doc}")

        context = "\n\n---\n\n".join(context_parts)

        # 프롬프트 구성
        system_prompt = """당신은 문서 기반 질의응답 전문가입니다.
주어진 문서 내용만을 바탕으로 정확하고 상세하게 답변해주세요.

규칙:
1. 반드시 제공된 문서 내용을 근거로 답변하세요
2. 문서에 없는 내용은 추측하지 마세요
3. 답변 시 출처(문서 번호, 페이지)를 언급해주세요
4. 표 데이터가 있으면 적절히 활용하세요
5. 한국어로 답변하세요"""

        user_prompt = f"""다음 문서 내용을 바탕으로 질문에 답변해주세요.

[검색된 문서 내용]
{context}

[질문]
{query}

[답변]"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            answer = response.content[0].text

            return {
                "answer": answer,
                "sources": sources,
                "error": None
            }

        except anthropic.APIError as e:
            return {
                "answer": None,
                "sources": sources,
                "error": f"API 오류: {str(e)}"
            }
        except Exception as e:
            return {
                "answer": None,
                "sources": sources,
                "error": f"오류 발생: {str(e)}"
            }


def get_llm_client(api_key: Optional[str] = None) -> RAGAnswerGenerator:
    """LLM 클라이언트 인스턴스 반환"""
    return RAGAnswerGenerator(api_key)
