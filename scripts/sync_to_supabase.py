import os
import re
import argparse
import frontmatter
from supabase import create_client
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. 환경 변수 및 설정
URL = os.environ.get("SUPABASE_URL")
KEY = os.environ.get("SUPABASE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# supabase 클라이언트 및 임베딩 모델 초기화
supabase = create_client(URL, KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

def generate_blog_url(filename, categories):
    """Minimal Mistakes 테마 규칙에 따른 URL 생성"""

    # 카테고리 경로 생성 (예: llm/rag/)

    dir_name = "".join([f"{c.lower()}/"for c in categories])

    # 파일명에서 날짜 제거 (YYYY-MM-DD-)
    url_name = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", filename)
    slug = url_name.replace('.md', '')
    
    return f"https://icechickentender.github.io/{dir_name}{slug}/"

def process_sync(added_modified_files, deleted_files):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    # 삭제된 파일 처리
    for file_path in deleted_files:
        if not file_path.endswith(".md"): continue

        # 삭제 시에는 카테고리 정보를 알기 어렵기 때문에,
        # 파일명 기반의 slug가 포함된 URL 패턴으로 DB에서 검색하여 삭제하는 것이 안전
        filename = os.path.basename(file_path)
        url_slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", filename).replace('.md', '')

        # 해당 slug가 포함된 모든 URL 데이터 삭제
        supabase.table("documents").delete().filter("metadata->>url", "ilike", f"%{url_slug}%").execute()
        print(f"데이터 삭제 완료(파일 제거됨): {file_path}")
    
    all_docs = []
    for file_path in added_modified_files:
        if not file_path.endswith(".md") or not os.path.exists(file_path):
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)

            # 1. URL 생성
            filename = os.path.basename(file_path)
            categories = post.get("categories", [])
            blog_url = generate_blog_url(filename, categories)

            # 2. 중복 방지: 기존에 저장된 동일 URL 데이터 먼저 삭제 (Upsert 효과)
            supabase.table("documents").delete().filter("metadata->>url", "eq", blog_url).execute()

            # 3. 메타데이터 구성
            metadata = {
                "title": post.get("title", "Untitled"),
                "category": categories,
                "tag": post.get("tags", []),
                "url": blog_url
            }

            # 4. 청킹 및 도큐먼트 객체 생성
            chunks = text_splitter.split_text(post.content)
            for chunk in chunks:
                all_docs.append(Document(page_content=chunk, metadata=metadata))
            print(f"갱신 준비 (추가/수정): {blog_url}")
    
    if all_docs:
        SupabaseVectorStore.from_documents(
            all_docs,
            embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents"
        )
        print(f"성공적으로 {len(all_docs)}개의 청크를 업데이트 했습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--added_modified", help="공백으로 구분된 추가/수정 파일 목록")
    parser.add_argument("--deleted", help="공백으로 구분된 삭제된 파일 목록")
    args = parser.parse_args()

    am_files = args.added_modified.split() if args.added_modified else []
    d_files = args.deleted.split() if args.deleted else []

    process_sync(am_files, d_files)