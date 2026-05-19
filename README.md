[ Agent 실행 명령어 ]
. 서버: node .pimono_bridge/dash_pimono_server.mjs
. 앱: streamlit run 05.18_pimono_v16.py

[ PostgreSQL & PG Vector 테이블 ]
. 표준용어 임베딩 조회: select * from standard_term_vectors
. 표준단어 keyword 조회: select * from standard_words
. 표준용어 keyword 조회: select * from standard_terms
. 사용자 문의 및 LLM 답변 이력: select * from qna_hist_vectors
. RAG 테이블: select * from answer_document_vectors
