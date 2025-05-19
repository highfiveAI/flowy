#!/bin/bash

# conda 초기화 (환경을 사용할 수 있게 함)
source /opt/conda/etc/profile.d/conda.sh

# flowy 환경 활성화
conda activate flowy

# FastAPI 서버 실행 (로그를 docker logs로 전달되게 함)
exec uvicorn app.main:app --host 0.0.0.0 --port 8000