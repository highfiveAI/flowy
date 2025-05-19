# flowy
오프라인 회의 녹음을 기반으로 자동 회의록 작성, 역할 분담, 이메일 발송까지 도와주는 회의 정리 자동화 서비스

```
flowy
├─ app
│  ├─ core
│  │  ├─ config.py
│  │  └─ __init__.py
│  ├─ dependencies.py
│  ├─ main.py
│  ├─ models
│  │  ├─ meeting.py
│  │  └─ __init__.py
│  ├─ routers
│  │  ├─ action_assignment.py
│  │  ├─ analysis.py
│  │  ├─ feedback.py
│  │  ├─ stt.py
│  │  ├─ summarization.py
│  │  └─ __init__.py
│  ├─ services
│  │  ├─ action_item_service.py
│  │  ├─ relevance_service.py
│  │  ├─ stt_service.py
│  │  ├─ summarizer_service.py
│  │  └─ __init__.py
│  └─ __init__.py
├─ README.md
├─ requirements.txt
├─ stt.py
├─ temp_audio_uploads
├─ tests
├─ 내용.txt
├─ 사담포함.m4a
└─ 출력파일.wav

```
```
flowy
├─ .dockerignore
├─ app
│  ├─ core
│  │  ├─ config.py
│  │  └─ __init__.py
│  ├─ dependencies.py
│  ├─ dockerfile
│  ├─ main.py
│  ├─ models
│  │  ├─ meeting.py
│  │  └─ __init__.py
│  ├─ routers
│  │  ├─ action_assignment.py
│  │  ├─ analysis.py
│  │  ├─ feedback.py
│  │  ├─ stt.py
│  │  ├─ summarization.py
│  │  └─ __init__.py
│  ├─ services
│  │  ├─ action_item_service.py
│  │  ├─ relevance_service.py
│  │  ├─ stt_service.py
│  │  ├─ summarizer_service.py
│  │  └─ __init__.py
│  └─ __init__.py
├─ docker-compose.yml
├─ README.md
├─ requirements.txt
├─ temp_audio_uploads
├─ tests
├─ 내용.txt
├─ 사담포함.m4a
└─ 출력파일.wav

```