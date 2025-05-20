from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from typing import Optional, List
import os
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
import json
from collections import defaultdict

# 환경 변수 로딩
load_dotenv(override=True)

# 나눔휴먼 폰트 등록
FONT_PATH = "app/static/fonts/NanumHumanRegular.ttf"
pdfmetrics.registerFont(TTFont('NanumHuman', FONT_PATH))

router = APIRouter()

class MeetingParticipant(BaseModel):
    name: str
    email: EmailStr
    role: str

class MeetingInfo(BaseModel):
    subj: str
    dt: str
    loc: str
    info_n: List[MeetingParticipant]
    summary_result: str
    action_items_result: List[dict] = []
    feedback_result: str

def safe_paragraph(text, style):
    if text:
        text = text.replace('<br>', '').replace('<br/>', '')
    return Paragraph(text, style)

def create_meeting_pdf(meeting_info: MeetingInfo) -> str:
    # PDF 파일 저장 경로 설정
    pdf_dir = "app/static/pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    
    # PDF 파일명 생성 (회의 주제와 날짜를 포함)
    filename = f"meeting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(pdf_dir, filename)
    
    # PDF 문서 생성
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # 한글 스타일 정의
    styles.add(ParagraphStyle(
        name='KoreanTitle',
        fontName='NanumHuman',
        fontSize=16,
        spaceAfter=30
    ))
    
    styles.add(ParagraphStyle(
        name='KoreanHeading',
        fontName='NanumHuman',
        fontSize=14,
        spaceAfter=12
    ))
    
    styles.add(ParagraphStyle(
        name='KoreanNormal',
        fontName='NanumHuman',
        fontSize=10,
        spaceAfter=12
    ))
    
    story = []
    
    # 회의 정보 섹션
    story.append(safe_paragraph("회의 정보", styles['KoreanTitle']))
    
    # 회의 기본 정보 테이블
    meeting_data = [
        ["회의 주제", meeting_info.subj],
        ["회의 일자", meeting_info.dt],
        ["회의 위치", meeting_info.loc]
    ]
    
    meeting_table = Table(meeting_data, colWidths=[100, 400])
    meeting_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'NanumHuman'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(meeting_table)
    story.append(Spacer(1, 20))
    
    # 참석자 명단
    story.append(safe_paragraph("참석자 명단", styles['KoreanHeading']))
    participants_data = [["이름", "이메일", "역할"]]
    for participant in meeting_info.info_n:
        participants_data.append([
            participant.name,
            participant.email,
            participant.role
        ])
    
    participants_table = Table(participants_data, colWidths=[100, 200, 200])
    participants_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'NanumHuman'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTNAME', (0, 1), (-1, -1), 'NanumHuman'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(participants_table)
    story.append(Spacer(1, 20))
    
    # 회의 요약
    story.append(safe_paragraph("회의 요약", styles['KoreanHeading']))
    summary_lines = []
    try:
        summary_json = json.loads(meeting_info.summary_result)
        if isinstance(summary_json, dict) and "summary" in summary_json:
            if summary_json["summary"]:
                summary_lines = [line.strip() for line in summary_json["summary"] if line.strip()]
    except Exception:
        pass

    if summary_lines:
        for line in summary_lines:
            story.append(safe_paragraph(line, styles['KoreanNormal']))
    else:
        story.append(safe_paragraph("", styles['KoreanNormal']))
    story.append(Spacer(1, 20))
    
    # 역할분담
    story.append(safe_paragraph("역할분담", styles['KoreanHeading']))
    if not meeting_info.action_items_result:
        story.append(safe_paragraph("역할분담 내용이 없습니다.", styles['KoreanNormal']))
    else:
        # name+role별로 tasks를 모으는 dict 생성
        task_map = defaultdict(list)
        for task in meeting_info.action_items_result:
            name = task.get('assignee', '') or task.get('name', '')
            role = task.get('role', '')
            task_content = task.get('task', '') or task.get('tasks', '')
            key = (name, role)
            if isinstance(task_content, list):
                task_map[key].extend([str(t) for t in task_content if t])
            elif task_content:
                task_map[key].append(str(task_content))
        for (name, role), tasks in task_map.items():
            if name and role and tasks:
                line = f"{name}({role}) : {', '.join(tasks)}"
            elif name and role:
                line = f"{name}({role})"
            elif name and tasks:
                line = f"{name} : {', '.join(tasks)}"
            elif role and tasks:
                line = f"({role}) : {', '.join(tasks)}"
            else:
                line = f"{name}{role}{', '.join(tasks)}"
            story.append(safe_paragraph(line, styles['KoreanNormal']))
    story.append(Spacer(1, 20))
    
    # 회의 피드백
    story.append(safe_paragraph("회의 피드백", styles['KoreanHeading']))
    feedback_text = ""
    try:
        feedback_json = json.loads(meeting_info.feedback_result)
        lines = []
        if isinstance(feedback_json, dict):
            necessary = feedback_json.get("necessary_ratio", None)
            unnecessary = feedback_json.get("unnecessary_ratio", None)
            if necessary is not None and unnecessary is not None:
                lines.append(f"주제와 일치하는 내용 : {necessary}%")
                lines.append(f"주제와 무관한 내용 : {unnecessary}%")
            rep_unnecessary = feedback_json.get("representative_unnecessary", [])
            for item in rep_unnecessary:
                sentence = item.get("sentence", "")
                reason = item.get("reason", "")
                if sentence and reason:
                    lines.append(f"주제와 무관한 문장: {sentence} / 이유: {reason}")
                elif sentence:
                    lines.append(f"주제와 무관한 문장: {sentence}")
                elif reason:
                    lines.append(f"이유: {reason}")
        feedback_text = "\n".join(lines)
    except Exception:
        feedback_text = ""
    # 여러 줄을 각각 Paragraph로 출력
    if feedback_text:
        for line in feedback_text.split("\n"):
            story.append(safe_paragraph(line, styles['KoreanNormal']))
    else:
        story.append(safe_paragraph("", styles['KoreanNormal']))
    
    # PDF 생성
    doc.build(story)
    return pdf_path

@router.post("/send-email")
async def send_email(request: Request):
    try:
        data = await request.json()
        meeting_info_data = data["meeting_info"]
        # MeetingInfo 객체로 변환
        meeting_info = MeetingInfo(**meeting_info_data)

        # name, email 백엔드에서 직접 지정
        name = "FLOWY"
        email = "dohyeongim29@gmail.com"

        # PDF 파일 생성
        pdf_path = create_meeting_pdf(meeting_info)
        
        # 이메일 설정
        conf = ConnectionConfig(
            MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
            MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
            MAIL_FROM=os.getenv("MAIL_FROM"),
            MAIL_PORT=int(os.getenv("MAIL_PORT", "587")),
            MAIL_SERVER=os.getenv("MAIL_SERVER"),
            MAIL_STARTTLS=True,
            MAIL_SSL_TLS=False,
            USE_CREDENTIALS=True
        )
        
        # 이메일 메시지 생성 및 발송 (참석자 각각에게)
        for participant in meeting_info.info_n:
            name = participant.name
            email = participant.email
            message = MessageSchema(
                subject=f"[FLOWY] {meeting_info.dt} '{meeting_info.subj}' 회의록",
                recipients=[email],
                body=f"""
                안녕하세요, {name}님 FLOWY입니다.<br><br>

                {meeting_info.dt}에 진행된 '{meeting_info.subj}' 회의록을 전달드립니다.<br><br>

                회의의 주요 내용과 논의 결과는 첨부된 PDF 파일에서 확인하실 수 있습니다.<br><br>

                감사합니다.<br><br>

                FLOWY 드림
                """,
                subtype="html",
                attachments=[{"file": pdf_path}]
            )
            fm = FastMail(conf)
            await fm.send_message(message)
        
        return {"message": "이메일이 성공적으로 발송되었습니다."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  
