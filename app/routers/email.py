from fastapi import APIRouter, HTTPException
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
    date: str
    start_time: str
    end_time: str
    participants: List[MeetingParticipant]
    summary: str
    tasks: List[dict] = []
    feedback: str

class EmailRequest(BaseModel):
    name: str
    email: EmailStr
    meeting_info: MeetingInfo

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
    story.append(Paragraph("회의 정보", styles['KoreanTitle']))
    
    # 회의 기본 정보 테이블
    meeting_data = [
        ["회의 주제", meeting_info.subj],
        ["회의 일자", meeting_info.date],
        ["회의 시간", f"{meeting_info.start_time} ~ {meeting_info.end_time}"]
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
    story.append(Paragraph("참석자 명단", styles['KoreanHeading']))
    participants_data = [["이름", "이메일", "역할"]]
    for participant in meeting_info.participants:
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
    story.append(Paragraph("회의 요약", styles['KoreanHeading']))
    story.append(Paragraph(meeting_info.summary, styles['KoreanNormal']))
    story.append(Spacer(1, 20))
    
    # 역할분담
    story.append(Paragraph("역할분담", styles['KoreanHeading']))
    if meeting_info.tasks:
        tasks_data = [["담당자", "업무", "기한"]]
        for task in meeting_info.tasks:
            tasks_data.append([
                task.get('assignee', ''),
                task.get('task', ''),
                task.get('deadline', '')
            ])
        
        tasks_table = Table(tasks_data, colWidths=[100, 300, 100])
        tasks_table.setStyle(TableStyle([
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
        story.append(tasks_table)
    else:
        story.append(Paragraph("역할분담 내용이 없습니다.", styles['KoreanNormal']))
    
    story.append(Spacer(1, 20))
    
    # 회의 피드백
    story.append(Paragraph("회의 피드백", styles['KoreanHeading']))
    story.append(Paragraph(meeting_info.feedback, styles['KoreanNormal']))
    
    # PDF 생성
    doc.build(story)
    return pdf_path

@router.post("/send-email")
async def send_email(request: EmailRequest):
    try:
        # PDF 파일 생성
        pdf_path = create_meeting_pdf(request.meeting_info)
        
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
        
        # 이메일 메시지 생성
        message = MessageSchema(
            subject=f"[FLOWY] {request.meeting_info.date} '{request.meeting_info.subj}' 회의록",
            recipients=[request.email],
            body=f"""
            안녕하세요, {request.name}님 FLOWY입니다.<br><br>

            {request.meeting_info.date} {request.meeting_info.start_time}에 진행된 '{request.meeting_info.subj}' 회의록을 전달드립니다.<br><br>

            회의의 주요 내용과 논의 결과는 첨부된 PDF 파일에서 확인하실 수 있습니다.<br><br>

            감사합니다.<br><br>

            FLOWY 드림
            """,
            subtype="html",
            attachments=[{"file": pdf_path}]
        )
        
        # 이메일 발송
        fm = FastMail(conf)
        await fm.send_message(message)
        
        return {"message": "이메일이 성공적으로 발송되었습니다."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  