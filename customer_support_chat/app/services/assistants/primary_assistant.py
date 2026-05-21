"""
Primary Health Assistant — Main orchestrator that routes user requests to specialized agents.

Handles: general health questions, triage, routing to:
- Appointment Agent (预约挂号)
- Medication Agent (用药管理)
- Emergency Agent (急救指导)
- Health Tips Agent (健康建议)
- Medical Record Agent (病历管理)
- Health Assessment Agent (健康评估)
- Medical KB Agent (医学知识)
- Medical KG Agent (医学知识图谱)
"""

from customer_support_chat.app.services.assistants.assistant_base import (
    Assistant, CompleteOrEscalate, llm,
)
from customer_support_chat.app.services.tools.health import (
    fetch_user_health_profile,
    update_user_health_profile,
    lookup_departments,
)
from customer_support_chat.app.services.tools.medical_kb import (
    search_medical_knowledge,
    search_drug_info,
)
from customer_support_chat.app.core.logger import logger


# ── Delegation tools (one per specialized agent) ──────────────────────

class ToAppointmentBooking:
    """Delegate to Appointment Agent for booking/canceling medical appointments."""
    def __init__(self, request: str):
        self.request = request

class ToMedicationManagement:
    """Delegate to Medication Agent for managing medications, reminders, drug interactions."""
    def __init__(self, request: str):
        self.request = request

class ToEmergencyAssist:
    """Delegate to Emergency Agent for urgent medical situations and first aid."""
    def __init__(self, request: str):
        self.request = request

class ToHealthTips:
    """Delegate to Health Tips Agent for exercise, diet, sleep, mental health advice."""
    def __init__(self, request: str):
        self.request = request

class ToMedicalRecords:
    """Delegate to Medical Record Agent for viewing/adding medical history."""
    def __init__(self, request: str):
        self.request = request

class ToHealthAssessment:
    """Delegate to Health Assessment Agent for symptom checking and risk evaluation."""
    def __init__(self, request: str):
        self.request = request

class ToMedicalKnowledgeSearch:
    """Delegate to Medical KB Agent for searching medical literature and drug info."""
    def __init__(self, request: str):
        self.request = request

class ToMedicalKnowledgeGraph:
    """Delegate to Medical KG Agent for disease-symptom-medication relationships."""
    def __init__(self, request: str):
        self.request = request


# ── Primary Assistant Tools ───────────────────────────────────────────

primary_assistant_tools = [
    fetch_user_health_profile,
    update_user_health_profile,
    lookup_departments,
    search_medical_knowledge,
    search_drug_info,
    CompleteOrEscalate,
]

# All delegation tools
primary_delegation_tools = [
    ToAppointmentBooking,
    ToMedicationManagement,
    ToEmergencyAssist,
    ToHealthTips,
    ToMedicalRecords,
    ToHealthAssessment,
    ToMedicalKnowledgeSearch,
    ToMedicalKnowledgeGraph,
]

# ── System prompt ─────────────────────────────────────────────────────

PRIMARY_SYSTEM_PROMPT = """You are a Personal Health Assistant (个人健康助手). Your role is to help users manage their health.

You have access to specialized agents. Route user requests to the right agent:

1. **ToAppointmentBooking** — Booking, checking, or canceling doctor appointments
2. **ToMedicationManagement** — Managing medications, reminders, drug interactions
3. **ToEmergencyAssist** — Emergency first aid, urgent situations (chest pain, bleeding, etc.)
4. **ToHealthTips** — Exercise, diet, sleep, mental health, prevention advice
5. **ToMedicalRecords** — Viewing or adding medical records, lab results, prescriptions
6. **ToHealthAssessment** — Symptom checking, risk assessment
7. **ToMedicalKnowledgeSearch** — Searching medical knowledge, drug information
8. **ToMedicalKnowledgeGraph** — Complex relationship queries (e.g., 'what causes my symptoms?')

IMPORTANT RULES:
- Use fetch_user_health_profile first to understand the user's health background
- For ANY emergency symptoms (chest pain, severe bleeding, stroke signs), route IMMEDIATELY to ToEmergencyAssist
- NEVER provide definitive medical diagnoses — always recommend consulting a doctor
- Be empathetic and professional. Health is personal and sensitive.
- For general health questions, answer directly using your medical knowledge
- Use search_medical_knowledge and search_drug_info for factual medical information

When a task is complete, call CompleteOrEscalate with a summary."""


# ── Create the assistant runnable ─────────────────────────────────────

primary_assistant = Assistant(
    delegation_tools=primary_delegation_tools,
    domain_tools=primary_assistant_tools,
    agent_name="primary_health_assistant",
    system_prompt=PRIMARY_SYSTEM_PROMPT,
)
