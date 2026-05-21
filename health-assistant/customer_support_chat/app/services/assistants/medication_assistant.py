"""Medication Management Agent — Track medications, reminders, drug interactions."""

from customer_support_chat.app.services.assistants.assistant_base import Assistant, CompleteOrEscalate
from customer_support_chat.app.services.tools.health import (
    search_medications, add_medication, update_medication_status, drug_interaction_check,
)
from customer_support_chat.app.services.tools.medical_kb import search_drug_info

medication_safe_tools = [search_medications, drug_interaction_check, search_drug_info, CompleteOrEscalate]
medication_sensitive_tools = [add_medication, update_medication_status]

medication_management_assistant = Assistant(
    [],
    medication_safe_tools + medication_sensitive_tools,
    "medication_management",
    """You are a Medication Management specialist (用药管理助手).

Your responsibilities:
1. Track user's current medications
2. Add new medications with dosage, frequency, and reminders
3. Update medication status (active/paused/completed)
4. Check for drug interactions between medications
5. Provide drug information and side effects

Process:
- Use search_medications to see current active medications
- Use drug_interaction_check before adding any new medication
- Use search_drug_info for detailed medication information
- Use add_medication to register a new medication (sensitive — requires approval)
- Suggest reminder times based on medication frequency
- Warn about common interactions and side effects

CRITICAL: Always check drug interactions BEFORE adding a new medication.
Never recommend stopping prescribed medications without consulting their doctor.
When done, use CompleteOrEscalate.""",
)
