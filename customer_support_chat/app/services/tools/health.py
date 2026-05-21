"""All health-domain tools for the Personal Health Assistant agents."""

import uuid
from datetime import datetime, date
from typing import Optional, List
from langchain_core.tools import tool

from customer_support_chat.app.core.mysql_client import mysql_client
from customer_support_chat.app.core.logger import logger


# ═══════════════════════════════════════════════════════════════════════
# Appointment Tools (挂号/预约)
# ═══════════════════════════════════════════════════════════════════════

@tool
def search_appointments(user_id: str) -> list:
    """
    Search for a user's existing appointments.
    Returns all scheduled/confirmed appointments for the given user.
    """
    try:
        rows = mysql_client.execute(
            "SELECT id, doctor_name, hospital_name, department, appointment_time, status, notes "
            "FROM appointments WHERE user_id = %s AND status IN ('scheduled', 'confirmed') "
            "ORDER BY appointment_time ASC",
            (user_id,),
        )
        return [dict(r) for r in rows] if rows else []
    except Exception as e:
        logger.error(f"search_appointments error: {e}")
        return []


@tool
def book_appointment(
    user_id: str,
    doctor_name: str,
    hospital_name: str,
    department: str,
    appointment_time: str,
    notes: Optional[str] = None,
) -> str:
    """
    Book a medical appointment for a user.
    Args:
        user_id: The user's unique identifier.
        doctor_name: Name of the doctor.
        hospital_name: Name of the hospital/clinic.
        department: Medical department (e.g., Cardiology, Dermatology).
        appointment_time: ISO datetime string for the appointment.
        notes: Optional notes or symptoms description.
    """
    try:
        apt_id = f"APT-{uuid.uuid4().hex[:8].upper()}"
        mysql_client.execute(
            "INSERT INTO appointments (id, user_id, doctor_name, hospital_name, department, appointment_time, notes) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (apt_id, user_id, doctor_name, hospital_name, department, appointment_time, notes or ""),
            fetch=False,
        )
        return f"✅ Appointment booked: {apt_id}\n  Doctor: {doctor_name}\n  Hospital: {hospital_name}\n  Department: {department}\n  Time: {appointment_time}"
    except Exception as e:
        logger.error(f"book_appointment error: {e}")
        return f"❌ Failed to book appointment: {e}"


@tool
def cancel_appointment(appointment_id: str, user_id: str) -> str:
    """Cancel a previously booked appointment."""
    try:
        rows = mysql_client.execute(
            "SELECT id FROM appointments WHERE id = %s AND user_id = %s AND status != 'cancelled'",
            (appointment_id, user_id),
        )
        if not rows:
            return f"❌ Appointment {appointment_id} not found or already cancelled."
        mysql_client.execute(
            "UPDATE appointments SET status = 'cancelled' WHERE id = %s",
            (appointment_id,),
            fetch=False,
        )
        return f"✅ Appointment {appointment_id} cancelled."
    except Exception as e:
        logger.error(f"cancel_appointment error: {e}")
        return f"❌ Failed to cancel: {e}"


@tool
def lookup_departments(symptom: Optional[str] = None) -> list:
    """
    Look up medical departments. Optionally filter by symptom keyword.
    Returns matching departments with descriptions.
    """
    departments = [
        {"dept": "Cardiology", "keywords": "chest pain, heart, palpitations, blood pressure", "desc": "Heart and cardiovascular system"},
        {"dept": "Dermatology", "keywords": "skin, rash, acne, eczema, mole", "desc": "Skin conditions"},
        {"dept": "Endocrinology", "keywords": "diabetes, thyroid, hormone, metabolism", "desc": "Hormones and metabolism"},
        {"dept": "Gastroenterology", "keywords": "stomach, digestion, abdominal pain, nausea", "desc": "Digestive system"},
        {"dept": "Neurology", "keywords": "headache, migraine, dizziness, nerve", "desc": "Brain and nervous system"},
        {"dept": "Orthopedics", "keywords": "bone, joint, fracture, back pain, arthritis", "desc": "Bones, joints, and muscles"},
        {"dept": "Ophthalmology", "keywords": "eye, vision, blurry, glaucoma", "desc": "Eye care"},
        {"dept": "Pediatrics", "keywords": "child, infant, baby, fever", "desc": "Children's health"},
        {"dept": "Psychiatry", "keywords": "anxiety, depression, insomnia, mental", "desc": "Mental health"},
        {"dept": "Pulmonology", "keywords": "cough, breathing, asthma, lung", "desc": "Respiratory system"},
        {"dept": "Urology", "keywords": "urinary, kidney, bladder, prostate", "desc": "Urinary system"},
        {"dept": "General Practice", "keywords": "general, checkup, fever, cold, flu", "desc": "Primary care and general health"},
    ]
    if not symptom:
        return [{"name": d["dept"], "description": d["desc"]} for d in departments]
    s = symptom.lower()
    return [{"name": d["dept"], "description": d["desc"]} for d in departments if s in d["keywords"].lower()]


# ═══════════════════════════════════════════════════════════════════════
# Medication Tools (用药管理)
# ═══════════════════════════════════════════════════════════════════════

@tool
def search_medications(user_id: str) -> list:
    """Get all active medications for a user."""
    try:
        rows = mysql_client.execute(
            "SELECT id, medication_name, dosage, frequency, start_date, end_date, "
            "reminder_time, status, notes FROM medications "
            "WHERE user_id = %s AND status = 'active' ORDER BY created_at DESC",
            (user_id,),
        )
        return [dict(r) for r in rows] if rows else []
    except Exception as e:
        logger.error(f"search_medications error: {e}")
        return []


@tool
def add_medication(
    user_id: str,
    medication_name: str,
    dosage: str,
    frequency: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    reminder_time: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Add a medication to the user's regimen.
    Args:
        user_id: User identifier.
        medication_name: Name of the medication (e.g., 'Metformin 500mg').
        dosage: Dosage instruction (e.g., '1 tablet').
        frequency: How often (e.g., 'twice daily', 'every 8 hours').
        start_date: Start date (YYYY-MM-DD).
        end_date: End date, optional.
        reminder_time: Daily reminder time (HH:MM), optional.
        notes: Additional instructions.
    """
    try:
        med_id = f"MED-{uuid.uuid4().hex[:8].upper()}"
        mysql_client.execute(
            "INSERT INTO medications (id, user_id, medication_name, dosage, frequency, start_date, end_date, reminder_time, notes) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (med_id, user_id, medication_name, dosage, frequency, start_date or date.today().isoformat(),
             end_date, reminder_time, notes or ""),
            fetch=False,
        )
        reminder = f" Reminder at {reminder_time} daily." if reminder_time else ""
        return f"✅ Medication added: {med_id}\n  {medication_name} — {dosage}, {frequency}{reminder}"
    except Exception as e:
        logger.error(f"add_medication error: {e}")
        return f"❌ Failed: {e}"


@tool
def update_medication_status(medication_id: str, user_id: str, new_status: str) -> str:
    """Update medication status: 'active', 'paused', or 'completed'."""
    try:
        rows = mysql_client.execute(
            "SELECT id FROM medications WHERE id = %s AND user_id = %s",
            (medication_id, user_id),
        )
        if not rows:
            return f"❌ Medication {medication_id} not found."
        valid = {"active", "paused", "completed"}
        if new_status not in valid:
            return f"❌ Invalid status: {new_status}. Use: {valid}"
        mysql_client.execute(
            "UPDATE medications SET status = %s WHERE id = %s",
            (new_status, medication_id),
            fetch=False,
        )
        return f"✅ Medication {medication_id} status → {new_status}."
    except Exception as e:
        logger.error(f"update_medication_status error: {e}")
        return f"❌ Failed: {e}"


@tool
def drug_interaction_check(medication_names: str) -> str:
    """
    Check for known drug interactions. Provide medication names separated by commas.
    This is a rule-based check for common interactions. Always consult a doctor.
    """
    interactions = {
        ("warfarin", "aspirin"): "⚠️ HIGH RISK: Increased bleeding risk. Do not combine without medical supervision.",
        ("warfarin", "ibuprofen"): "⚠️ HIGH RISK: NSAIDs increase bleeding risk with warfarin.",
        ("metformin", "alcohol"): "⚠️ MODERATE: Risk of lactic acidosis. Avoid heavy alcohol consumption.",
        ("lisinopril", "potassium"): "⚠️ MODERATE: May increase potassium levels. Monitor intake.",
        ("statins", "grapefruit"): "⚠️ MODERATE: Grapefruit increases statin levels. Avoid grapefruit juice.",
        ("aspirin", "ibuprofen"): "⚠️ MODERATE: Increased GI bleeding risk. Space doses apart.",
    }
    names = [n.strip().lower() for n in medication_names.split(",")]
    warnings = []
    for (a, b), warning in interactions.items():
        for n1 in names:
            for n2 in names:
                if n1 != n2 and (a in n1 or n1 in a) and (b in n2 or n2 in b):
                    warnings.append(f"  {warning}")
    if warnings:
        return "🔍 Drug Interaction Check Results:\n" + "\n".join(set(warnings)) + "\n\n⚠️ This is not exhaustive. Consult your doctor or pharmacist."
    return "✅ No known major interactions found for these medications. Still, consult your doctor."


# ═══════════════════════════════════════════════════════════════════════
# Emergency Tools (急救/紧急)
# ═══════════════════════════════════════════════════════════════════════

@tool
def emergency_guidance(situation: str) -> str:
    """
    Provide first-aid and emergency guidance for common medical emergencies.
    Args:
        situation: Description of the emergency (e.g., 'chest pain', 'severe bleeding', 'choking').
    """
    guidance = {
        "chest pain": "🚨 EMERGENCY: Call 120 immediately.\n  • Have the person sit down and stay calm\n  • If not allergic, chew 300mg aspirin\n  • Loosen tight clothing\n  • If unconscious and not breathing, begin CPR",
        "choking": "🚨 If person cannot cough, speak, or breathe:\n  • Call 120\n  • Perform Heimlich maneuver: stand behind, fist above navel, thrust inward and upward\n  • For infants: 5 back blows + 5 chest thrusts",
        "severe bleeding": "🚨 Call 120.\n  • Apply direct pressure with clean cloth\n  • Elevate the wound above heart level\n  • Do NOT remove embedded objects\n  • Apply tourniquet only as last resort",
        "stroke": "🚨 Call 120 immediately. Remember FAST:\n  • Face drooping\n  • Arm weakness\n  • Speech difficulty\n  • Time to call emergency",
        "seizure": "⚠️ Stay calm.\n  • Clear the area of dangerous objects\n  • Do NOT restrain or put anything in mouth\n  • Time the seizure\n  • Call 120 if >5 minutes or first seizure",
        "allergic reaction": "⚠️ If severe (anaphylaxis):\n  • Call 120\n  • Use epinephrine auto-injector if available\n  • Have person lie down, elevate legs\n  • For mild: antihistamines may help",
        "burn": "⚠️ For thermal burns:\n  • Cool under running water for 20 minutes\n  • Remove clothing/jewelry near burn\n  • Cover with clean, non-stick dressing\n  • Seek medical help for large or deep burns",
        "fracture": "⚠️ Do NOT move if spine injury suspected.\n  • Immobilize the injured area\n  • Apply ice wrapped in cloth\n  • Seek medical attention",
    }
    query = situation.lower()
    for key, response in guidance.items():
        if key in query:
            return response
    return (
        "🚨 For any medical emergency, call 120 (China) or your local emergency number.\n\n"
        "General first-aid principles:\n"
        "  • Stay calm and assess the situation\n"
        "  • Ensure the scene is safe\n"
        "  • Check responsiveness and breathing\n"
        "  • Control severe bleeding with direct pressure\n"
        "  • Do not move the person if spinal injury is suspected\n"
        f"  • For '{situation}', seek professional medical evaluation.\n\n"
        "⚠️ This guidance is for informational purposes only and does not replace professional medical advice."
    )


@tool
def emergency_services_nearby(location: str = "") -> str:
    """
    Provide information about nearby emergency services.
    In production, this would query a real geo-location API.
    """
    return (
        "🏥 Emergency Services Information:\n\n"
        "  • Emergency Number (China): 📞 120\n"
        "  • Police: 📞 110\n"
        "  • Fire: 📞 119\n\n"
        "Nearby Hospitals (sample data — production would use geo-API):\n"
        "  • Peking Union Medical College Hospital — Emergency Dept 24/7\n"
        "  • Chinese PLA General Hospital (301) — 24/7 Emergency\n"
        "  • Ruijin Hospital (Shanghai) — 24/7 Emergency\n"
        "  • West China Hospital (Chengdu) — 24/7 Emergency\n\n"
        f"📍 For your location ({location or 'unknown'}), use map apps to find the nearest ER."
    )


# ═══════════════════════════════════════════════════════════════════════
# Health Tips Tools (健康建议)
# ═══════════════════════════════════════════════════════════════════════

@tool
def health_tips_by_category(category: str) -> list:
    """
    Get health tips by category: exercise, diet, sleep, mental_health, chronic, prevention.
    """
    tips_db = {
        "exercise": [
            {"tip": "Aim for 150 minutes of moderate aerobic activity per week", "source": "WHO Guidelines"},
            {"tip": "Include strength training 2+ times per week for all major muscle groups", "source": "ACSM"},
            {"tip": "Take 10,000 steps daily — use stairs instead of elevator", "source": "CDC"},
            {"tip": "Warm up for 5-10 minutes before exercise to prevent injury", "source": "Mayo Clinic"},
            {"tip": "Stretch after exercise when muscles are warm for best flexibility gains", "source": "Harvard Health"},
        ],
        "diet": [
            {"tip": "Eat 5 servings of fruits and vegetables daily for optimal nutrition", "source": "WHO"},
            {"tip": "Limit added sugar to 25g/day (women) or 36g/day (men)", "source": "AHA"},
            {"tip": "Choose whole grains over refined carbohydrates", "source": "Harvard Nutrition"},
            {"tip": "Drink 8 glasses (2L) of water daily — more if exercising", "source": "Mayo Clinic"},
            {"tip": "Reduce sodium intake to <2300mg/day for heart health", "source": "AHA"},
        ],
        "sleep": [
            {"tip": "Adults need 7-9 hours of quality sleep per night", "source": "National Sleep Foundation"},
            {"tip": "Maintain a consistent sleep schedule — even on weekends", "source": "CDC"},
            {"tip": "Avoid screens 1 hour before bedtime (blue light disrupts melatonin)", "source": "Harvard Health"},
            {"tip": "Keep bedroom cool (18-20°C), dark, and quiet for optimal sleep", "source": "Sleep Foundation"},
            {"tip": "Limit caffeine after 2 PM to avoid sleep disruption", "source": "AASM"},
        ],
        "mental_health": [
            {"tip": "Practice mindfulness meditation 10-15 minutes daily to reduce stress", "source": "APA"},
            {"tip": "Social connections are vital — maintain relationships with friends and family", "source": "Harvard Study"},
            {"tip": "Regular exercise reduces depression and anxiety symptoms by 20-30%", "source": "Lancet Psychiatry"},
            {"tip": "Keep a gratitude journal — write 3 things you're grateful for daily", "source": "Positive Psychology"},
            {"tip": "Seek professional help if symptoms persist >2 weeks. Therapy is effective for 75% of people.", "source": "APA"},
        ],
        "chronic": [
            {"tip": "Monitor blood pressure regularly — target <120/80 mmHg", "source": "AHA"},
            {"tip": "Diabetics: check HbA1c every 3 months, target <7%", "source": "ADA"},
            {"tip": "Take medications as prescribed — don't stop without consulting your doctor", "source": "WHO"},
            {"tip": "Keep a symptom diary to track patterns and triggers", "source": "Mayo Clinic"},
            {"tip": "Regular check-ups catch issues early — don't skip annual physicals", "source": "CDC"},
        ],
        "prevention": [
            {"tip": "Get annual flu vaccine — reduces risk by 40-60%", "source": "CDC"},
            {"tip": "Wash hands frequently with soap for 20 seconds", "source": "WHO"},
            {"tip": "Regular cancer screenings: mammogram (40+), colonoscopy (45+), pap smear (21+)", "source": "ACS"},
            {"tip": "Use sunscreen SPF 30+ daily to prevent skin cancer and aging", "source": "AAD"},
            {"tip": "Don't smoke — it's the #1 preventable cause of death worldwide", "source": "WHO"},
        ],
    }
    key = category.lower().strip()
    if key in tips_db:
        return tips_db[key]
    return tips_db.get("prevention", [])


@tool
def bmi_calculator(height_cm: float, weight_kg: float) -> str:
    """
    Calculate Body Mass Index (BMI) and provide interpretation.
    Args:
        height_cm: Height in centimeters.
        weight_kg: Weight in kilograms.
    """
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Underweight"
        advice = "Consider consulting a nutritionist. Focus on nutrient-dense foods."
    elif bmi < 24.9:
        category = "Normal weight"
        advice = "Great! Maintain your healthy lifestyle with balanced diet and exercise."
    elif bmi < 29.9:
        category = "Overweight"
        advice = "Consider increasing physical activity and reviewing your diet. Aim for gradual weight loss of 0.5-1 kg/week."
    else:
        category = "Obese"
        advice = "Consult a healthcare provider for a personalized weight management plan. Even 5-10% weight loss provides significant health benefits."
    return f"📊 BMI: {bmi:.1f} — {category}\n💡 {advice}"


# ═══════════════════════════════════════════════════════════════════════
# Medical Record Tools (病历管理)
# ═══════════════════════════════════════════════════════════════════════

@tool
def search_medical_records(user_id: str, record_type: Optional[str] = None) -> list:
    """
    Search user's medical records. Optionally filter by type.
    Types: diagnosis, lab_result, prescription, vaccination, surgery, other.
    """
    try:
        if record_type:
            rows = mysql_client.execute(
                "SELECT id, record_type, title, description, doctor_name, hospital_name, record_date "
                "FROM medical_records WHERE user_id = %s AND record_type = %s ORDER BY record_date DESC LIMIT 20",
                (user_id, record_type),
            )
        else:
            rows = mysql_client.execute(
                "SELECT id, record_type, title, description, doctor_name, hospital_name, record_date "
                "FROM medical_records WHERE user_id = %s ORDER BY record_date DESC LIMIT 20",
                (user_id,),
            )
        return [dict(r) for r in rows] if rows else []
    except Exception as e:
        logger.error(f"search_medical_records error: {e}")
        return []


@tool
def add_medical_record(
    user_id: str,
    record_type: str,
    title: str,
    description: str,
    doctor_name: Optional[str] = None,
    hospital_name: Optional[str] = None,
    record_date: Optional[str] = None,
) -> str:
    """
    Add a new medical record for the user.
    Args:
        user_id: User identifier.
        record_type: Type (diagnosis, lab_result, prescription, vaccination, surgery, other).
        title: Brief title for the record.
        description: Detailed description.
        doctor_name: Name of the attending doctor, optional.
        hospital_name: Name of the hospital, optional.
        record_date: Date of the record (YYYY-MM-DD), optional.
    """
    valid_types = {"diagnosis", "lab_result", "prescription", "vaccination", "surgery", "other"}
    if record_type not in valid_types:
        return f"❌ Invalid record_type: {record_type}. Use: {valid_types}"
    try:
        rec_id = f"REC-{uuid.uuid4().hex[:8].upper()}"
        mysql_client.execute(
            "INSERT INTO medical_records (id, user_id, record_type, title, description, doctor_name, hospital_name, record_date) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (rec_id, user_id, record_type, title, description, doctor_name or "", hospital_name or "",
             record_date or date.today().isoformat()),
            fetch=False,
        )
        return f"✅ Medical record added: {rec_id} — {title} ({record_type})"
    except Exception as e:
        logger.error(f"add_medical_record error: {e}")
        return f"❌ Failed: {e}"


# ═══════════════════════════════════════════════════════════════════════
# Health Assessment Tools (健康评估)
# ═══════════════════════════════════════════════════════════════════════

@tool
def submit_health_assessment(
    user_id: str,
    assessment_type: str,
    symptoms: str,
) -> str:
    """
    Submit a health assessment for the user (symptom check / risk assessment).
    Args:
        user_id: User identifier.
        assessment_type: Type of assessment (general_checkup, symptom_review, lifestyle_risk, mental_health).
        symptoms: Description of symptoms or concerns.
    """
    valid_types = {"general_checkup", "symptom_review", "lifestyle_risk", "mental_health"}
    if assessment_type not in valid_types:
        return f"❌ Invalid assessment_type: {assessment_type}. Use: {valid_types}"
    try:
        import hashlib
        # Simple risk estimation based on symptom keywords
        red_flags = ["severe", "chest pain", "breathing difficulty", "unconscious", "bleeding", "stroke",
                     "suicide", "self harm", "seizure", "paralysis"]
        amber_flags = ["persistent", "worsening", "fever", "vomiting", "diarrhea", "weight loss",
                       "anxiety", "depression", "insomnia"]
        symptoms_lower = symptoms.lower()
        if any(f in symptoms_lower for f in red_flags):
            risk = "urgent"
            recommendation = "⚠️ Seek immediate medical attention. Call 120 or visit the nearest ER."
        elif any(f in symptoms_lower for f in amber_flags):
            risk = "medium"
            recommendation = "Schedule an appointment with your doctor within 1-2 weeks. Monitor symptoms."
        else:
            risk = "low"
            recommendation = "Monitor your symptoms. If they persist or worsen, consult your doctor."

        score = 100 if risk == "urgent" else 50 if risk == "medium" else 10
        rec_id = hashlib.md5(f"{user_id}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        mysql_client.execute(
            "INSERT INTO health_assessments (id, user_id, assessment_type, symptoms, risk_level, recommendation, score) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (rec_id, user_id, assessment_type, symptoms, risk, recommendation, score),
            fetch=False,
        )
        return f"📋 Health Assessment ({assessment_type})\n  Risk Level: {risk.upper()}\n  Recommendation: {recommendation}\n  ID: {rec_id}"
    except Exception as e:
        logger.error(f"submit_health_assessment error: {e}")
        return f"❌ Failed: {e}"


# ═══════════════════════════════════════════════════════════════════════
# User Tools
# ═══════════════════════════════════════════════════════════════════════

@tool
def fetch_user_health_profile(user_id: str) -> str:
    """Fetch a user's health profile including allergies, conditions, and recent records."""
    try:
        rows = mysql_client.execute(
            "SELECT name, date_of_birth, gender, blood_type, allergies, chronic_conditions, emergency_contact "
            "FROM users WHERE id = %s",
            (user_id,),
        )
        if not rows:
            return f"❌ User {user_id} not found."
        u = rows[0]
        profile = f"👤 Health Profile: {u['name']}\n"
        profile += f"  DOB: {u.get('date_of_birth', 'N/A')} | Gender: {u.get('gender', 'N/A')} | Blood: {u.get('blood_type', 'N/A')}\n"
        profile += f"  Allergies: {u.get('allergies', 'None')}\n"
        profile += f"  Chronic Conditions: {u.get('chronic_conditions', 'None')}\n"
        profile += f"  Emergency Contact: {u.get('emergency_contact', 'Not set')}"
        return profile
    except Exception as e:
        logger.error(f"fetch_user_health_profile error: {e}")
        return f"❌ Error: {e}"


@tool
def update_user_health_profile(
    user_id: str,
    allergies: Optional[str] = None,
    chronic_conditions: Optional[str] = None,
    emergency_contact: Optional[str] = None,
) -> str:
    """Update user's health profile fields."""
    updates = []
    params = []
    if allergies is not None:
        updates.append("allergies = %s")
        params.append(allergies)
    if chronic_conditions is not None:
        updates.append("chronic_conditions = %s")
        params.append(chronic_conditions)
    if emergency_contact is not None:
        updates.append("emergency_contact = %s")
        params.append(emergency_contact)
    if not updates:
        return "❌ No fields to update."
    try:
        params.append(user_id)
        mysql_client.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = %s",
            tuple(params),
            fetch=False,
        )
        return "✅ Health profile updated."
    except Exception as e:
        logger.error(f"update_user_health_profile error: {e}")
        return f"❌ Failed: {e}"
