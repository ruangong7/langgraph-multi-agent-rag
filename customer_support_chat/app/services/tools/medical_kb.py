"""Medical knowledge search tools for the Health Assistant."""

from langchain_core.tools import tool


@tool
def search_medical_knowledge(query: str) -> str:
    """
    Search medical knowledge base for information about diseases, medications, treatments, etc.
    This is a structured medical knowledge tool — in production, queries a vector DB.
    """
    knowledge_entries = {
        "hypertension": "Hypertension (High Blood Pressure): BP consistently >130/80 mmHg. "
            "Risk factors: age, obesity, high sodium diet, family history, sedentary lifestyle. "
            "Management: DASH diet, exercise, medication (ACE inhibitors, beta-blockers, calcium channel blockers), "
            "regular BP monitoring. Complications if untreated: stroke, heart attack, kidney failure.",
        "diabetes": "Diabetes Mellitus Type 2: Chronic condition where body becomes insulin resistant. "
            "Symptoms: increased thirst, frequent urination, fatigue, blurred vision, slow healing. "
            "Management: blood glucose monitoring, metformin, healthy diet, exercise, HbA1c <7%. "
            "Complications: neuropathy, retinopathy, cardiovascular disease.",
        "covid": "COVID-19: Respiratory illness caused by SARS-CoV-2. "
            "Symptoms: fever, dry cough, fatigue, loss of taste/smell, shortness of breath. "
            "Prevention: vaccination, mask in crowded areas, good hand hygiene. "
            "Treatment: rest, hydration, antipyretics for fever; severe cases need hospitalization.",
        "insomnia": "Insomnia: Difficulty falling or staying asleep. "
            "Causes: stress, anxiety, caffeine, irregular schedule, screen time before bed. "
            "Management: cognitive behavioral therapy for insomnia (CBT-I), sleep hygiene, "
            "melatonin supplements, avoiding screens 1hr before bed. "
            "See a doctor if symptoms persist >4 weeks.",
        "migraine": "Migraine: Recurrent severe headache, often with nausea, light/sound sensitivity. "
            "Triggers: stress, certain foods, hormonal changes, weather changes. "
            "Treatment: NSAIDs, triptans, preventive medications (beta-blockers, topiramate). "
            "Non-drug: dark quiet room, cold compress, regular sleep schedule.",
        "asthma": "Asthma: Chronic airway inflammation causing wheezing, coughing, chest tightness. "
            "Triggers: allergens, exercise, cold air, respiratory infections. "
            "Management: inhaled corticosteroids (controller), bronchodilators (reliever), "
            "avoid triggers, peak flow monitoring. Action plan: green/yellow/red zones.",
        "depression": "Major Depressive Disorder: Persistent low mood, loss of interest, fatigue, "
            "sleep/appetite changes persisting >2 weeks. "
            "Treatment: psychotherapy (CBT), SSRIs/SNRIs medication, regular exercise, "
            "social support, light therapy for seasonal pattern. "
            "🚨 If suicidal thoughts: call 988 (US) or 400-161-9995 (China Mental Health Hotline).",
        "allergy": "Allergic Rhinitis: Immune response to allergens (pollen, dust mites, pet dander). "
            "Symptoms: sneezing, runny nose, itchy eyes, congestion. "
            "Treatment: antihistamines (cetirizine, loratadine), nasal corticosteroids, "
            "allergen avoidance, air purifiers. Consider allergy testing for immunotherapy.",
        "obesity": "Obesity: BMI ≥30 kg/m². Associated with diabetes, hypertension, heart disease, joint problems. "
            "Management: caloric deficit (500-750 kcal/day), 150+ min exercise/week, "
            "behavioral therapy, GLP-1 agonists (semaglutide), bariatric surgery for severe cases.",
        "vaccination": "Vaccination Schedule (China):\n"
            "  • Birth: BCG (TB), HepB (first dose)\n"
            "  • 2 months: Polio (IPV)\n"
            "  • 3 months: DTaP, Polio (OPV)\n"
            "  • 8 months: Measles\n"
            "  • Adults: Annual influenza, Tdap booster every 10 years, "
            "Shingles (50+), Pneumococcal (65+), HPV (9-26 years).",
    }
    q = query.lower()
    for key, answer in knowledge_entries.items():
        if key in q:
            return f"📚 Medical Knowledge — {key}:\n\n{answer}"
    # Fallback: return all matching entries
    matches = []
    for key, answer in knowledge_entries.items():
        if any(word in q for word in key.split("_")) or any(word in key for word in q.split()):
            matches.append(f"📚 {key.replace('_', ' ').title()}:\n{answer}")
    if matches:
        return "\n\n".join(matches[:3])
    return f"🔍 No specific medical knowledge found for '{query}'. Try searching for: hypertension, diabetes, insomnia, migraine, asthma, depression, allergy, vaccination."


@tool
def search_drug_info(drug_name: str) -> str:
    """
    Get information about a specific medication.
    Args:
        drug_name: Name of the drug (e.g., 'metformin', 'ibuprofen', 'lisinopril').
    """
    drug_db = {
        "metformin": "💊 Metformin (Glucophage):\n"
            "  Class: Biguanide\n"
            "  Use: First-line treatment for Type 2 diabetes\n"
            "  Mechanism: Decreases liver glucose production, improves insulin sensitivity\n"
            "  Common dose: 500mg-2000mg/day\n"
            "  Side effects: GI upset, diarrhea, lactic acidosis (rare)\n"
            "  Monitoring: Renal function, B12 levels, HbA1c",
        "ibuprofen": "💊 Ibuprofen (Advil, Motrin):\n"
            "  Class: NSAID\n"
            "  Use: Pain relief, fever reduction, inflammation\n"
            "  Common dose: 200-400mg every 4-6 hours (max 1200mg/day OTC)\n"
            "  Side effects: GI bleeding, kidney issues with long-term use\n"
            "  Warning: Avoid with aspirin or other NSAIDs. Not for children <6 months.",
        "acetaminophen": "💊 Acetaminophen (Paracetamol, Tylenol):\n"
            "  Class: Analgesic/Antipyretic\n"
            "  Use: Pain and fever\n"
            "  Common dose: 325-1000mg every 4-6 hours (max 4000mg/day)\n"
            "  Side effects: Liver damage at high doses or with alcohol\n"
            "  Safe for most people, but watch total daily intake across all medications.",
        "lisinopril": "💊 Lisinopril (Zestril, Prinivil):\n"
            "  Class: ACE Inhibitor\n"
            "  Use: Hypertension, heart failure\n"
            "  Common dose: 5-40mg/day\n"
            "  Side effects: Dry cough, dizziness, elevated potassium\n"
            "  Monitoring: Blood pressure, renal function, potassium levels",
        "amoxicillin": "💊 Amoxicillin:\n"
            "  Class: Penicillin antibiotic\n"
            "  Use: Bacterial infections (respiratory, ear, urinary)\n"
            "  Common dose: 250-500mg every 8-12 hours\n"
            "  Side effects: GI upset, rash, yeast infections\n"
            "  Warning: Not effective for viral infections (cold, flu). Allergic reactions possible.",
        "omeprazole": "💊 Omeprazole (Prilosec):\n"
            "  Class: Proton Pump Inhibitor (PPI)\n"
            "  Use: GERD, stomach ulcers, heartburn\n"
            "  Common dose: 20-40mg/day\n"
            "  Side effects: Headache, GI upset, long-term: B12/magnesium deficiency\n"
            "  Take 30-60 min before meals for best effect.",
    }
    key = drug_name.lower().strip()
    if key in drug_db:
        return drug_db[key]
    # Partial match
    for dk, info in drug_db.items():
        if key in dk or dk in key:
            return info
    return f"🔍 Drug '{drug_name}' not in knowledge base. Always consult the medication leaflet or your pharmacist."
