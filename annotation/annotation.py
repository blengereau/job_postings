"""
Détection de features dans les offres d'emploi
=============================================================================
Pipeline :
  1. Échantillonnage stratifié par domaine ROME (1ère lettre du code ROME)
  2. Annotation via OpenAI API (5 offres/appel → ~20 appels pour 100 offres)
  3. Checkpointing automatique (reprise possible en cas d'interruption)
  4. Rapport QC post-annotation

Features détectées (binaire 0/1) :
  - remote_work                   : télétravail possible
  - schedule_flexibility          : flexibilité dans le choix des horaires, hors télétravail
  - on_the_job_training           : formation/accompagnement reçu dans l'emploi
  - internal_career_progression   : évolution interne, promotion, mobilité
  - non_salary_benefits           : tickets-restaurant, mutuelle, CE, transport, congé maternité, sport
  - company_culture               : culture, ambiance, diversité, bien-être
  - work_meaning_impact           : sens, impact social/environnemental, mission
  - junior_offer                  : alternance, jeune diplômé, débutant, < 2 ans
  - experienced_offer             : expérience requise > 2 ans
  - manager_offer                 : leadership/management avec au moins 5 ans d'expérience
  - seniority_unclear             : niveau junior/expérimenté/manager impossible à catégoriser
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# 11 labels
FEATURES = [ 
    "remote_work",
    "schedule_flexibility",
    "on_the_job_training",
    "internal_career_progression",
    "company_culture",
    "non_salary_benefits",
    "work_meaning_impact",
    "junior_offer",
    "experienced_offer",
    "very_experienced_offer",
    "manager_offer",
    "programming_skills", # développement logiciels , code
    "communication_skills", # communication rédiger
    "creativity_skills", # innover créer
    "analysis_skills", # skills analytiques
]

# Descriptions injectées dans le prompt pour guider le LLM
FEATURE_DEFINITIONS = {
    "remote_work": (
        "Label 1 uniquement si l'offre mentionne explicitement que le télétravail est possible pour le poste : "
        "télétravail, remote, hybride, travail à distance, home office, jours de télétravail, poste partiellement à distance. "
        "Mettre 0 si le texte mentionne seulement autonomie, mobilité, déplacements, flexibilité, ou outils numériques sans possibilité explicite de télétravail. "
        "Mettre 0 si le télétravail concerne l'entreprise en général mais pas le poste proposé. "
        "Mettre 0 si l'offre indique explicitement que le poste est sur site, présentiel ou non télétravaillable."
    ),
    "schedule_flexibility": (
        "Label 1 uniquement si l'offre mentionne explicitement une flexibilité dans le choix ou l'organisation des horaires de travail : "
        "horaires flexibles, horaires aménageables, choix des horaires, liberté d'organisation du temps, planning adaptable, souplesse horaire. "
        "Ne pas inclure le télétravail ici : si l'offre mentionne seulement du télétravail sans flexibilité horaire, mettre schedule_flexibility = 0 et remote_work = 1. "
        "Mettre 0 si les horaires sont fixes ou imposés, même s'ils sont détaillés : 2x8, 3x8, horaires de nuit, 6h-14h, 8h-17h, shifts, planning défini, astreintes, heures supplémentaires, travail le week-end. "
        "Mettre 0 pour temps partiel, mi-temps, forfait jour ou autonomie si le texte ne dit pas clairement que le candidat peut choisir ou adapter ses horaires. "
        "Mettre 0 si la flexibilité concerne la gestion de projets, les délais, la polyvalence ou l'adaptabilité attendue du candidat."
    ),
    "on_the_job_training": (
        "Label 1 uniquement si l'offre mentionne une formation, certification, parcours d'intégration, accompagnement ou montée en compétences offert au candidat dans le cadre du poste. "
        "Exemples valides : formation interne, formation à la prise de poste, parcours d'intégration, tutorat, mentorat, accompagnement, certification financée, montée en compétences. "
        "Ne pas mettre 1 si la formation est seulement une compétence ou un diplôme requis. "
        "Ne pas mettre 1 si le candidat doit former, encadrer ou accompagner d'autres personnes : c'est une mission du poste, pas une formation reçue. "
        "Ne pas mettre 1 pour les mentions générales d'un cabinet de recrutement ou d'un organisme qui accompagne ses candidats, sauf si l'avantage est clairement rattaché au poste proposé. "
        "Ne pas mettre 1 pour l'évolution de carrière sans mention de formation ou d'apprentissage reçu : cela relève de internal_career_progression."
    ),
    
    "internal_career_progression": (
        "Label 1 uniquement si l'offre mentionne une possibilité concrète d'évolution dans l'entreprise ou le groupe : promotion, évolution hiérarchique, mobilité interne, passerelle vers d'autres postes, parcours de carrière, prise de responsabilités future. "
        "Ne pas mettre 1 si le texte mentionne seulement des missions responsabilisantes, un poste stimulant, une entreprise en croissance, ou des projets intéressants. "
        "Ne pas mettre 1 si l'évolution concerne des tiers, des clients, des élèves, des bénéficiaires, ou l'activité générale d'un cabinet de recrutement. "
        "Ne pas mettre 1 si le texte mentionne uniquement de la formation ou de la montée en compétences sans perspective d'évolution de poste ou de carrière : cela relève de on_the_job_training."
    ),
    
    # "career_progression": (
    #     "Label 1 uniquement si l'offre indique que le poste offre au candidat une opportunité concrète de développement professionnel : "
    #     "formation reçue par le candidat, certification, montée en compétences, "
    #     "accompagnement à la prise de poste, évolution de carrière, mobilité interne, promotion, parcours d'évolution lié au poste. "
    #     "Mettre 0 si le texte mentionne seulement les compétences requises, le diplôme attendu, l'expérience souhaitée, "
    #     "les qualités du candidat, des missions responsabilisantes, un métier stimulant ou des projets intéressants. "
    #     "Mettre 0 si le candidat doit former, accompagner ou sensibiliser d'autres personnes : c'est une mission du poste. "
    #     "Mettre 0 si la phrase décrit l'activité d'un cabinet de recrutement ou un service général de suivi de carrière. Par exemple mettre 0 "
    #     "si l'offre fait mention d'un cabinet de recrutement qui propose un suivi de carrière personnalisé."
        
    #     "Mettre 0 si la progression concerne des tiers (clients d'un cabinet de recrutements, enfants, élèves). "
    #     "Mettre 0 si l'information est générale et non spécifique au poste (ex : 'nous accompagnons nos candidats dans leur carrière')."
    # ),
    "company_culture": (
        "Label 1 uniquement si l'offre décrit explicitement l'environnement de travail du poste proposé au candidat : "
        "culture d'entreprise vécue, ambiance de travail, esprit d'équipe, convivialité, collaboration, bienveillance, "
        "diversité et inclusion, environnement motivant, etc... "
        "Ne pas mettre 1 si le texte décrit uniquement l'entreprise de manière générale (présentation institutionnelle, marketing)."
        "Ne pas mettre 1 si l'annonce décrit seulement pour les qualités attendues du candidat : autonomie, rigueur, curiosité, etc "
        "Ne pas mettre 1 pour une description simple de la taille de l'équipe, de l'entreprise."
    ),
    "non_salary_benefits": (
        "Label 1 uniquement si l'offre mentionne un avantage matériel concret qui s'ajoute à la rémunération directe. Exemples valides : tickets ou carte restaurant"
        "mutuelle, RTT, CSE/CE, chèques vacances, prise en charge transport, voiture de fonction, véhicule de service utilisable,"
        "crèche, salle de sport, logement, aide au déménagement, avantages sociaux clairement identifiés. "
        "Ne pas mettre 1 pour les mentions de salaire, rémunération, primes, bonus, commissions, variables, pourcentage du chiffre d'affaires, "
        "13e mois, package salarial ou rémunération attractive : ce sont des éléments de rémunération, pas des avantages non salariaux. "
        "Ne pas inclure le télétravail ici : il est annoté séparément dans remote_work. "
        "Ne pas mettre 1 pour formation, certification, parcours d'intégration, accompagnement, évolution de carrière, projets stimulants, "
        "équipe dynamique, ambiance conviviale, environnement de travail, autonomie, responsabilités ou section intitulée 'Nos avantages' si aucun avantage matériel/social concret n'est mentionné. "
    ),
    "work_meaning_impact": (
        "Label 1 uniquement si l'offre mentionne explicitement le sens, l'utilité ou l'impact du travail : "
        "impact social, impact environnemental, mission d'intérêt général, contribution à la société, transition écologique, "
        "inclusion, santé, éducation, service public, aide aux personnes, projet porteur de sens ou utilité sociale claire. "
        "Ne pas mettre 1 pour une simple description du produit, de l'activité commerciale, de la croissance de l'entreprise "
        "ou du fait de servir des clients, sauf si le texte affirme clairement une finalité sociale, environnementale."
    ),
    "junior_offer": (
        "Label 1 uniquement si l'offre cible explicitement un profil junior, débutant ou sans expérience professionnelle requise : "
        "alternance, apprentissage, stage, jeune diplômé, premier emploi, débutant accepté, profil junior, aucune expérience requise, première expérience acceptée. "
        "Mettre 1 si l'offre indique 0 à 2 ans d'expérience maximum, ou si elle indique clairement que les débutants sont acceptés. "
        "Mettre 0 si l'offre demande clairement une expérience professionnelle préalable dans un poste, métier, secteur ou fonction similaire. "
        "Mettre 0 si l'expérience est présentée comme nécessaire, même si la durée n'est pas précisée. "
    ),
    "experienced_offer": (
        "Label 1 uniquement si l'offre requiert explicitement une expérience professionnelle préalable dans un poste, métier, secteur ou fonction similaire. "
        "Exemples valides : expérience requise, expérience exigée, expérience réussie, expérience confirmée, expérience significative, solide expérience, expérience sur un poste similaire, première expérience dans le métier ou le secteur, plusieurs années d'expérience, 2 ans d'expérience ou plus. "
        "Mettre 1 même si l'offre demande un profil très expérimenté ou senior : dans ce cas very_experienced_offer peut aussi valoir 1. "
        "Mettre 0 si l'offre indique débutant accepté, alternance, stage, jeune diplômé, premier emploi, aucune expérience requise, première expérience acceptée ou 0 à 2 ans d'expérience maximum. "
        "Ne pas mettre 1 si l'expérience est seulement souhaitée, optionnelle, vague ou non discriminante, par exemple : 'une première expérience serait un plus'"
    ),
    "very_experienced_offer": (
        "Label 1 uniquement si l'offre requiert explicitement un profil très expérimenté, très senior ou de haut niveau. "
        "Exemples valides : au moins 10 ans d'expérience dans le secteur ou dans un poste similaire, profil très senior, expert confirmé, partner, associé, directeur, cadre dirigeant, leadership stratégique, pilotage d'une direction. "
        "Mettre 1 seulement si le texte indique clairement un niveau d'expérience ou de séniorité supérieur à une simple expérience professionnelle. "
        "Mettre 0 pour une simple demande d'expérience, même de plusieurs années, si elle ne signale pas un niveau très senior. "
        "Mettre 0 si le texte mentionne seulement 3 à 5 ans d'expérience, ou une expérience significative sans signal explicite de très forte séniorité. "
        "Si very_experienced_offer = 1, alors experienced_offer doit aussi valoir 1."
    ),
    "manager_offer": (
        "Label 1 uniquement si l'offre correspond clairement à un poste avec responsabilités de management, leadership ou encadrement d'équipe. "
        "Exemples valides : manager une équipe, encadrer des collaborateurs, superviser une équipe, responsabilité hiérarchique, animation d'équipe, coordination / pilotage / leadership d'équipe. "
        "Mettre 1 si le poste implique une responsabilité managériale claire, même si aucune durée d'expérience n'est indiquée. "
        "Mettre 0 si le candidat doit seulement gérer des projets, des clients, des dossiers sans encadrement clair d'équipe. "
        "Ce label est indépendant des labels de séniorité et ne regarde que la qualité managériales du poste. "
    )
}

ANNOTATION_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")
BATCH_SIZE = 5            # offres par appel API
CHECKPOINT_EVERY = 50     # sauvegarder tous les N batches
MAX_RETRIES = 3           # tentatives en cas d'erreur API ou parsing
RETRY_DELAY = 5           # secondes entre tentatives
MAX_TEXT_CHARS = 5000     # tronquer les offres trop longues (économise des tokens)
MAX_OUTPUT_TOKENS = 3500  


# 1. STRATIFIED SAMPLING

def extract_rome_domain(code: str) -> str:
    """
    Extrait la première lettre du code ROME (domaine professionnel).
    Ex : 'H1206' → 'H'  |  NaN/invalide → 'UNKNOWN'
    """
    if not isinstance(code, str) or not re.match(r"^[A-Z]\d{4}$", code.strip()):
        return "UNKNOWN"
    return code.strip()[0]


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    text_col: str = "description",
    rome_col: str = "job_ROME_code",
    min_text_len: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Échantillonnage stratifié par domaine ROME (lettre initiale).

    Stratégie :
      - On filtre les offres trop courtes (< min_text_len chars).
      - On attribue chaque offre à un stratum = lettre ROME (A-N + UNKNOWN).
      - On tire proportionnellement au nombre d'offres dans chaque stratum,
        avec un plancher par stratum pour garantir la représentation des codes rares.

    Returns :
      DataFrame de n lignes avec colonne 'rome_domain' ajoutée.
    """
    rng = np.random.default_rng(random_state)

    # Filtrer les offres trop courtes
    df = df.copy()
    df = df[df[text_col].fillna("").str.len() >= min_text_len].reset_index(drop=True)

    if len(df) < n:
        raise ValueError(
            f"Seulement {len(df)} offres valides après filtrage, "
            f"impossible d'en tirer {n}."
        )

    df["rome_domain"] = df[rome_col].apply(extract_rome_domain)

    # Calcul des quotas par stratum
    domain_counts = df["rome_domain"].value_counts()
    total = domain_counts.sum()
    quotas: dict[str, int] = {}
    remaining = n

    strata = domain_counts.index.tolist()

    # Passe 1 : on garantit un minimum par stratum
    min_per_stratum = 5
    for domain in strata:
        count = int(domain_counts[domain])
        guaranteed = min(min_per_stratum, count)
        quotas[domain] = guaranteed
        remaining -= guaranteed

    if remaining < 0:
        # Si le plancher dépasse déjà n, on réduit proportionnellement
        scale = n / sum(quotas.values())
        quotas = {k: max(1, int(v * scale)) for k, v in quotas.items()}
        remaining = 0

    # Passe 2 : on distribue le reste proportionnellement
    if remaining > 0:
        proportional_weights = {d: domain_counts[d] / total for d in strata}
        for domain in strata:
            extra = int(remaining * proportional_weights[domain])
            available = int(domain_counts[domain]) - quotas[domain]
            quotas[domain] += min(extra, available)

    # Ajustement final pour atteindre exactement n
    sampled_total = sum(quotas.values())
    diff = n - sampled_total
    if diff != 0:
        # Ajoute/retire au stratum le plus grand
        biggest = max(quotas, key=lambda k: domain_counts[k])
        quotas[biggest] += diff
        quotas[biggest] = max(0, quotas[biggest])

    # Tirage
    frames = []
    for domain, quota in quotas.items():
        if quota <= 0:
            continue
        pool = df[df["rome_domain"] == domain]
        k = min(quota, len(pool))
        sampled = pool.sample(n=k, random_state=int(rng.integers(0, 1_000_000)))
        frames.append(sampled)

    result = pd.concat(frames, ignore_index=True)

    # Log de la distribution
    print("\n📊 Distribution de l'échantillon par domaine ROME :")
    dist = result["rome_domain"].value_counts().sort_index()
    for domain, count in dist.items():
        pct = count / len(result) * 100
        print(f"   {domain} : {count:4d} offres ({pct:.1f}%)")
    print(f"   TOTAL : {len(result)} offres\n")

    return result


# 2. PROMPT D'ANNOTATION + SCHÉMA JSON

SYSTEM_PROMPT = """Tu es un assistant expert en analyse d'offres d'emploi françaises.
Ta tâche est d'annoter des offres d'emploi selon des critères précis.
Tu dois retourner uniquement un objet JSON conforme au schéma demandé.
"""


def annotation_response_format() -> dict[str, Any]:
    """
    Format JSON Schema pour Structured Outputs de l'API OpenAI.
    Le prompt vérifie le nombre exact d'offres ; la validation Python re-contrôle
    ensuite les IDs attendus, car JSON Schema ne doit pas porter la logique métier.
    """
    annotation_properties: dict[str, Any] = {
        "id": {"type": "string", "description": "Identifiant exact de l'offre annotée."}
    }
    for feat in FEATURES:
        annotation_properties[feat] = {
            "type": "integer",
            "enum": [0, 1],
            "description": f"Label binaire pour {feat}: 1 si présent, sinon 0.",
        }

    return {
        "format": {
            "type": "json_schema",
            "name": "job_offer_annotations",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "annotations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": annotation_properties,
                            "required": ["id", *FEATURES],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["annotations"],
                "additionalProperties": False,
            },
        }
    }


def build_annotation_prompt(offers: list[dict]) -> str:
    """
    Construit le prompt utilisateur pour annoter un batch d'offres.

    Args:
        offers : liste de dicts avec clés 'id' et 'text'

    Returns :
        str : prompt formaté
    """

    # Définitions des features
    features_block = "\n".join(
        f"  - **{feat}** : {desc}"
        for feat, desc in FEATURE_DEFINITIONS.items()
    )

    # Bloc des offres à annoter
    offers_block = ""
    for i, offer in enumerate(offers, 1):
        text = offer["text"][:MAX_TEXT_CHARS]  # tronquer si trop long
        offers_block += f"\n--- OFFRE {i} (id: {offer['id']}) ---\n{text}\n"

    prompt = f"""Analyse les offres d'emploi ci-dessous et détermine, pour chacune,
si les éléments suivants sont clairement MENTIONNÉS dans le texte.

## Définitions des features à détecter

{features_block}

## Règles d'annotation

- Label **1** = l'offre contient une preuve textuelle précise correspondant strictement à la définition du label.
- Si la preuve textuelle ne pourrait pas être citée mot pour mot, mets 0.
- Ne mets pas 1 à partir d'un simple ton positif, d'une section intitulée "Nos avantages", ou d'une supposition.
- Label **0** = l'élément n'est pas mentionné ou est trop vague pour être certain.
- Sois conservateur : en cas de doute, mets 0.
- Ne te base QUE sur le texte fourni, pas sur des suppositions sur l'entreprise.
- Retourne exactement une annotation par offre, en reprenant l'id exact fourni.
- Un titre de section ne suffit jamais à mettre un label à 1. Par exemple, une section "Nos avantages" ne justifie pas non_salary_benefits si elle ne liste pas un avantage matériel/social concret.
- Ne pas annoter comme avantage du poste les descriptions générales d'un cabinet de recrutement, de son activité ou de son accompagnement des candidats.
- Pour chaque label, vérifier que l'information concerne le candidat et le poste proposé. Si l'information concerne des tiers (clients, enfants, élèves, entreprise en général), ne pas attribuer le label.
- Pour chaque label, demander : "Est-ce que cette information décrit concrètement ce que vivra le candidat dans ce poste ?" Si non,  mets 0.

Pour les labels de séniorité :
- junior_offer et experienced_offer sont mutuellement exclusifs.
- Une offre ne peut pas avoir junior_offer = 1 et experienced_offer = 1 en même temps.
- Si l'offre ne donne aucune information claire sur l'expérience requise, laisser junior_offer = 0 et experienced_offer = 0. Ne pas créer de label positif à partir d'une simple absence d'information.

Pour very_experienced_offer :
- very_experienced_offer est un sous-label de experienced_offer.
- Si very_experienced_offer = 1, alors experienced_offer doit aussi valoir 1.
- very_experienced_offer ne doit jamais valoir 1 si experienced_offer = 0.

## Offres à annoter

{offers_block}

## Format de réponse attendu

Retourne uniquement un objet JSON avec une clé "annotations" contenant exactement {len(offers)} entrées.
Chaque entrée doit contenir : id, career_progression, company_culture, non_salary_benefits, work_meaning_impact, schedule_flexibility.
Chaque label doit valoir 0 ou 1.
"""
    return prompt


# 3. ANNOTATION PIPELINE

def parse_annotation_response(response_text: str, expected_ids: list[str]) -> list[dict] | None:
    """
    Parse la réponse JSON du LLM.

    Returns :
      Liste de dicts d'annotations, ou None si le parsing échoue.
    """
    # Extraire le JSON même si le modèle ajoute du texte autour
    match = re.search(r"\{[\s\S]*\}", response_text)
    if not match:
        return None

    try:
        data = json.loads(match.group())
        annotations = data.get("annotations", [])

        # Validation : nombre, présence des IDs, absence de doublons
        if len(annotations) != len(expected_ids):
            return None

        expected_set = set(map(str, expected_ids))
        normalized_annotations: dict[str, dict] = {}

        for ann in annotations:
            if "id" not in ann:
                return None

            ann_id = str(ann["id"])
            if ann_id not in expected_set or ann_id in normalized_annotations:
                return None

            normalized = {"id": ann_id}
            for feat in FEATURES:
                if feat not in ann:
                    return None
                normalized[feat] = int(bool(ann[feat]))  # normaliser en 0/1

            normalized_annotations[ann_id] = normalized

        # On retourne dans l'ordre attendu pour faciliter les audits manuels
        return [normalized_annotations[str(offer_id)] for offer_id in expected_ids]

    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def response_text_from_openai(response: Any, verbose: bool = False) -> str | None:
    """
    Extrait le texte utile d'une réponse OpenAI Responses API.
    Gère les cas incomplets ou refusés pour éviter de parser du vide.
    """
    status = getattr(response, "status", None)
    if status == "incomplete":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", "raison inconnue")
        if verbose:
            print(f"\n⚠️  Réponse OpenAI incomplète : {reason}")
        return None

    # Chemin standard du SDK OpenAI Responses API
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    # Fallback défensif si la forme de l'objet SDK évolue
    try:
        first_content = response.output[0].content[0]
        content_type = getattr(first_content, "type", None)
        if content_type == "refusal":
            if verbose:
                refusal = getattr(first_content, "refusal", "refus sans détail")
                print(f"\n⚠️  Réponse refusée par le modèle : {refusal}")
            return None
        if content_type == "output_text":
            return getattr(first_content, "text", None)
    except (AttributeError, IndexError, TypeError):
        pass

    return None


def annotate_batch(
    client: OpenAI,
    batch: list[dict],
    verbose: bool = False,
) -> list[dict] | None:
    """
    Envoie un batch d'offres au LLM et retourne les annotations.

    Args:
        client  : client OpenAI
        batch   : liste de dicts {'id': ..., 'text': ...}
        verbose : afficher les erreurs de parsing

    Returns :
        Liste d'annotations ou None si toutes les tentatives échouent.
    """
    prompt = build_annotation_prompt(batch)
    expected_ids = [str(o["id"]) for o in batch]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.create(
                model=ANNOTATION_MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=MAX_OUTPUT_TOKENS,
                text=annotation_response_format(),
            )
            raw_text = response_text_from_openai(response, verbose=verbose)
            if raw_text is None:
                if verbose:
                    print(f"\n⚠️  Aucune sortie textuelle exploitable (tentative {attempt+1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
                continue

            annotations = parse_annotation_response(raw_text, expected_ids)

            if annotations is not None:
                return annotations

            if verbose:
                print(f"\n⚠️  Parsing échoué (tentative {attempt+1}/{MAX_RETRIES})")
                print(f"   Réponse brute : {raw_text[:300]}")

        except RateLimitError:
            wait = RETRY_DELAY * (attempt + 1) * 2
            print(f"\n⏳ Rate limit OpenAI, attente {wait}s...")
            time.sleep(wait)

        except (APITimeoutError, APIConnectionError) as e:
            wait = RETRY_DELAY * (attempt + 1)
            print(f"\n🌐 Erreur réseau OpenAI ({type(e).__name__}), attente {wait}s...")
            time.sleep(wait)

        except APIError as e:
            status_code = getattr(e, "status_code", "?")
            message = getattr(e, "message", str(e))
            print(f"\n❌ Erreur API OpenAI ({status_code}): {message}")
            time.sleep(RETRY_DELAY)

    return None  # Toutes les tentatives ont échoué


def run_annotation_pipeline(
    df_sample: pd.DataFrame,
    client: OpenAI,
    output_dir: Path,
    text_col: str = "description",
    id_col: str | None = None,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> pd.DataFrame:
    """
    Pipeline principal d'annotation.

    - Reprend depuis le dernier checkpoint si disponible.
    - Sauvegarde un checkpoint toutes les `checkpoint_every` batches.
    - Retourne le DataFrame annoté (avec les offres non annotées marquées NaN).

    Args:
        df_sample      : DataFrame des offres échantillonnées
        client         : client OpenAI
        output_dir     : dossier de sauvegarde
        text_col       : colonne du texte de l'offre
        id_col         : colonne identifiant unique (utilise l'index si None)
        checkpoint_every : fréquence de sauvegarde
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_annotations_openai.jsonl"
    final_path = output_dir / "training_set_labeled.parquet"

    # ── Utiliser l'index comme identifiant si pas de colonne id
    df_sample = df_sample.copy().reset_index(drop=True)
    if id_col is None:
        df_sample["_annotation_id"] = df_sample.index.astype(str)
        id_col = "_annotation_id"
    else:
        df_sample["_annotation_id"] = df_sample[id_col].astype(str)
        id_col = "_annotation_id"

    # ── Charger le checkpoint (reprendre là où on s'était arrêté)
    already_done = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ann = json.loads(line)
                    already_done[str(ann["id"])] = ann
        print(f"♻️  Checkpoint trouvé : {len(already_done)} offres déjà annotées")

    # ── Identifier les offres restantes
    remaining = df_sample[~df_sample[id_col].isin(already_done.keys())]
    print(f"📋 {len(remaining)} offres à annoter ({len(already_done)} déjà faites)")

    if len(remaining) == 0:
        print("✅ Toutes les offres sont déjà annotées !")
    else:
        # ── Préparer les batches
        rows = [
            {"id": str(row[id_col]), "text": str(row[text_col])}
            for _, row in remaining.iterrows()
        ]
        batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]

        failed_ids = []
        batch_checkpoint_counter = 0

        with tqdm(total=len(batches), desc="🤖 Annotation OpenAI") as pbar:
            with open(checkpoint_path, "a", encoding="utf-8") as ckpt_file:

                for batch in batches:
                    annotations = annotate_batch(client, batch, verbose=False)

                    if annotations is not None:
                        for ann in annotations:
                            ckpt_file.write(json.dumps(ann, ensure_ascii=False) + "\n")
                            already_done[str(ann["id"])] = ann
                    else:
                        # Marquer les offres de ce batch comme échouées
                        for item in batch:
                            failed_ids.append(item["id"])
                        tqdm.write(
                            f"⚠️  Batch de {len(batch)} offres échoué "
                            f"(IDs: {[b['id'] for b in batch]})"
                        )

                    batch_checkpoint_counter += 1

                    # Flush du checkpoint à intervalles réguliers
                    if batch_checkpoint_counter % checkpoint_every == 0:
                        ckpt_file.flush()
                        tqdm.write(f"💾 Checkpoint sauvegardé ({len(already_done)} offres annotées)")

                    pbar.update(1)

        if failed_ids:
            print(f"\n⚠️  {len(failed_ids)} offres n'ont pas pu être annotées : {failed_ids[:10]}...")

    # ── Assembler le DataFrame final
    annotations_list = list(already_done.values())
    df_annotations = pd.DataFrame(annotations_list).rename(columns={"id": "_annotation_id"})

    # S'assurer que toutes les features sont présentes, même si aucune annotation n'a abouti
    if "_annotation_id" not in df_annotations.columns:
        df_annotations["_annotation_id"] = pd.Series(dtype=str)
    for feat in FEATURES:
        if feat not in df_annotations.columns:
            df_annotations[feat] = np.nan

    # Merger avec le DataFrame source
    df_result = df_sample.merge(df_annotations, on="_annotation_id", how="left")

    # Sauvegarder
    df_result.to_parquet(final_path)
    print(f"\n✅ Training set sauvegardé : {final_path}")
    print(f"   Lignes totales     : {len(df_result)}")
    print(f"   Offres annotées    : {df_annotations['_annotation_id'].nunique()}")
    print(f"   Offres sans labels : {df_result[FEATURES[0]].isna().sum()}")

    return df_result


# ─────────────────────────────────────────────
# 4. RAPPORT QC POST-ANNOTATION
# ─────────────────────────────────────────────

def print_qc_report(df: pd.DataFrame) -> None:
    """
    Affiche un rapport de qualité de l'annotation :
      - Distribution des labels par feature
      - Taux de co-occurrence entre features
      - Offres sans aucun label positif
    """
    df_ann = df[FEATURES].dropna()
    n = len(df_ann)

    if n == 0:
        print("⚠️  Aucune offre annotée, pas de rapport QC disponible.")
        return

    print("\n" + "=" * 60)
    print("📊 RAPPORT QC — ANNOTATION")
    print("=" * 60)

    print(f"\n{'Feature':<28} {'N(1)':>6}  {'%(1)':>7}  {'N(0)':>6}  {'%(0)':>7}")
    print("-" * 60)
    for feat in FEATURES:
        n1 = int(df_ann[feat].sum())
        n0 = n - n1
        p1 = n1 / n * 100
        p0 = n0 / n * 100
        print(f"  {feat:<26} {n1:>6}  {p1:>6.1f}%  {n0:>6}  {p0:>6.1f}%")

    # Offres sans aucun label positif
    zero_rows = (df_ann.sum(axis=1) == 0).sum()
    print(f"\n  Offres sans aucun label positif : {zero_rows} ({zero_rows/n*100:.1f}%)")

    # Co-occurrences (top paires)
    print("\n  Co-occurrences les plus fréquentes :")
    pairs = []
    for i, f1 in enumerate(FEATURES):
        for f2 in FEATURES[i + 1:]:
            cooc = int((df_ann[f1] & df_ann[f2]).sum())
            pairs.append((f1, f2, cooc))
    pairs.sort(key=lambda x: -x[2])
    for f1, f2, cooc in pairs[:5]:
        print(f"    {f1} × {f2} : {cooc} offres")

    # Distribution par domaine ROME (si disponible)
    if "rome_domain" in df.columns:
        print("\n  Taux de labels positifs par domaine ROME :")
        for domain in sorted(df["rome_domain"].unique()):
            sub = df[df["rome_domain"] == domain][FEATURES].dropna()
            if len(sub) == 0:
                continue
            mean_labels = sub.mean().mean() * 100
            print(f"    {domain} : {len(sub):4d} offres, {mean_labels:.1f}% labels positifs en moy.")

    print("\n" + "=" * 60)


# 5. ENTRY POINT
def main() -> None:
    global ANNOTATION_MODEL
    parser = argparse.ArgumentParser(description="Pipeline d'annotation de données")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", default="data/annotation/")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--text-col", default="description")
    parser.add_argument("--rome-col", default="job_ROME_code")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=ANNOTATION_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ANNOTATION_MODEL = args.model

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    print(f"Client OpenAI initialisé — modèle : {ANNOTATION_MODEL}")

    print(f"Chargement de {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"   {len(df):,} offres chargées")

    print(f"Échantillonnage de {args.n} offres (stratifié par domaine ROME)...")
    df_sample = stratified_sample(
        df,
        n=args.n,
        text_col=args.text_col,
        rome_col=args.rome_col,
        random_state=args.seed,
    )
    df_labeled = run_annotation_pipeline(
        df_sample=df_sample,
        client=client,
        output_dir=Path(args.output),
        text_col=args.text_col,
    )
    print_qc_report(df_labeled)

    final_output = Path(args.output) / "training_set_labeled.parquet"
    print(f"\n🎉 Pipeline terminé. Fichier de sortie : {final_output}")


if __name__ == "__main__":
    main()
