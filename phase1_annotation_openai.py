"""
PHASE 1 — Annotation LLM pour détection de features dans les offres d'emploi
=============================================================================
Pipeline :
  1. Échantillonnage stratifié par domaine ROME (1ère lettre du code ROME)
  2. Annotation via OpenAI API (5 offres/appel → ~20 appels pour 100 offres)
  3. Checkpointing automatique (reprise possible en cas d'interruption)
  4. Rapport QC post-annotation

Features détectées (binaire 0/1) :
  - career_progression    : évolution, formation, promotion
  - company_culture       : valeurs, ambiance, diversité, bien-être
  - non_salary_benefits   : tickets-restaurant, mutuelle, CE, transport, congé maternité, sport
  - work_meaning_impact   : sens, impact social/environnemental, mission
  - schedule_flexibility  : télétravail, horaires flexibles, autonomie

Usage :
  export OPENAI_API_KEY="sk-..."

  python phase1_annotation_openai.py \
      --input data/JOCAS/jocas_clean.parquet \
      --output data/annotation/ \
      --n 100

Options utiles :
  --api-key YOUR_KEY          # sinon variable OPENAI_API_KEY
  --model gpt-5.4-nano        # moins cher pour classification/extraction simple
  --model gpt-5.4-mini        # défaut : meilleur compromis qualité/coût
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

try:
    from openai import (
        APIConnectionError,
        APIError,
        APITimeoutError,
        OpenAI,
        RateLimitError,
    )
except ImportError as exc:
    raise ImportError(
        "SDK OpenAI manquant ou trop ancien. Installe-le avec : "
        "pip install --upgrade openai"
    ) from exc
    
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

FEATURES = [
    "career_progression",
    "company_culture",
    "non_salary_benefits",
    "work_meaning_impact",
    "schedule_flexibility",
]

# Descriptions injectées dans le prompt pour guider le LLM
FEATURE_DEFINITIONS = {
    "career_progression": (
        "Mentions explicites de progression de carrière, évolution professionnelle, "
        "formation continue, montée en compétences, plan de carrière, promotions internes."
    ),
    "company_culture": (
        "Mentions de culture d'entreprise, valeurs organisationnelles, ambiance de travail, "
        "cohésion d'équipe, diversité et inclusion, bien-être au travail, engagement des employés."
    ),
    "non_salary_benefits": (
        "Mentions d'avantages non salariaux : tickets-restaurant, mutuelle/prévoyance, "
        "intéressement/participation, comité d'entreprise, voiture de fonction, "
        "remboursement transport, RTT supplémentaires, crèche, salle de sport, etc."
    ),
    "work_meaning_impact": (
        "Mentions du sens du travail, de l'impact social ou environnemental du poste, "
        "de la mission de l'entreprise, de l'utilité ou de la contribution à un projet porteur de sens."
    ),
    "schedule_flexibility": (
        "Mentions de flexibilité des horaires, télétravail, travail à distance, "
        "horaires aménagés ou variables, autonomie dans l'organisation du temps de travail."
    ),
}

# Modèle OpenAI par défaut : bon compromis qualité/coût pour annotation en batch.
ANNOTATION_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
BATCH_SIZE = 5            # offres par appel API
CHECKPOINT_EVERY = 50     # sauvegarder tous les N batches
MAX_RETRIES = 3           # tentatives en cas d'erreur API ou parsing
RETRY_DELAY = 5           # secondes entre tentatives
MAX_TEXT_CHARS = 3000     # tronquer les offres trop longues (économise des tokens)
MAX_OUTPUT_TOKENS = 2000  # marge confortable pour 5 annotations JSON


# ─────────────────────────────────────────────
# 1. STRATIFIED SAMPLING
# ─────────────────────────────────────────────

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
    text_col: str = "description_clean",
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


# ─────────────────────────────────────────────
# 2. PROMPT D'ANNOTATION + SCHÉMA JSON
# ─────────────────────────────────────────────

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
si les éléments suivants sont MENTIONNÉS (même implicitement) dans le texte.

## Définitions des features à détecter

{features_block}

## Règles d'annotation

- Label **1** = l'élément est mentionné explicitement ou clairement implicitement.
- Label **0** = l'élément n'est pas mentionné ou est trop vague pour être certain.
- Sois conservateur : en cas de doute, mets 0.
- Ne te base QUE sur le texte fourni, pas sur des suppositions sur l'entreprise.
- Retourne exactement une annotation par offre, en reprenant l'id exact fourni.

## Offres à annoter

{offers_block}

## Format de réponse attendu

Retourne uniquement un objet JSON avec une clé "annotations" contenant exactement {len(offers)} entrées.
Chaque entrée doit contenir : id, career_progression, company_culture, non_salary_benefits, work_meaning_impact, schedule_flexibility.
Chaque label doit valoir 0 ou 1.
"""
    return prompt


# ─────────────────────────────────────────────
# 3. ANNOTATION PIPELINE
# ─────────────────────────────────────────────

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
    text_col: str = "description_clean",
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


# ─────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    global ANNOTATION_MODEL
    parser = argparse.ArgumentParser(description="Phase 1 : annotation LLM des offres d'emploi via OpenAI")
    parser.add_argument("--input", required=True, help="Chemin vers le parquet source")
    parser.add_argument("--output", default="data/annotation/", help="Dossier de sortie")
    parser.add_argument("--n", type=int, default=100, help="Nombre d'offres à annoter")
    parser.add_argument("--text-col", default="description", help="Colonne texte")
    parser.add_argument("--rome-col", default="job_ROME_code", help="Colonne code ROME")
    parser.add_argument("--api-key", default=None, help="Clé API OpenAI (ou variable OPENAI_API_KEY)")
    parser.add_argument("--model", default=ANNOTATION_MODEL, help="Modèle OpenAI à utiliser")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    args = parser.parse_args()

    ANNOTATION_MODEL = args.model

    # ── Init client OpenAI
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Clé API manquante. Passe --api-key ou définis la variable OPENAI_API_KEY."
        )
    client = OpenAI(api_key=api_key)
    print(f"🔑 Client OpenAI initialisé — modèle : {ANNOTATION_MODEL}")

    # ── Charger les données
    print(f"📂 Chargement de {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"   {len(df):,} offres chargées")

    # ── Échantillonnage stratifié
    print(f"\n🎯 Échantillonnage de {args.n} offres (stratifié par domaine ROME)...")
    df_sample = stratified_sample(
        df,
        n=args.n,
        text_col=args.text_col,
        rome_col=args.rome_col,
        random_state=args.seed,
    )

    # ── Annotation
    df_labeled = run_annotation_pipeline(
        df_sample=df_sample,
        client=client,
        output_dir=Path(args.output),
        text_col=args.text_col,
    )

    # ── Rapport QC
    print_qc_report(df_labeled)

    final_output = Path(args.output) / "training_set_labeled.parquet"
    print(f"\n🎉 Pipeline terminé. Fichier de sortie : {final_output}")


if __name__ == "__main__":
    main()
