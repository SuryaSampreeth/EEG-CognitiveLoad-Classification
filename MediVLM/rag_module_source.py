"""
AMG-RAG: Clinical Report Hallucination Detection System
5-Step Pipeline:
  1. Report Ingestion & Extraction
  2. LLM Baseline Check (typical / atypical)
  3. PubMed Grounding — only if atypical
  4. MKG Contradiction Engine
  5. Final Verdict: Hallucinated / Not Hallucinated with confidence score
"""

import time
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import networkx as nx
from langchain_google_genai import ChatGoogleGenerativeAI  # pyright: ignore[reportMissingImports]
from langchain_core.prompts import PromptTemplate  # pyright: ignore[reportMissingImports]
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser  # pyright: ignore[reportMissingImports]
import requests
from xml.etree import ElementTree as ET
from decouple import config  # pyright: ignore[reportMissingImports]

# Configuration
GOOGLE_API_KEY = config('GOOGLE_API_KEY', default=os.environ.get('GOOGLE_API_KEY'))
PUBMED_API_KEY = config('pubmed_api', default=os.environ.get('pubmed_api') or None)


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────
@dataclass
class MedicalEntity:
    """Represents a medical entity in the knowledge graph"""
    name: str
    description: str
    entity_type: str
    confidence: float = 1.0
    sources: List[str] = field(default_factory=list)

@dataclass
class MedicalRelation:
    """Represents a relationship between medical entities"""
    source: str
    target: str
    relation_type: str
    confidence: float
    evidence: str
    sources: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# Medical Knowledge Graph
# ──────────────────────────────────────────────
class MedicalKnowledgeGraph:
    """Dynamic Medical Knowledge Graph with confidence scoring"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities = {}
        self.relations = []

    def add_entity(self, entity: MedicalEntity):
        self.entities[entity.name] = entity
        self.graph.add_node(
            entity.name,
            description=entity.description,
            entity_type=entity.entity_type,
            confidence=entity.confidence,
            sources=entity.sources,
        )

    def add_relation(self, relation: MedicalRelation):
        self.relations.append(relation)
        self.graph.add_edge(
            relation.source,
            relation.target,
            relation_type=relation.relation_type,
            confidence=relation.confidence,
            evidence=relation.evidence,
            sources=relation.sources,
        )

    def get_connected_nodes(self, node_name: str, confidence_threshold: float = 0.0):
        connected = []
        if node_name in self.graph:
            for neighbor in self.graph.neighbors(node_name):
                edge_data = self.graph[node_name][neighbor]
                if edge_data.get("confidence", 0) >= confidence_threshold:
                    connected.append(
                        {
                            "node": neighbor,
                            "relation": edge_data.get("relation_type"),
                            "confidence": edge_data.get("confidence"),
                            "evidence": edge_data.get("evidence"),
                        }
                    )
        return connected


# ──────────────────────────────────────────────
# PubMed Searcher
# ──────────────────────────────────────────────
class PubMedSearcher:
    """PubMed API wrapper for medical literature search"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def search(self, query: str, max_results: int = 3) -> List[str]:
        time.sleep(0.5)
        search_url = f"{self.base_url}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "xml",
            "retmax": max_results,
        }
        if self.api_key:
            search_params["api_key"] = self.api_key

        try:
            response = requests.get(search_url, params=search_params, timeout=30)
            if response.status_code != 200:
                return []
            if not response.text.strip().startswith("<"):
                return []

            root = ET.fromstring(response.text)
            pmids = [id_elem.text for id_elem in root.findall(".//Id")]
            if not pmids:
                return []

            time.sleep(0.5)
            fetch_url = f"{self.base_url}/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "text",
                "rettype": "abstract",
            }
            if self.api_key:
                fetch_params["api_key"] = self.api_key

            response = requests.get(fetch_url, params=fetch_params, timeout=30)
            if response.status_code != 200:
                return []

            articles = response.text.split("\n\n")
            abstracts = []
            for article in articles:
                lines = article.split("\n")
                abstract_lines = [
                    line
                    for line in lines
                    if line.strip()
                    and not any(
                        skip in line.lower()
                        for skip in ["author", "doi", "pmid", "copyright"]
                    )
                ]
                if abstract_lines:
                    abstracts.append(" ".join(abstract_lines))
            return abstracts

        except Exception as e:
            print(f"  PubMed search error: {e}")
            return []


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _align_lists(*lists, fill="N/A"):
    """Force all extracted arrays to the same length so index-based
    alignment (findings[i] <-> anatomical_targets[i] <-> ...) can't
    silently drift when the LLM returns mismatched array lengths."""
    max_len = max((len(l) for l in lists), default=0)
    return [list(l) + [fill] * (max_len - len(l)) for l in lists]


def _normalize_contradiction_pairs(raw_pairs, num_findings):
    """The LLM is asked for pairs like [0, 2] but the scoring logic needs
    a list of index-lists, e.g. [[0, 2], [1, 3]]. Handle both shapes and
    drop anything that doesn't parse as a valid pair of in-range indices."""
    normalized = []
    if not raw_pairs:
        return normalized

    # Case: flat list like [0, 2, 1, 3] -> treat as consecutive pairs
    if raw_pairs and all(isinstance(x, (int, float)) for x in raw_pairs):
        it = iter(raw_pairs)
        raw_pairs = [[a, b] for a, b in zip(it, it)]

    for pair in raw_pairs:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            try:
                a, b = int(pair[0]), int(pair[1])
            except (TypeError, ValueError):
                continue
            if 0 <= a < num_findings and 0 <= b < num_findings:
                normalized.append([a, b])
    return normalized


# ──────────────────────────────────────────────
# Main System
# ──────────────────────────────────────────────
class AMG_RAG_ReportSystem:
    """AMG-RAG system for Clinical Report Hallucination Detection"""

    def __init__(self, google_api_key: str = None):
        if google_api_key:
            self.llm = ChatGoogleGenerativeAI(
                model="gemma-4-31b-it",
                temperature=0.0,
                google_api_key=google_api_key,
            )
        else:
            raise ValueError("Google API key is required.")

        self.kg = MedicalKnowledgeGraph()
        self.pubmed = PubMedSearcher(api_key=PUBMED_API_KEY)
        self._setup_chains()

    # ──────────────────────────────────────────
    # Chain Setup
    # ──────────────────────────────────────────
    def _setup_chains(self):
        """Setup LLM chains for the 5-step report pipeline"""

        # ── Step 1 chain: Report Ingestion & Extraction ──
        finding_schemas = [
            ResponseSchema(
                name="findings",
                description="List of clinical claims/findings parsed from the report",
                type="array",
            ),
            ResponseSchema(
                name="anatomical_targets",
                description="The anatomical structures associated with each finding",
                type="array",
            ),
            ResponseSchema(
                name="clinical_status",
                description="Clinical observation status for each finding (e.g. 'normal', 'abnormal', 'clear')",
                type="array",
            ),
            ResponseSchema(
                name="attribute_type",
                description=(
                    "The specific clinical dimension this finding measures, e.g. 'inspiratory_volume',"
                    " 'structural_clarity', 'opacity', 'size', 'vascularity', 'effusion', 'pneumothorax_presence'."
                    " Two findings about the same anatomical structure can only contradict each other"
                    " if they share the same attribute_type."
                ),
                type="array",
            ),
        ]
        finding_parser = StructuredOutputParser.from_response_schemas(finding_schemas)

        self.finding_extractor = (
            PromptTemplate(
                template="""Extract all individual medical findings/claims and associated anatomical structures from this clinical report.

                Report: {report}

                For each finding/claim, provide:
                1. The finding statement
                2. The anatomical target structure
                3. The clinical status (e.g. normal, abnormal, clear, consolidated, etc.)
                4. The attribute_type: the clinical dimension this finding refers to
                    (e.g. 'inspiratory_volume', 'structural_clarity', 'opacity', 'size', 'vascularity', 'effusion', 'pneumothorax_presence').

                IMPORTANT: The four output arrays (findings, anatomical_targets, clinical_status, attribute_type)
                MUST all be the same length and index-aligned — element i of each array must
                describe the same finding.

                Return in JSON format:
                {format_instructions}""",
                input_variables=["report"],
                partial_variables={
                    "format_instructions": finding_parser.get_format_instructions()
                },
            )
            | self.llm
            | finding_parser
        )

        # ── Step 2 chain: LLM Baseline Check ──
        baseline_schemas = [
            ResponseSchema(
                name="classifications",
                description="For each finding: 'typical' if a standard/expected clinical observation, 'atypical' if unusual, rare, or potentially suspicious",
                type="array",
            ),
            ResponseSchema(
                name="reasoning",
                description="Brief reasoning for each classification",
                type="array",
            ),
        ]
        baseline_parser = StructuredOutputParser.from_response_schemas(
            baseline_schemas
        )

        self.baseline_checker = (
            PromptTemplate(
                template="""You are a clinical radiology expert. For each finding below, classify it as either 'typical' or 'atypical'.

            DEFINITIONS:
            - 'typical': A standard, commonly seen clinical observation (e.g. 'heart is normal size', 'lungs are clear', 'no pleural effusion', 'mild cardiomegaly', 'pneumothorax').
            - 'atypical': A finding that is extremely rare, highly unusual, internally contradictory, physiologically impossible, or highly suspicious. This includes:
              a) Extremely rare congenital conditions (e.g. 'dextrocardia', 'situs inversus')
              b) Internally contradictory statements (e.g. 'normal size with severe cardiomegaly', 'clear lungs with massive consolidation')
              c) Impossible whole-organ counts (e.g. 'patient has 3 lungs')
              d) Impossible or incorrect counts of normal anatomical SUBSTRUCTURES — lobes, chambers,
                 valves, vessels, etc. — relative to known human anatomy. For example: the left lung
                 normally has 2 lobes and the right lung has 3 lobes, so "three lobes in the left lung"
                 is atypical/anatomically impossible. The heart normally has 4 chambers, so "five
                 chambers" would be atypical. Apply this same substructure-count check to any
                 anatomical structure mentioned, not just the examples given here.

            NOT ATYPICAL — do NOT flag these as atypical:
              e) Imprecise but commonly understood clinical language. Radiologists routinely use
                 terminology loosely when the clinical intent is clear and unambiguous.
                 For example: "aorta is clear without focal consolidation" — while "consolidation"
                 is technically a pulmonary-parenchyma term, radiologists commonly use it as shorthand
                 for "no focal abnormality" when describing the aorta or mediastinum. This is standard
                 radiology dictation style, NOT an atypical finding. Similarly, "clear" can be applied
                 to any structure to mean "unremarkable" or "without pathology", not just the lungs.
                 Only flag language as atypical if it is truly physiologically impossible or self-contradictory,
                 NOT merely because a term is being applied loosely to a neighbouring anatomical structure.
              f) Common abnormal findings (e.g. pleural effusion, tumors, cardiomegaly, atelectasis) are
                 still 'typical' — they are routine observations in clinical radiology.

            NOTE: Extremely rare congenital conditions, impossible/contradictory statements, and impossible substructure counts MUST be flagged as 'atypical'. But imprecise wording that is standard radiology dictation practice must remain 'typical'.

            Findings to classify:
            {findings_list}

            IMPORTANT: Return exactly one classification and one reasoning entry per finding listed above, in the same order.

            Return in JSON format:
            {format_instructions}""",
                input_variables=["findings_list"],
                partial_variables={
                    "format_instructions": baseline_parser.get_format_instructions()
                },
            )
            | self.llm
            | baseline_parser
        )

        # ── Step 4 chain: MKG Contradiction Engine ──
        contradiction_schemas = [
            ResponseSchema(
                name="contradictions",
                description="List of contradiction descriptions found between findings. Empty list if none found.",
                type="array",
            ),
            ResponseSchema(
                name="contradiction_pairs",
                description="List of [index_a, index_b] pairs indicating which findings are involved in each contradiction. For a contradiction between two different findings, use their two distinct indices, e.g. [0, 2]. For a contradiction involving only ONE finding (self-contradictory statement, or physiologically/anatomically impossible claim), use the SAME index twice, e.g. [5, 5]. Each element must be a 2-item list. Empty list if none.",
                type="array",
            ),
            ResponseSchema(
                name="severity_scores",
                description="Severity score (0.0 to 1.0) for each contradiction, same order/length as contradictions. 1.0 = definite contradiction, 0.5 = possible, 0.0 = no contradiction.",
                type="array",
            ),
        ]
        contradiction_parser = StructuredOutputParser.from_response_schemas(
            contradiction_schemas
        )

        self.contradiction_detector = (
                PromptTemplate(
                        template="""You are a clinical contradiction detection engine. Analyze the following clinical findings from a single medical report for internal contradictions.

                A CONTRADICTION occurs when:
                1. A contradiction (type 1) requires BOTH: (a) same anatomical_target, AND (b) same attribute_type, with opposing values. If attribute_type differs between two findings about the same structure, they CANNOT be a type-1 contradiction, regardless of how their wording sounds. Encode as [index_a, index_b] with two distinct indices.
                2. A single finding contains self-contradictory statements (e.g. 'clear lungs with massive consolidation'). Encode as [index, index] — the SAME index twice, since only one finding is involved.
                3. A finding describes something physiologically or anatomically impossible (e.g. 'patient has 3 lungs', 'three lobes in the left lung', 'five cardiac chambers'). Encode as [index, index] — the SAME index twice, since only one finding is involved.

                NOT a contradiction (these are normal, expected patterns in clinical radiology):
                1. Multiple abnormal findings in different regions (e.g. effusion in lungs + cardiomegaly in heart)
                2. Worsening or progression of a condition across serial reports
                3. Minor/trace findings coexisting with general "clear" or "no acute findings" statements
                        - In radiology convention, "clear lungs" means no consolidation/infiltrate/pneumonia, NOT absence of ALL findings
                        - Minor findings like mild basilar atelectasis, trace granulomas, or small nodules can coexist with "clear" statements
                        - This is standard practice and NOT a contradiction (e.g. "lungs are clear" + "mild basilar atelectasis" = NORMAL, not contradictory)
                4. Incidental findings mentioned alongside primary pathology (e.g. "no pneumonia" + "small granuloma noted")
                5. ACUITY MISMATCH RULE: "No acute findings" is a claim about the absence of new/emergent
                            pathology — it is NOT a claim that the study is entirely normal, and it does NOT
                            contradict a finding unless that finding is itself explicitly framed as acute, new,
                            or an interval change.

                            To evaluate whether a pairing is a contradiction, classify the abnormal finding's
                            acuity based on its own wording:
                            - Explicitly acute/new/interval-change language ("new," "acute," "increased from prior,"
                                "worsening," "interval development of") → CAN contradict "no acute findings"
                            - No acuity language, or language suggesting a chronic/structural/incidental process
                                ("enlarged," "prominent," "tortuous," "calcified," "stable," "chronic," "old,"
                                "longstanding") → does NOT contradict "no acute findings" on its own

                            Do not infer acuity from the finding's anatomical category or clinical severity —
                            infer it only from the acuity language actually present in that finding's text.
                            If acuity is ambiguous or unstated, do NOT flag it as a contradiction
                            (default to non-contradiction when acuity is unclear).

                            Examples (illustrative only, not exhaustive): enlarged pulmonary arteries, old rib
                            fracture, stable pulmonary nodule, chronic scarring, calcified granuloma.
                6. VOLUME vs. STRUCTURE DISTINCTION: "Low lung volumes" (or "hypoinflation," "poor
                        inspiratory effort," "low lung volumes") describes breath/inspiratory effort at
                        the time of imaging — it is NOT a claim about the shape or clarity of lung
                        structures. "Well-expanded silhouettes," "clear pulmonary silhouettes," or
                        "hilar/pulmonary contours well defined" describe structural appearance, not air
                        volume. These two attributes are independent and routinely coexist — do NOT treat
                        "low volume" + "well-expanded silhouette" as a contradiction. Only flag a true
                        contradiction if the SAME attribute (either volume or structural clarity) is
                        described with opposite values (e.g. "low lung volumes" AND "hyperinflated lungs"
                        would be a genuine volume-vs-volume contradiction).

                Findings:
                {findings_list}

                Knowledge Graph Context:
                {kg_context}

                Identify ONLY true logical contradictions. Each entry in contradiction_pairs MUST be
                a 2-item list of 0-based finding indices — [index_a, index_b] for a two-finding
                contradiction (type 1), or [index, index] for a single-finding contradiction
                (types 2 and 3). contradictions / contradiction_pairs / severity_scores MUST all be
                the same length, in matching order.

                Return in JSON format:
                {format_instructions}""",
                input_variables=["findings_list", "kg_context"],
                partial_variables={
                    "format_instructions": contradiction_parser.get_format_instructions()
                },
            )
            | self.llm
            | contradiction_parser
        )

    # ──────────────────────────────────────────
    # Main Pipeline
    # ──────────────────────────────────────────
    def evaluate_medical_report(self, report: str) -> Dict[str, Any]:
        """Execute the 5-step report hallucination detection pipeline."""

        # ════════════════════════════════════════
        # STEP 1: Report Ingestion & Extraction
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 1: Report Ingestion & Extraction")
        print("=" * 60)
        print(f"\n  Input Report:\n  \"{report}\"\n")

        try:
            finding_result = self.finding_extractor.invoke({"report": report})
            findings = finding_result.get("findings", []) or []
            anatomical_targets = finding_result.get("anatomical_targets", []) or []
            clinical_statuses = finding_result.get("clinical_status", []) or []
            attribute_types = finding_result.get("attribute_type", []) or []
        except Exception as e:
            print(f"  Finding extraction error: {e}")
            findings = [s.strip() for s in report.split(".") if s.strip()]
            anatomical_targets = ["General"] * len(findings)
            clinical_statuses = ["Unspecified"] * len(findings)
            attribute_types = ["Unspecified"] * len(findings)

        # FIXED: enforce index alignment instead of relying on ad-hoc
        # "if i < len(...)" guards scattered through the rest of the pipeline.
        findings, anatomical_targets, clinical_statuses, attribute_types = _align_lists(
            findings, anatomical_targets, clinical_statuses, attribute_types
        )

        print(f"  Extracted {len(findings)} clinical claims:\n")
        for i, f in enumerate(findings):
            print(f"    Claim {i+1}: \"{f}\"")
            print(f"             Target: {anatomical_targets[i]} | Status: {clinical_statuses[i]} | Attribute: {attribute_types[i]}")

        # ════════════════════════════════════════
        # STEP 2: LLM Baseline Check
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 2: LLM Baseline Check (typical / atypical)")
        print("=" * 60)

        findings_formatted = "\n".join(
            [
                f"  Finding {i+1}: \"{findings[i]}\" (Target: {anatomical_targets[i]}, Status: {clinical_statuses[i]}, Attribute: {attribute_types[i]})"
                for i in range(len(findings))
            ]
        )

        try:
            baseline_result = self.baseline_checker.invoke(
                {"findings_list": findings_formatted}
            )
            classifications = baseline_result.get("classifications", []) or []
            baseline_reasoning = baseline_result.get("reasoning", []) or []
        except Exception as e:
            print(f"  Baseline check error: {e}")
            classifications = []
            baseline_reasoning = []

        # FIXED: pad/align instead of silently defaulting everything to
        # "typical" on a partial or failed response.
        classifications, baseline_reasoning = _align_lists(
            classifications, baseline_reasoning, fill="typical"
        )

        atypical_indices = []
        for i in range(len(findings)):
            cls = classifications[i]
            reason = baseline_reasoning[i]
            marker = "[!] ATYPICAL" if cls.lower() == "atypical" else "[OK] TYPICAL"
            print(f"\n    Claim {i+1}: {marker}")
            print(f"      \"{findings[i]}\"")
            print(f"      Reasoning: {reason}")
            if cls.lower() == "atypical":
                atypical_indices.append(i)

        print(f"\n  Summary: {len(atypical_indices)} atypical / {len(findings)} total findings")

        # ════════════════════════════════════════
        # STEP 3: PubMed Grounding (only if atypical)
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 3: PubMed Grounding — only if atypical")
        print("=" * 60)

        pubmed_evidence = {}  # index -> list of abstracts
        if not atypical_indices:
            print("\n  No atypical findings detected. Skipping PubMed grounding.")
        else:
            for idx in atypical_indices:
                finding_text = findings[idx]
                target = anatomical_targets[idx]
                query = f"{target} {finding_text}"
                print(f"\n    Searching PubMed for Claim {idx+1}: \"{finding_text[:80]}...\"")
                abstracts = self.pubmed.search(query, max_results=2)
                pubmed_evidence[idx] = abstracts
                if abstracts:
                    print(f"      Found {len(abstracts)} relevant article(s)")
                    for j, abstract in enumerate(abstracts):
                        print(f"        Article {j+1}: {abstract[:120]}...")
                else:
                    print("      No PubMed articles found — finding remains ungrounded")

        # ════════════════════════════════════════
        # STEP 4: MKG Contradiction Engine
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 4: MKG Contradiction Engine")
        print("=" * 60)

        # Build Knowledge Graph from findings
        self.kg = MedicalKnowledgeGraph()

        print("\n  Building Medical Knowledge Graph from report claims...")
        for i, target in enumerate(anatomical_targets):
            entity_name = target.title()
            if entity_name not in self.kg.entities:
                med_entity = MedicalEntity(
                    name=entity_name,
                    description=f"Anatomical region: {entity_name}. Status: {clinical_statuses[i]}",
                    entity_type="anatomical_structure",
                    confidence=1.0,
                    sources=["Report Parser"],
                )
                self.kg.add_entity(med_entity)

            finding_name = f"Claim_{i+1}"
            finding_entity = MedicalEntity(
                name=finding_name,
                description=findings[i],
                entity_type="clinical_finding",
                confidence=1.0,
                sources=["Report Parser"],
            )
            self.kg.add_entity(finding_entity)

            relation = MedicalRelation(
                source=entity_name,
                target=finding_name,
                relation_type="has_finding",
                confidence=1.0,
                evidence=f"Status: {clinical_statuses[i]}",
                sources=["Report Parser"],
            )
            self.kg.add_relation(relation)

        print(f"    Nodes: {len(self.kg.entities)} | Edges: {len(self.kg.relations)}")

        # Print KG structure
        for entity_name in self.kg.entities.keys():
            connections = self.kg.get_connected_nodes(entity_name)
            if connections:
                for conn in connections:
                    print(f"    {entity_name} --[{conn['relation']}]--> {conn['node']}")

        # Run contradiction detection
        kg_context_lines = []
        for entity_name in self.kg.entities.keys():
            connections = self.kg.get_connected_nodes(entity_name)
            if connections:
                for conn in connections:
                    kg_context_lines.append(
                        f"{entity_name} --[{conn['relation']}]--> {conn['node']}: {self.kg.entities.get(conn['node'], MedicalEntity('','','',0,[])).description}"
                    )
        kg_context = "\n".join(kg_context_lines) if kg_context_lines else "No graph context available."

        print("\n  Running contradiction detection...")
        try:
            contradiction_result = self.contradiction_detector.invoke(
                {"findings_list": findings_formatted, "kg_context": kg_context}
            )
            contradictions = contradiction_result.get("contradictions", []) or []
            raw_pairs = contradiction_result.get("contradiction_pairs", []) or []
            severity_scores = contradiction_result.get("severity_scores", []) or []
        except Exception as e:
            print(f"  Contradiction detection error: {e}")
            contradictions = []
            raw_pairs = []
            severity_scores = []

        # FIXED: robustly normalize pair format instead of assuming the LLM
        # always returns a list of 2-item lists.
        contradiction_pairs = _normalize_contradiction_pairs(raw_pairs, len(findings))

        # Keep contradictions/severity_scores aligned to the (possibly
        # shrunk-by-validation) contradiction_pairs list.
        n = min(len(contradictions), len(contradiction_pairs)) if contradiction_pairs else 0
        contradictions = contradictions[:n]
        contradiction_pairs = contradiction_pairs[:n]
        severity_scores = (severity_scores[:n] if severity_scores else []) + [0.5] * max(0, n - len(severity_scores))

        if contradictions:
            print(f"\n  [!] {len(contradictions)} contradiction(s) detected:\n")
            for i, contradiction in enumerate(contradictions):
                pair = contradiction_pairs[i]
                severity = severity_scores[i]
                print(f"    Contradiction {i+1}: {contradiction}")
                print(f"      Between claims: {pair} | Severity: {severity}")
        else:
            print("\n  [OK] No internal contradictions detected.")

        # ════════════════════════════════════════
        # STEP 5: Final Verdict
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 5: Final Verdict — Hallucinated / Not Hallucinated")
        print("=" * 60)

        # Build per-finding verdict
        detailed_findings = []
        for i in range(len(findings)):
            cls = classifications[i]
            reason = baseline_reasoning[i]

            # Start with baseline confidence
            if cls.lower() == "typical":
                grounding_score = 1.0
            else:
                # Atypical: check if PubMed grounded it
                if pubmed_evidence.get(i):
                    grounding_score = 0.7  # atypical but has PubMed support
                else:
                    grounding_score = 0.3  # atypical and ungrounded

            # Check if this finding is involved in a contradiction
            is_contradicted = False
            contradiction_details = []
            for ci, pair in enumerate(contradiction_pairs):
                if i in pair:
                    is_contradicted = True
                    severity = severity_scores[ci]
                    grounding_score = max(0.0, grounding_score - severity)
                    contradiction_details.append(
                        contradictions[ci] if ci < len(contradictions) else "Contradiction detected"
                    )
            contradiction_detail = "; ".join(contradiction_details)

            is_hallucination = grounding_score < 0.5

            detailed_findings.append(
                {
                    "finding": findings[i],
                    "target": anatomical_targets[i],
                    "status": clinical_statuses[i],
                    "baseline_classification": cls,
                    "baseline_reasoning": reason,
                    "pubmed_grounded": bool(pubmed_evidence.get(i)),
                    "is_contradicted": is_contradicted,
                    "contradiction_detail": contradiction_detail,
                    "grounding_score": grounding_score,
                    "hallucination_risk": is_hallucination,
                }
            )

            # Update KG confidence
            finding_name = f"Claim_{i+1}"
            if finding_name in self.kg.entities:
                self.kg.entities[finding_name].confidence = grounding_score

            # Update edge confidence
            target_name = anatomical_targets[i].title()
            if target_name in self.kg.graph and finding_name in self.kg.graph[target_name]:
                self.kg.graph[target_name][finding_name]["confidence"] = grounding_score

        # Calculate overall verdict
        if detailed_findings:
            overall_confidence = sum(f["grounding_score"] for f in detailed_findings) / len(detailed_findings)
            hallucination_detected = any(f["hallucination_risk"] for f in detailed_findings)
        else:
            overall_confidence = 0.0
            hallucination_detected = False

        # Print per-finding verdict
        for i, item in enumerate(detailed_findings):
            verdict = "[FAIL] HALLUCINATED" if item["hallucination_risk"] else "[PASS] NOT HALLUCINATED"
            print(f"\n    Claim {i+1}: {verdict}  (Confidence: {item['grounding_score']:.2f})")
            print(f"      \"{item['finding']}\"")
            print(f"      Baseline: {item['baseline_classification']} | PubMed Grounded: {item['pubmed_grounded']} | Contradicted: {item['is_contradicted']}")
            if item["contradiction_detail"]:
                print(f"      Contradiction: {item['contradiction_detail']}")

        # Print overall verdict
        overall_verdict = "[FAIL] HALLUCINATION DETECTED" if hallucination_detected else "[PASS] REPORT IS CLINICALLY CONSISTENT"
        print(f"\n  {'-' * 50}")
        print(f"  OVERALL VERDICT: {overall_verdict}")
        print(f"  Average Confidence Score: {overall_confidence:.2f}")
        print(f"  Graph Stats: {len(self.kg.entities)} entities, {len(self.kg.relations)} relations")
        print(f"  {'-' * 50}")

        return {
            "report": report,
            "overall_confidence": overall_confidence,
            "hallucination_detected": hallucination_detected,
            "detailed_findings": detailed_findings,
            "contradictions": contradictions,
            "graph_stats": {
                "num_entities": len(self.kg.entities),
                "num_relations": len(self.kg.relations),
            },
        }



