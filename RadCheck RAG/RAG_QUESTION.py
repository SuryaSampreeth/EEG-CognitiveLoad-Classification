"""
AMG-RAG: Medical Question Answering System
6-Step Pipeline:
  1. Entity Extraction (Relevance Scores 1-10)
  2. KG Construction (Bidirectional Relationships)
  3. Multi-source Retrieval (PubMed / Wikipedia / VectorDB)
  4. Entity Summarization
  5. Chain-of-Thought Reasoning over Graph
  6. Final Answer + Confidence Score
"""

import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import networkx as nx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import requests
from xml.etree import ElementTree as ET
import wikipedia
from decouple import config

# Configuration
GOOGLE_API_KEY = config('GOOGLE_API_KEY')
PUBMED_API_KEY = config('pubmed_api', default=None)


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
        self.graph.add_node(entity.name, description=entity.description,
                           entity_type=entity.entity_type, confidence=entity.confidence,
                           sources=entity.sources)

    def add_relation(self, relation: MedicalRelation):
        self.relations.append(relation)
        self.graph.add_edge(relation.source, relation.target,
                           relation_type=relation.relation_type, confidence=relation.confidence,
                           evidence=relation.evidence, sources=relation.sources)

    def get_connected_nodes(self, node_name: str, confidence_threshold: float = 0.3):
        connected = []
        if node_name in self.graph:
            for neighbor in self.graph.neighbors(node_name):
                edge_data = self.graph[node_name][neighbor]
                if edge_data.get('confidence', 0) >= confidence_threshold:
                    connected.append({'node': neighbor, 'relation': edge_data.get('relation_type'),
                                      'confidence': edge_data.get('confidence'),
                                      'evidence': edge_data.get('evidence')})
        return connected

    def explore_path(self, start_node: str, max_depth: int = 2, confidence_threshold: float = 0.3):
        paths, visited = [], set()
        def dfs(node, path, acc_conf, depth):
            if depth > max_depth or node in visited: return
            visited.add(node)
            if path:
                paths.append({'path': path.copy(), 'confidence': acc_conf, 'final_node': node})
            for nd in self.get_connected_nodes(node, confidence_threshold):
                new_conf = acc_conf * nd['confidence']
                if new_conf >= confidence_threshold:
                    dfs(nd['node'], path + [(node, nd['node'], nd['relation'])], new_conf, depth + 1)
            visited.remove(node)
        dfs(start_node, [], 1.0, 0)
        return paths


# ──────────────────────────────────────────────
# PubMed Searcher
# ──────────────────────────────────────────────
class PubMedSearcher:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def search(self, query: str, max_results: int = 3) -> List[str]:
        time.sleep(0.5)
        params = {"db": "pubmed", "term": query, "retmode": "xml", "retmax": max_results}
        if self.api_key: params["api_key"] = self.api_key
        try:
            resp = requests.get(f"{self.base_url}/esearch.fcgi", params=params, timeout=30)
            if resp.status_code != 200 or not resp.text.strip().startswith("<"): return []
            pmids = [e.text for e in ET.fromstring(resp.text).findall(".//Id")]
            if not pmids: return []
            time.sleep(0.5)
            fp = {"db": "pubmed", "id": ",".join(pmids), "retmode": "text", "rettype": "abstract"}
            if self.api_key: fp["api_key"] = self.api_key
            resp = requests.get(f"{self.base_url}/efetch.fcgi", params=fp, timeout=30)
            if resp.status_code != 200: return []
            abstracts = []
            for article in resp.text.split("\n\n"):
                lines = [l for l in article.split("\n") if l.strip()
                         and not any(s in l.lower() for s in ["author","doi","pmid","copyright"])]
                if lines: abstracts.append(" ".join(lines))
            return abstracts
        except Exception as e:
            print(f"    PubMed error: {e}")
            return []


# ──────────────────────────────────────────────
# Main QA System
# ──────────────────────────────────────────────
class AMG_RAG_QASystem:
    """AMG-RAG system for Medical Question Answering"""

    def __init__(self, google_api_key: str = None):
        if not google_api_key:
            raise ValueError("Google API key is required.")
        self.llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0.0,
                                          google_api_key=google_api_key)
        self.kg = MedicalKnowledgeGraph()
        self.pubmed = PubMedSearcher(api_key=PUBMED_API_KEY)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = Chroma(collection_name="medical_qa", embedding_function=self.embeddings)
        self._setup_chains()

    def _setup_chains(self):
        # ── Step 1: Entity Extraction ──
        ep = StructuredOutputParser.from_response_schemas([
            ResponseSchema(name="entities", description="List of medical entities", type="array"),
            ResponseSchema(name="scores", description="Relevance scores (1-10) for each entity", type="array"),
            ResponseSchema(name="descriptions", description="Brief descriptions of each entity", type="array"),
        ])
        self.entity_extractor = PromptTemplate(
            template="""Extract all medical entities from this question and options with relevance scoring.
            Include diseases, drugs, symptoms, treatments, and medical concepts.
            Question: {question}
            Options: {options}
            Context: {context}
            For each entity provide: 1. Entity name 2. Relevance score (1-10) 3. Brief description
            Return in JSON format: {format_instructions}""",
            input_variables=["question", "options", "context"],
            partial_variables={"format_instructions": ep.get_format_instructions()}
        ) | self.llm | ep

        # ── Step 2: Relation Extraction ──
        rp = StructuredOutputParser.from_response_schemas([
            ResponseSchema(name="relationships",
                          description="List of relationship dicts with entityA, entityB, relationship_type, confidence_score, evidence",
                          type="array"),
        ])
        self.relation_extractor = PromptTemplate(
            template="""Analyze medical relationships between these entities.
            Entities and Descriptions: {entities_with_descriptions}
            Context: {context}
            Provide relationships in JSON: {{"relationships": [{{"entityA": "...", "entityB": "...", "relationship_type": "...", "confidence_score": 8, "evidence": "..."}}]}}
            Use types: treats, causes, symptom_of, risk_factor_for, contraindicated_with, differential_diagnosis, etc.
            Return ONLY the JSON: {format_instructions}""",
            input_variables=["entities_with_descriptions", "context"],
            partial_variables={"format_instructions": rp.get_format_instructions()}
        ) | self.llm | rp

        # ── Step 4: Entity Summarization ──
        sp = StructuredOutputParser.from_response_schemas([
            ResponseSchema(name="summaries", description="Concise summaries for each entity", type="array"),
            ResponseSchema(name="scores", description="Relevance scores (1-10) for each summary", type="array"),
        ])
        self.summary_chain = PromptTemplate(
            template="""Generate concise summaries for each medical entity based on the context.
            Entities: {entities}
            Context: {context}
            For each: 1. Concise summary (2-3 sentences) 2. Relevance score (1-10)
            Return in JSON format: {format_instructions}""",
            input_variables=["entities", "context"],
            partial_variables={"format_instructions": sp.get_format_instructions()}
        ) | self.llm | sp

        # ── Step 5: Chain-of-Thought Reasoning ──
        cp = StructuredOutputParser.from_response_schemas([
            ResponseSchema(name="reasoning", description="Step-by-step medical reasoning", type="string"),
        ])
        self.cot_chain = PromptTemplate(
            template="""Based on the medical knowledge graph and search results, provide step-by-step reasoning.
            Question: {question}
            Graph Knowledge: {graph_context}
            Search Results: {search_context}
            Provide detailed medical reasoning: {format_instructions}""",
            input_variables=["question", "graph_context", "search_context"],
            partial_variables={"format_instructions": cp.get_format_instructions()}
        ) | self.llm | cp

        # ── Step 6: Final Answer ──
        ap = StructuredOutputParser.from_response_schemas([
            ResponseSchema(name="answer", description="Final answer (A, B, C, D, or E)", type="string"),
            ResponseSchema(name="confidence", description="Confidence in the answer (0-1)", type="number"),
            ResponseSchema(name="explanation", description="Brief explanation", type="string"),
        ])
        self.answer_chain = PromptTemplate(
            template="""Based on the reasoning and evidence, select the best answer.
            Question: {question}
            Options: {options}
            Reasoning: {reasoning}
            Evidence: {evidence}
            Select the best answer (A, B, C, D, or E): {format_instructions}""",
            input_variables=["question", "options", "reasoning", "evidence"],
            partial_variables={"format_instructions": ap.get_format_instructions()}
        ) | self.llm | ap

    # ──────────────────────────────────────────
    # Main Pipeline
    # ──────────────────────────────────────────
    def answer_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        question = question_data["question"]
        options = question_data.get("options", {})
        options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])

        print(f"\n{'='*60}")
        print(f"  Question: {question[:120]}...")
        print(f"  Options: {options}")
        print(f"{'='*60}")

        # ════════════════════════════════════════
        # STEP 1: Entity Extraction (Relevance Scores 1-10)
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 1: Entity Extraction (Relevance Scores 1-10)")
        print("=" * 60)

        # Initial PubMed context for extraction
        init_query = question + " " + " ".join(list(options.values())[:3])
        init_results = self.pubmed.search(init_query, max_results=3)
        init_context = "\n".join(init_results) if init_results else ""

        try:
            ent_result = self.entity_extractor.invoke({
                "question": question, "options": options_text, "context": init_context
            })
            entities = ent_result.get("entities", [])
            scores = ent_result.get("scores", [])
            descriptions = ent_result.get("descriptions", [])
        except Exception as e:
            print(f"  Entity extraction error: {e}")
            entities = list(options.values())[:3]
            scores = [5] * len(entities)
            descriptions = [f"Medical concept: {e}" for e in entities]

        print(f"\n  Extracted {len(entities)} entities:\n")
        for i, ent in enumerate(entities):
            score = scores[i] if i < len(scores) else "N/A"
            desc = descriptions[i] if i < len(descriptions) else ""
            print(f"    {i+1}. {ent}  (Relevance: {score}/10)")
            print(f"       {desc[:100]}")

        # ════════════════════════════════════════
        # STEP 2: KG Construction (Bidirectional Relationships)
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 2: KG Construction (Bidirectional Relationships)")
        print("=" * 60)

        self.kg = MedicalKnowledgeGraph()

        # Add entities to KG (with basic info first, retrieval enriches later)
        for i, entity in enumerate(entities[:8]):
            desc = descriptions[i] if i < len(descriptions) else f"Medical entity: {entity}"
            rel_score = scores[i] if i < len(scores) else 5
            confidence = min(1.0, rel_score / 10.0 + 0.1)
            med_entity = MedicalEntity(name=entity, description=desc[:500],
                                       entity_type="medical_concept", confidence=confidence,
                                       sources=["LLM Extraction"])
            self.kg.add_entity(med_entity)

        # Extract relationships
        entity_list = list(self.kg.entities.keys())
        if len(entity_list) > 1:
            desc_block = "\n".join([f"- {e}: {self.kg.entities[e].description[:200]}" for e in entity_list])
            rel_context = f"Question: {question}\n\nOptions: {options_text}\n\nSearch Results: {init_context}"
            try:
                print("\n  Extracting bidirectional relationships...")
                rel_result = self.relation_extractor.invoke({
                    "entities_with_descriptions": desc_block, "context": rel_context
                })
                relationships = rel_result.get("relationships", [])
                for rel in relationships:
                    if isinstance(rel, dict):
                        ea, eb = rel.get("entityA", ""), rel.get("entityB", "")
                        if ea in self.kg.entities and eb in self.kg.entities:
                            self.kg.add_relation(MedicalRelation(
                                source=ea, target=eb,
                                relation_type=rel.get("relationship_type", "related_to"),
                                confidence=rel.get("confidence_score", 5) / 10.0,
                                evidence=rel.get("evidence", ""), sources=["LLM Analysis"]
                            ))
            except Exception as e:
                print(f"  Relation extraction error: {e}")

        print(f"\n  Graph built: {len(self.kg.entities)} entities, {len(self.kg.relations)} relations\n")
        for rel in self.kg.relations:
            print(f"    {rel.source} --[{rel.relation_type}]--> {rel.target}  (conf: {rel.confidence:.2f})")

        # ════════════════════════════════════════
        # STEP 3: Multi-source Retrieval (PubMed / Wikipedia / VectorDB)
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 3: Multi-source Retrieval (PubMed / Wikipedia / VectorDB)")
        print("=" * 60)

        for i, entity in enumerate(entities[:8]):
            print(f"\n    Retrieving for: \"{entity}\"")

            # PubMed
            abstracts = self.pubmed.search(entity, max_results=2)
            if abstracts:
                print(f"      [PubMed] {len(abstracts)} article(s) found")
            else:
                print(f"      [PubMed] No results")

            # Wikipedia
            wiki_content = ""
            try:
                wiki_results = wikipedia.search(entity, results=1)
                if wiki_results:
                    wiki_content = wikipedia.summary(wiki_results[0], sentences=3)
                    print(f"      [Wikipedia] {wiki_content[:100]}...")
                else:
                    print(f"      [Wikipedia] No results")
            except:
                print(f"      [Wikipedia] No results")

            # Enrich entity description with retrieved info
            ext_desc = " ".join(abstracts) if abstracts else wiki_content
            if ext_desc:
                orig = self.kg.entities[entity].description
                self.kg.entities[entity].description = f"{orig}. {ext_desc}"[:500]
                self.kg.entities[entity].sources = ["PubMed", "Wikipedia"] if abstracts else ["Wikipedia"]
                rel_score = scores[i] if i < len(scores) else 5
                self.kg.entities[entity].confidence = min(1.0, rel_score / 10.0 + (0.2 if abstracts else 0.1))

        # ════════════════════════════════════════
        # STEP 4: Entity Summarization
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 4: Entity Summarization")
        print("=" * 60)

        try:
            ent_list = list(self.kg.entities.keys())
            sum_result = self.summary_chain.invoke({
                "entities": ent_list, "context": f"Question: {question}\n\nContext: {init_context}"
            })
            summaries = sum_result.get("summaries", [])
            sum_scores = sum_result.get("scores", [])

            print()
            for i, ename in enumerate(ent_list):
                if i < len(summaries) and i < len(sum_scores):
                    enhanced = summaries[i]
                    rel_s = sum_scores[i]
                    orig = self.kg.entities[ename].description
                    self.kg.entities[ename].description = f"{orig}\n\nEnhanced Summary: {enhanced}"[:500]
                    cur_conf = self.kg.entities[ename].confidence
                    self.kg.entities[ename].confidence = min(1.0, (cur_conf + min(1.0, rel_s / 10.0)) / 2)
                    print(f"    {ename} (updated confidence: {self.kg.entities[ename].confidence:.2f})")
                    print(f"      Summary: {enhanced[:120]}...")
        except Exception as e:
            print(f"  Entity summarization error: {e}")

        # ════════════════════════════════════════
        # STEP 5: Chain-of-Thought Reasoning over Graph
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 5: Chain-of-Thought Reasoning over Graph")
        print("=" * 60)

        graph_context = []
        for entity in list(self.kg.entities.keys())[:3]:
            connections = self.kg.get_connected_nodes(entity)
            paths = self.kg.explore_path(entity)
            ctx = f"Entity: {entity}\nDescription: {self.kg.entities[entity].description[:200]}\n"
            if connections:
                ctx += "Direct connections:\n"
                for c in connections[:3]:
                    ctx += f"  - {c['relation']} -> {c['node']} (conf: {c['confidence']:.2f})\n"
            if paths:
                ctx += "Reasoning paths:\n"
                for pd in paths[:2]:
                    ps = " -> ".join([f"{p[0]} [{p[2]}]" for p in pd['path']])
                    if ps: ctx += f"  - {ps} -> {pd['final_node']} (conf: {pd['confidence']:.2f})\n"
            graph_context.append(ctx)

        print("\n  Graph paths explored for top entities:")
        for gc in graph_context:
            for line in gc.strip().split("\n"):
                print(f"    {line}")
            print()

        # Additional PubMed evidence for reasoning
        search_query = question + " " + " ".join(list(self.kg.entities.keys())[:3])
        search_results = self.pubmed.search(search_query, max_results=2)
        search_context = "\n".join(search_results) if search_results else "No additional search results."

        try:
            cot_result = self.cot_chain.invoke({
                "question": question, "graph_context": "\n\n".join(graph_context),
                "search_context": search_context
            })
            reasoning = cot_result.get("reasoning", "Unable to generate reasoning")
        except Exception as e:
            print(f"  CoT error: {e}")
            reasoning = "Error in reasoning generation"

        print(f"  Reasoning:\n    {reasoning[:400]}{'...' if len(reasoning) > 400 else ''}")

        # ════════════════════════════════════════
        # STEP 6: Final Answer + Confidence Score
        # ════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Step 6: Final Answer + Confidence Score")
        print("=" * 60)

        evidence = "\n".join(graph_context[:2])
        try:
            ans_result = self.answer_chain.invoke({
                "question": question, "options": options_text,
                "reasoning": reasoning, "evidence": evidence
            })
            answer = ans_result.get("answer", "N/A")
            confidence = ans_result.get("confidence", 0.0)
            explanation = ans_result.get("explanation", "")
        except Exception as e:
            print(f"  Answer generation error: {e}")
            answer, confidence, explanation = "Error", 0.0, str(e)

        print(f"\n    Answer: {answer}")
        print(f"    Confidence: {confidence:.2f}")
        print(f"    Explanation: {explanation}")

        return {
            "answer": answer, "confidence": confidence, "explanation": explanation,
            "reasoning": reasoning, "graph_context": graph_context,
            "search_context": search_context, "question": question, "options": options,
            "expected_answer": question_data.get("answer", "Unknown"),
            "graph_stats": {"num_entities": len(self.kg.entities), "num_relations": len(self.kg.relations)},
        }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AMG-RAG: Medical Question Answering System")
    print("=" * 60)

    print("\nInitializing system...")
    system = AMG_RAG_QASystem(google_api_key=GOOGLE_API_KEY)

    question_data = {
        "question": "A 45-year-old man presents to the emergency department with severe chest pain that started 2 hours ago. The pain is substernal, crushing in nature, and radiates to his left arm. He has a history of hypertension and diabetes mellitus. His father died of a myocardial infarction at age 50. On examination, he is diaphoretic and in distress. His blood pressure is 150/90 mmHg, pulse is 110/min, and respirations are 22/min. An ECG shows ST-segment elevation in leads II, III, and aVF. Which of the following is the most likely diagnosis?",
        "options": {
            "A": "Unstable angina",
            "B": "Acute inferior wall myocardial infarction",
            "C": "Acute anterior wall myocardial infarction",
            "D": "Aortic dissection",
            "E": "Pulmonary embolism",
        },
        "answer": "B",
    }

    result = system.answer_question(question_data)

    # ── Final Summary ──
    print("\n\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n  Expected: {result['expected_answer']}")
    print(f"  Predicted: {result['answer']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Correct: {'[PASS] YES' if result['answer'] == result['expected_answer'] else '[FAIL] NO'}")
    print(f"\n  Graph: {result['graph_stats']['num_entities']} entities, {result['graph_stats']['num_relations']} relations")
    print(f"\n  Explanation: {result['explanation']}")

    # Print KG structure
    print("\n" + "=" * 60)
    print("  KNOWLEDGE GRAPH STRUCTURE")
    print("=" * 60)
    for ename, entity in list(system.kg.entities.items())[:5]:
        print(f"\n  [{ename}] (conf: {entity.confidence:.2f})")
        conns = system.kg.get_connected_nodes(ename)
        for c in conns[:3]:
            print(f"    -> {c['relation']} -> {c['node']} (conf: {c['confidence']:.2f})")


if __name__ == "__main__":
    main()