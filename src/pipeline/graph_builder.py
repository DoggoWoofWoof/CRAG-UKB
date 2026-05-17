"""
Graph Builder — Entity-based cross-linking for the "One Source, Two Views" pipeline.
====================================================================================
Takes StandardNode list from loaders and enriches it with cross-document edges
derived from shared article title mentions (for HotPotQA) or optional NER entities.

Run:
    python -m src.pipeline.graph_builder
"""

import logging
from typing import List, Dict, Set
from collections import defaultdict

from .standardizer import StandardNode

log = logging.getLogger(__name__)


def enrich_with_title_links(nodes: List[StandardNode]) -> List[StandardNode]:
    """
    Cross-link document nodes based on shared title mentions.
    
    If document A's content mentions the title of document B,
    create an edge A → B (entity co-occurrence).
    
    This is the primary graph construction method for HotPotQA
    where article titles serve as natural entity identifiers.
    """
    # Build title → node_id mapping (document nodes only)
    title_to_nid: Dict[str, str] = {}
    for node in nodes:
        if node.metadata.get("type") == "document":
            title = node.metadata.get("title", "")
            if title:
                title_to_nid[title] = node.node_id

    # Cross-link: if document A mentions title of document B
    links_added = 0
    for node in nodes:
        if node.metadata.get("type") != "document":
            continue
        content_lower = node.content.lower()
        for title, target_nid in title_to_nid.items():
            if target_nid == node.node_id:
                continue  # skip self
            if title.lower() in content_lower:
                if target_nid not in node.neighbors:
                    node.neighbors.append(target_nid)
                    links_added += 1

    log.info(f"Title-based cross-linking: added {links_added} edges")
    return nodes


def enrich_with_spacy_entities(nodes: List[StandardNode],
                                model_name: str = "en_core_web_sm") -> List[StandardNode]:
    """
    Optional: Run spaCy NER on document nodes to extract named entities,
    then cross-link documents that share the same entity.

    Creates entity StandardNodes (type='entity') and MENTIONS edges.
    """
    try:
        import spacy
        nlp = spacy.load(model_name)
    except (ImportError, OSError) as e:
        log.warning(f"spaCy not available ({e}), skipping entity enrichment.")
        return nodes

    # Extract entities per document node
    doc_entities: Dict[str, Set[str]] = {}  # node_id → set of entity strings
    entity_id_counter = 0
    entity_cache: Dict[str, StandardNode] = {}  # entity_text → StandardNode

    document_nodes = [n for n in nodes if n.metadata.get("type") == "document"]
    log.info(f"Running NER on {len(document_nodes)} document nodes...")

    for node in document_nodes:
        doc = nlp(node.content[:10000])  # truncate for performance
        entities = set()
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART"):
                ent_text = ent.text.strip()
                if len(ent_text) > 2:
                    entities.add(ent_text)

                    # Create entity node if new
                    if ent_text not in entity_cache:
                        ent_node_id = f"entity_{entity_id_counter}"
                        entity_id_counter += 1
                        ent_node = StandardNode(
                            node_id=ent_node_id,
                            content=ent_text,
                            metadata={"source": "ner", "type": "entity",
                                      "entity_label": ent.label_}
                        )
                        entity_cache[ent_text] = ent_node
                        nodes.append(ent_node)

                    # Link document ↔ entity
                    ent_nid = entity_cache[ent_text].node_id
                    if ent_nid not in node.neighbors:
                        node.neighbors.append(ent_nid)
                    if node.node_id not in entity_cache[ent_text].neighbors:
                        entity_cache[ent_text].neighbors.append(node.node_id)

        doc_entities[node.node_id] = entities

    # Cross-link documents with shared entities
    entity_to_docs: Dict[str, List[str]] = defaultdict(list)
    for nid, ents in doc_entities.items():
        for e in ents:
            entity_to_docs[e].append(nid)

    cross_links = 0
    for ent, doc_ids in entity_to_docs.items():
        if len(doc_ids) < 2:
            continue
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                nid1, nid2 = doc_ids[i], doc_ids[j]
                # Find nodes and add edges
                for node in nodes:
                    if node.node_id == nid1 and nid2 not in node.neighbors:
                        node.neighbors.append(nid2)
                        cross_links += 1
                    elif node.node_id == nid2 and nid1 not in node.neighbors:
                        node.neighbors.append(nid1)
                        cross_links += 1

    log.info(f"Entity NER enrichment: {len(entity_cache)} entities, {cross_links} cross-links")
    return nodes


def build_graph_view(nodes: List[StandardNode], use_spacy: bool = False) -> List[StandardNode]:
    """
    Full graph construction pipeline:
    1. Title-based cross-linking (always, no dependencies)
    2. Optional spaCy NER enrichment

    Returns the enriched node list (same list, mutated in-place).
    """
    log.info("Building graph view...")
    nodes = enrich_with_title_links(nodes)

    if use_spacy:
        nodes = enrich_with_spacy_entities(nodes)

    # Summary stats
    doc_nodes = sum(1 for n in nodes if n.metadata.get("type") == "document")
    q_nodes = sum(1 for n in nodes if n.metadata.get("type") == "question")
    ent_nodes = sum(1 for n in nodes if n.metadata.get("type") == "entity")
    total_edges = sum(len(n.neighbors) for n in nodes) // 2
    log.info(f"Graph view: {doc_nodes} docs, {q_nodes} questions, "
             f"{ent_nodes} entities, ~{total_edges} edges")
    return nodes
