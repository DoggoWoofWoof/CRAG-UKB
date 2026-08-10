import ast
import json
import os
import logging
from typing import List, Dict, Any
from collections import defaultdict

from .standardizer import StandardNode, save_nodes

log = logging.getLogger(__name__)


def load_squad(file_path: str) -> List[StandardNode]:
    """Parse SQuAD v2 into document chunk nodes and linked question nodes."""
    nodes = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunk_id: int = 0
    q_id: int = 0
    for article in data['data']:
        title = article.get('title', 'Unknown')
        prev_doc_node = None                      # previous DOCUMENT node in this article (not nodes[-1], which is a q node)
        for p_idx, para in enumerate(article['paragraphs']):
            context = para['context']
            node_id = f"squad_{chunk_id}"
            chunk_id = chunk_id + 1

            node = StandardNode(
                node_id=node_id,
                content=context,
                metadata={"source": "squad", "type": "document", "title": title}
            )
            if prev_doc_node is not None:          # bidirectional doc<->doc chain within the article
                node.neighbors.append(prev_doc_node.node_id)
                prev_doc_node.neighbors.append(node_id)

            nodes.append(node)
            prev_doc_node = node

            for qa in para.get('qas', []):
                question_text = qa['question']
                q_node_id = f"squad_q_{q_id}"
                q_id = q_id + 1
                answers = [
                    str(answer.get("text", ""))
                    for answer in qa.get("answers", [])
                    if answer.get("text")
                ]

                q_node = StandardNode(
                    node_id=q_node_id,
                    content=question_text,
                    metadata={"source": "squad", "type": "question",
                              "is_impossible": qa.get('is_impossible', False),
                              "answer": answers[0] if answers else "",
                              "answers": answers}
                )
                q_node.neighbors.append(node_id)
                node.neighbors.append(q_node_id)
                nodes.append(q_node)

    log.info(f"  SQuAD: {chunk_id} document nodes, {q_id} question nodes, {len(nodes)} total")
    return nodes


def load_hotpotqa(file_path: str) -> List[StandardNode]:
    """
    Parse HotPotQA JSON/JSONL into document and question nodes with multi-hop edges.
    Supports line-by-line JSONL (preferred by mirrors) or full-block JSON.
    """
    nodes: List[StandardNode] = []
    items = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try sensing if it's JSONL or JSON
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                items = json.load(f)
            else:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
    except Exception as e:
        log.error(f"Failed to parse HotPotQA from {file_path}: {e}")
        return []

        # article_title → StandardNode (deduplicated across questions)
    article_cache: Dict[str, StandardNode] = {}
    doc_id_counter: int = 0

    for item in items:
        qid = item.get('_id', item.get('id', 'unknown'))
        question = item['question']
        answers = item.get("golden_answers") or []
        if isinstance(answers, str):
            answers = [answers]
        answer = item.get("answer") or (answers[0] if answers else "")
        answers = [answer, *answers] if answer else list(answers)
        answers = list(dict.fromkeys(str(value) for value in answers if value))
        
        # ── Build context article nodes ─────────────────────────────
        context_map: Dict[str, str] = {}  # title → node_id
        
        meta = item.get('metadata', item)
        context = meta.get('context', {})
        
        if isinstance(context, dict):
            # FlashRAG uses "sentences"; older HotpotQA dumps use "content"
            ctx_iterator = zip(context.get("title", []),
                               context.get("content", context.get("sentences", [])))
        else:
            ctx_iterator = context
            
        for title, sentences in ctx_iterator:
            if title in article_cache:
                context_map[str(title)] = article_cache[title].node_id
                continue
            
            # sentences can be a list of strings or a single string
            content = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            node_id = f"hotpot_doc_{doc_id_counter}"
            doc_id_counter = int(doc_id_counter) + 1

            doc_node = StandardNode(
                node_id=node_id,
                content=content,
                metadata={"source": "hotpotqa", "type": "document", "title": title}
            )
            article_cache[str(title)] = doc_node
            context_map[str(title)] = node_id
            nodes.append(doc_node)

        # ── Build question node ─────────────────────────────────────
        q_node_id = f"hotpot_q_{qid}"
        q_node = StandardNode(
            node_id=q_node_id,
            content=question,
            metadata={
                "source": "hotpotqa",
                "type": "question",
                "answer": answer,
                "answers": answers,
            }
        )

        # ── Link question ↔ supporting fact articles ────────────────
        supporting_titles = set()
        
        # FlashRAG schema: supporting_facts might be {'title': [...], 'sent_id': [...]}
        sf_field = item.get('supporting_facts', meta.get('supporting_facts', []))
        if isinstance(sf_field, dict):
            supporting_titles = set(sf_field.get('title', []))
        else:
            for sf in sf_field:
                # sf format: [title, sent_idx]
                if isinstance(sf, (list, tuple)) and len(sf) > 0:
                    supporting_titles.add(sf[0])

        for sf_title in supporting_titles:
            if sf_title in context_map:
                doc_nid = context_map[sf_title]
                q_node.neighbors.append(doc_nid)
                article_cache[sf_title].neighbors.append(q_node_id)

        # BRIDGE EDGES REMOVED (label leak) — see load_2wiki note + build_clean.py.
        # HotpotQA dedups by title (article_cache), so its label-free structure is
        # title-mention content edges + kNN, built downstream.
        nodes.append(q_node)

    return nodes


def load_musique(file_path: str) -> List[StandardNode]:
    """Parse MuSiQue JSONL into document and question nodes."""
    nodes = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            qid = item.get('id', 'unknown')
            question = item.get('question', '')
            
            doc_nodes = []
            
            # FlashRAG schema might wrap these inside metadata
            meta = item.get('metadata', {})
            paragraphs = item.get('paragraphs', meta.get('paragraphs', []))
            
            # If still not found, check for a generic context or within decomposition
            if not paragraphs:
                if 'context' in meta:
                    p_dict = meta['context']
                    if isinstance(p_dict, dict):
                        paragraphs = [{"paragraph_text": " ".join(s) if isinstance(s, list) else s}
                                     for s in p_dict.get('content', [])]
                elif 'question_decomposition' in meta:
                    # FlashRAG stores each support passage as a stringified Python dict
                    # {'idx','title','paragraph_text'} — parse it so we keep the TITLE
                    # (dropping it left musique with 0 relational edges; L3 was inert).
                    for step in meta['question_decomposition']:
                        sp = step.get('support_paragraph')
                        if not sp:
                            continue
                        try:
                            pd = ast.literal_eval(sp) if isinstance(sp, str) else sp
                        except (ValueError, SyntaxError):
                            continue
                        if isinstance(pd, dict):
                            pd.setdefault('is_supporting', True)   # decomposition paras are all gold
                            paragraphs.append(pd)

            # Context paragraphs
            for i, p in enumerate(paragraphs):
                if not isinstance(p, dict):
                    continue
                p_text = p.get('paragraph_text', p.get('content', ''))
                if not p_text: continue

                node_id = f"musique_doc_{qid}_{i}"
                doc_nodes.append(StandardNode(
                    node_id=node_id,
                    content=p_text,
                    metadata={"source": "musique", "type": "document", "title": p.get('title', ''),
                              "is_supporting": p.get('is_supporting', False)}
                ))
            
            # Question
            answer = item.get('answer', item.get('golden_answers', [""])[0] if item.get('golden_answers') else "")
            q_node = StandardNode(
                node_id=f"musique_q_{qid}",
                content=question,
                metadata={"source": "musique", "type": "question", "answer": answer}
            )
            
            # Link support docs to question
            supporting_doc_ids = []
            for d in doc_nodes:
                if d.metadata.get("is_supporting"):
                    q_node.neighbors.append(d.node_id)
                    d.neighbors.append(q_node.node_id)
                    supporting_doc_ids.append(d.node_id)
            
            # BRIDGE EDGES REMOVED (label leak) — see load_2wiki note + build_clean.py.
            # musique passages have no titles, so its label-free structure is kNN-only.
            nodes.extend(doc_nodes)
            nodes.append(q_node)

    log.info(f"  MuSiQue: {len(nodes)} total nodes parsed (label-free: no bridge edges)")
    return nodes


def load_musique_ans(file_path: str) -> List[StandardNode]:
    """Parse the ORIGINAL MuSiQue (``musique_ans``) JSONL into document + question nodes.

    Standard DEV *distractor* setting. Each record carries its own 20 candidate
    ``paragraphs`` ({idx,title,paragraph_text,is_supporting}); the corpus is the
    UNION of every question's paragraphs, deduplicated by (title, paragraph_text)
    so a passage shared across questions maps to ONE doc node (a passage that is a
    gold for one question and a distractor for another therefore stays a single
    node -> real distractors, not an all-gold pool).

    Modeled on ``load_hotpotqa``: dedup docs across questions, one question node
    per record, GOLD edges (q<->doc) added ONLY for ``is_supporting == True``
    paragraphs, and NO doc-doc bridge edges (label leak — see load_2wiki note and
    build_clean.py). Question answers = ``answer`` + ``answer_aliases``.
    """
    nodes: List[StandardNode] = []
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    # (title, paragraph_text) → StandardNode, deduplicated across all questions.
    para_cache: Dict[tuple, StandardNode] = {}
    doc_id_counter: int = 0

    for item in items:
        qid = item.get('id', 'unknown')
        question = item.get('question', '')

        # Answers: primary answer first, then aliases (deduped, order-preserving).
        answer = str(item.get('answer', '') or '')
        aliases = item.get('answer_aliases', []) or []
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [str(a) for a in aliases if a]
        answers = list(dict.fromkeys([answer, *aliases] if answer else list(aliases)))

        # ── Build / reuse paragraph doc nodes ───────────────────────
        supporting_doc_nodes: List[StandardNode] = []
        for p in item.get('paragraphs', []):
            if not isinstance(p, dict):
                continue
            title = p.get('title', '')
            p_text = p.get('paragraph_text', '')
            if not p_text:
                continue
            key = (title, p_text)
            doc_node = para_cache.get(key)
            if doc_node is None:
                node_id = f"musique_doc_{doc_id_counter}"
                doc_id_counter = doc_id_counter + 1
                doc_node = StandardNode(
                    node_id=node_id,
                    content=p_text,
                    metadata={"source": "musique", "type": "document", "title": title}
                )
                para_cache[key] = doc_node
                nodes.append(doc_node)
            if p.get('is_supporting'):
                supporting_doc_nodes.append(doc_node)

        # ── Build question node ─────────────────────────────────────
        q_node_id = f"musique_q_{qid}"
        q_node = StandardNode(
            node_id=q_node_id,
            content=question,
            metadata={
                "source": "musique",
                "type": "question",
                "answer": answer,
                "answers": answers,
                "answer_aliases": aliases,
            }
        )

        # ── Link question ↔ supporting (gold) paragraphs only ───────
        for doc_node in supporting_doc_nodes:
            if doc_node.node_id not in q_node.neighbors:
                q_node.neighbors.append(doc_node.node_id)
            if q_node_id not in doc_node.neighbors:
                doc_node.neighbors.append(q_node_id)

        # BRIDGE EDGES REMOVED (label leak) — same policy as load_hotpotqa/load_2wiki:
        # doc-doc structure comes from label-free title-mention links + kNN built
        # downstream in build_clean.py; q->gold edges above are eval labels only.
        nodes.append(q_node)

    log.info(f"  MuSiQue-ans: {doc_id_counter} unique doc nodes, "
             f"{len(items)} questions, {len(nodes)} total nodes")
    return nodes


def load_2wiki(file_path: str) -> List[StandardNode]:
    """Parse 2WikiMultiHopQA JSONL into document and question nodes."""
    nodes = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            qid = item.get('_id', item.get('id', 'unknown'))
            question = item['question']
            
            meta = item.get('metadata', item)
            supporting_facts = meta.get('supporting_facts', [])
            context = meta.get('context', [])
            
            supporting_titles: set[str] = set()
            if isinstance(supporting_facts, dict):
                supporting_titles = set(supporting_facts.get('title', []))
            else:
                supporting_titles = {str(fact[0]) for fact in supporting_facts if isinstance(fact, (list, tuple)) and len(fact) > 0}
            
            # Question
            answer = item.get('answer', item.get('golden_answers', [""])[0] if item.get('golden_answers') else "")
            q_node = StandardNode(
                node_id=f"2wiki_q_{qid}",
                content=question,
                metadata={"source": "2wiki", "type": "question", "answer": answer}
            )

            # Context
            doc_nodes = []
            supporting_doc_ids = []
            
            if isinstance(context, dict):
                ctx_iterator = zip(context.get("title", []), context.get("content", []))
            else:
                ctx_iterator = context
                
            for i, item_ctx in enumerate(ctx_iterator):
                title = item_ctx[0]
                sentences = item_ctx[1]
                node_id = f"2wiki_doc_{qid}_{i}"
                content = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
                d_node = StandardNode(
                    node_id=node_id,
                    content=content,
                    metadata={"source": "2wiki", "type": "document", "title": title}
                )
                if str(title) in supporting_titles:
                    q_node.neighbors.append(d_node.node_id)
                    d_node.neighbors.append(q_node.node_id)
                    supporting_doc_ids.append(d_node.node_id)
                doc_nodes.append(d_node)
                
            # BRIDGE EDGES REMOVED (label leak): linking co-supporting gold docs bakes
            # test-question annotations into the doc graph. Doc-doc structure now comes
            # ONLY from label-free content edges (title mentions) + kNN, built downstream
            # (src/pipeline/build_clean.py). Question->gold edges above are eval labels only.
            nodes.extend(doc_nodes)
            nodes.append(q_node)

    log.info(f"  2Wiki: {len(nodes)} total nodes parsed (label-free: no bridge edges)")
    return nodes

def _download_if_missing(url: str, dest_path: str):
    import urllib.request
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        log.info(f"Downloading {url} to {dest_path}...")
        try:
            # Add user-agent to avoid 403 from some mirrors
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, dest_path)
            log.info("Download complete.")
        except Exception as e:
            log.error(f"Failed to download {url}: {e}")


def _download_metaqa(dest_dir: str):
    """Download and extract MetaQA dataset from Google Drive if not already present."""
    kb_path = os.path.join(dest_dir, "kb.txt")
    # Also check nested extraction structure
    if os.path.exists(kb_path):
        log.info("MetaQA already downloaded.")
        return
    # Check if kb.txt exists in a subdirectory (nested zip extraction)
    for root, dirs, files in os.walk(dest_dir):
        if "kb.txt" in files:
            log.info(f"MetaQA already downloaded (found at {root}).")
            return

    os.makedirs(dest_dir, exist_ok=True)
    
    # MetaQA is hosted on Google Drive by the original authors (AAAI 2018)
    # Folder: https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE
    gdrive_folder_id = "0B-36Uca2AvwhTWVFSUZqRXVtbUE"
    
    try:
        import gdown
    except ImportError:
        log.error(
            "MetaQA requires 'gdown' to download from Google Drive. "
            "Install via: pip install gdown\n"
            "Or manually download from: https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE "
            f"and extract into {dest_dir}/"
        )
        return
    
    log.info(f"Downloading MetaQA from Google Drive (folder: {gdrive_folder_id})...")
    try:
        gdown.download_folder(
            id=gdrive_folder_id,
            output=dest_dir,
            quiet=False
        )
        # Verify download
        found = False
        for root, dirs, files in os.walk(dest_dir):
            if "kb.txt" in files:
                found = True
                break
        if found:
            log.info(f"MetaQA downloaded successfully to {dest_dir}")
        else:
            log.error("MetaQA download completed but kb.txt not found. Check folder contents.")
    except Exception as e:
        log.error(f"Failed to download MetaQA: {e}")
        log.error(
            "Please manually download from: "
            "https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE "
            f"and extract into {dest_dir}/"
        )


def load_metaqa(data_dir: str) -> List[StandardNode]:
    """
    Parse MetaQA knowledge base and QA pairs into StandardNodes.

    Entity nodes are created from kb.txt triples (subject|relation|object).
    Each entity's content is built from all triples it participates in,
    providing rich textual context for embedding.

    Question nodes are created from qa_train.txt across all 3 hops,
    linked to their answer entity nodes via neighbors.
    """
    nodes: List[StandardNode] = []
    entity_nodes: Dict[str, StandardNode] = {}  # entity_name → node
    # Track subject-position vs object-position triples separately
    # Subject triples define what the entity IS; object triples are weaker signals
    entity_subj_triples: Dict[str, List[str]] = defaultdict(list)
    entity_obj_triples: Dict[str, List[str]] = defaultdict(list)
    canonical_names: Dict[str, str] = {} # Store prettiest casing found for each lowercase entity
    unmatched_count: int = 0
    q_id_counter: int = 0

    # ── 1. Parse kb.txt → entity nodes with relational edges ────────
    kb_path = os.path.join(data_dir, "kb.txt")
    if not os.path.exists(kb_path):
        # Try nested extraction structure (data/MetaQA/kb.txt)
        for root, dirs, files in os.walk(data_dir):
            if "kb.txt" in files:
                kb_path = os.path.join(root, "kb.txt")
                data_dir = root  # Update base for QA file lookups
                break

    if not os.path.exists(kb_path):
        log.error(f"MetaQA kb.txt not found in {data_dir}")
        return []

    log.info(f"Parsing MetaQA knowledge base from {kb_path}...")
    edges_seen = set()  # (subj, obj) pairs already processed — prevents duplicate edges
    edges: List[tuple] = []

    with open(kb_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) != 3:
                continue
            # Normalize to lowercase and remove spaces/underscores for consistent matching
            subj_raw, rel, obj_raw = parts[0].strip(), parts[1].strip(), parts[2].strip()
            subj = subj_raw.lower().replace('_', ' ')
            obj = obj_raw.lower().replace('_', ' ')

            # Track canonical (prettiest) name: favor strings with any capital letters
            # Also replace underscores so the display name looks natural
            for raw, low in [(subj_raw.replace('_', ' '), subj), (obj_raw.replace('_', ' '), obj)]:
                existing = canonical_names.get(low)
                if existing is None or (any(c.isupper() for c in raw) and not any(c.isupper() for c in existing)):
                    canonical_names[low] = raw

            # Build natural-language triple sentence for embedding content
            triple_text = f"{subj_raw} {rel.replace('_', ' ')} {obj_raw}"
            entity_subj_triples[subj].append(triple_text)
            entity_obj_triples[obj].append(triple_text)

            # Deduplicate bidirectional edges (KB may contain A|r|B and B|r|A)
            edge_key = tuple(sorted([subj, obj]))
            if edge_key not in edges_seen:
                edges_seen.add(edge_key)
                edges.append((subj, obj))

    # Create entity nodes with strictly unique IDs
    all_entities = sorted(list(set(entity_subj_triples.keys()) | set(entity_obj_triples.keys())))
    # ID is now based on lowercase name — collision impossible because keys are already lowered
    entity_name_to_id = {ent: f"metaqa_ent_{ent}" for ent in all_entities}
    
    # Pre-calculate entity nodes to use during question linking
    for ent_low in all_entities:
        display_name = canonical_names.get(ent_low, ent_low)
        node_id = entity_name_to_id[ent_low]

        # Prioritize subject-position triples (define what entity IS)
        subj_triples = list(dict.fromkeys(entity_subj_triples.get(ent_low, [])))
        obj_triples = list(dict.fromkeys(entity_obj_triples.get(ent_low, [])))
        
        # Take up to 10 subject triples, fill remaining slots with object triples
        selected = subj_triples[:10]
        remaining = 10 - len(selected)
        if remaining > 0:
            selected.extend(obj_triples[:remaining])
            
        content = f"{display_name}. " + " | ".join(selected)

        entity_nodes[ent_low] = StandardNode(
            node_id=node_id,
            content=content,
            metadata={"source": "metaqa", "type": "document", "title": display_name}
        )

    # Wire neighbor edges from KB relations (already deduplicated above)
    for subj, obj in edges:
        if subj in entity_nodes and obj in entity_nodes:
            s_node = entity_nodes[subj]
            o_node = entity_nodes[obj]
            if o_node.node_id not in s_node.neighbors:
                s_node.neighbors.append(o_node.node_id)
            if s_node.node_id not in o_node.neighbors:
                o_node.neighbors.append(s_node.node_id)

    nodes.extend(entity_nodes.values())
    log.info(f"  MetaQA KB: {len(entity_nodes)} entity nodes, {len(edges)} unique relation edges")


    # ── 2. Parse QA files from all hops and splits ──────────────────
    for hop in [1, 2, 3]:
        for split in ["train", "dev", "test"]:
            qa_path = os.path.join(data_dir, f"{hop}-hop", "vanilla", f"qa_{split}.txt")
            if not os.path.exists(qa_path):
                continue

            log.info(f"  Parsing MetaQA {hop}-hop {split} questions...")
            with open(qa_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or '\t' not in line:
                        continue
                    question_text, answers_text = line.split('\t', 1)
                    answers = [a.strip() for a in answers_text.split('|') if a.strip()]

                    q_node_id = f"metaqa_q_{hop}hop_{split}_{q_id_counter}"
                    q_id_counter = q_id_counter + 1

                    q_node = StandardNode(
                        node_id=q_node_id,
                        content=question_text,
                        metadata={
                            "source": "metaqa",
                            "type": "question",
                            "hop": hop,
                            "split": split,
                            "answer": answers[0] if answers else ""
                        }
                    )

                    # Link question → answer entity nodes (one-way only)
                    for ans in answers:
                        ans_low = ans.lower()
                        if ans_low in entity_name_to_id:
                            q_node.neighbors.append(entity_name_to_id[ans_low])
                        else:
                            # Skip unmatched answers to avoid 'ghost' neighbors that break benchmarks
                            unmatched_count = unmatched_count + 1

                    nodes.append(q_node)

    total_q: int = q_id_counter
    log.info(f"  MetaQA QA: {total_q} question nodes across 3 hops")
    if unmatched_count > 0:
        log.warning(f"  MetaQA: {unmatched_count} answer strings were not found in KB (skipped)")
    log.info(f"  MetaQA Total: {len(nodes)} nodes")
    return nodes


def build_unified_dataset(
    squad_path: str = "data/raw/squad_v2.json",
    musique_path: str = "data/raw/musique.jsonl",
    twiki_path: str = "data/raw/2wiki.jsonl",
    metaqa_dir: str = "data/raw/metaqa",
    output_path: str = "data/processed/master_nodes.json"
):
    """Build master_nodes.json from SQuAD, MuSiQue, 2Wiki, and MetaQA."""
    nodes: List[StandardNode] = []

    # SQuAD
    squad_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    _download_if_missing(squad_url, squad_path)
    if os.path.exists(squad_path):
        log.info("Loading SQuAD...")
        nodes.extend(load_squad(squad_path))


    # MuSiQue
    musique_url = "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/musique/train.jsonl"
    _download_if_missing(musique_url, musique_path)
    if os.path.exists(musique_path):
        log.info("Loading MuSiQue...")
        nodes.extend(load_musique(musique_path))

    # 2WikiMultiHopQA
    twiki_url = "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/2wikimultihopqa/train.jsonl"
    _download_if_missing(twiki_url, twiki_path)
    if os.path.exists(twiki_path):
        log.info("Loading 2WikiMultiHopQA...")
        nodes.extend(load_2wiki(twiki_path))

    # MetaQA
    _download_metaqa(metaqa_dir)
    if os.path.exists(metaqa_dir):
        log.info("Loading MetaQA...")
        nodes.extend(load_metaqa(metaqa_dir))

    log.info(f"Total Unified Nodes: {len(nodes)}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_nodes(nodes, output_path)
    log.info(f"Saved to {output_path}")

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    squad = "data/raw/squad_v2.json"
    musique = "data/raw/musique.jsonl"
    twiki = "data/raw/2wiki.jsonl"
    metaqa = "data/raw/metaqa"
    output = "data/processed/master_nodes.json"
    build_unified_dataset(squad, musique, twiki, metaqa, output)
