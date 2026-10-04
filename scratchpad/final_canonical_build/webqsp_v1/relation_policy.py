"""Which relations may appear in semantic_text, and which are plumbing.

The user's rule: "Prefer informative relations and exclude obvious plumbing/schema relations.
Otherwise high-degree nodes will generate enormous, noisy text and hurt embeddings rather than
help them."

Freebase's edge mass is dominated by machinery, not meaning. Three relations alone
(notable_for.display_name, type.type.instance, topic_equivalent_webpage) are 67% of all
2.06B edges and none of them belongs in a sentence describing a node.

THE notable_for DECISION IS THE IMPORTANT ONE
  common.notable_for.display_name is 1,151,814,885 edges -- 52.54% of the whole graph -- because
  a notable_for CVT carries its label in ~37 languages. Selecting 8 of those at random would fill
  a CVT's entire budget with the same word in Danish, Malay and Georgian.

  It is dropped, and common.notable_for.object is kept instead. The object is the same fact
  reached through the entity rather than the literal: /m/0kyk rather than "Author"@da. That MID
  already carries an English display_name in the frozen overlay, so the language problem is
  solved by construction rather than by a language filter, and 52% of the candidate edges
  disappear from the pass.

DENY IS BY PREFIX, KEEP IS THE DEFAULT
  Freebase has ~1,700 domain namespaces and they are all fine. The plumbing is a short, closed
  list, so an explicit denylist stays honest as the schema grows: a relation nobody classified
  shows up as itself, which is legible, instead of vanishing silently.
"""

# Whole namespaces that exist to run Freebase, not to describe the world.
DENY_NAMESPACE = (
    "freebase.",       # valuenotation, object_hints, flag votes, user activity, domain profiles
    "dataworld.",      # gardening hints, mass data operations, provenance
    "pipeline.",       # curation tasks, merge/delete tasks, review flags
    "topic_server.",   # internal serving statistics
    "user.",           # user-space scratch schemas
    "usergroup.",
    "http://rdf.freebase.com/key/",   # identifiers, not prose (mined separately as keys)
    "http://rdf.freebase.com/ns/type.",
)

# Individual relations that are machinery inside otherwise meaningful namespaces.
DENY_RELATION = frozenset({
    # schema membership / reflection
    "type.type.instance", "type.object.type", "type.type.expected_by",
    "type.property.schema", "type.property.expected_type", "type.property.reverse_property",
    "type.property.master_property", "type.property.unique", "type.type.properties",
    "type.type.domain", "type.domain.types", "type.type.default_property",
    # access control
    "type.object.permission", "type.permission.controls", "type.usergroup.member",
    "type.permission.default_permission",
    # internal identifiers (handled by the key tier, not by prose)
    "type.object.id", "type.object.key", "type.object.guid",
    # blob storage metadata
    "type.content.blob_id", "type.content.media_type", "type.content.length",
    "type.content.text_encoding", "type.content.language", "type.content.source",
    "type.content.uploaded_by", "type.content_import.uri", "type.content_import.content",
    "type.content_import.header", "type.content_import.last_modified",
    # document bodies and source plumbing: multi-KB blobs, ruinous inside an embedding string
    "common.document.text", "common.document.source_uri", "common.document.content",
    "common.document.updated", "common.topic.article",
    # link farms
    "common.topic.topic_equivalent_webpage", "common.webpage.uri", "common.webpage.category",
    "common.webpage.in_index", "common.webpage.description",
    "common.licensed_object.license", "common.topic.image",
    "common.image.appears_in_topic_gallery", "common.image.size", "common.image.media_type",
    # see module docstring: superseded by common.notable_for.object
    "common.notable_for.display_name",
    # the predicate points at a schema property node, so it renders as
    # "predicate: /type/object/type". common.notable_for.object carries the same fact as a named
    # entity ("Musical Recording"), which is the readable half of the pair.
    "common.notable_for.predicate",
})

# Relations whose object is a literal that IS the node's human-readable identity. These are the
# one-hop label tier: a word sense has no type.object.name, but base.wordnet.word_sense.word
# names it exactly, and that is source-attested Freebase data, not an inference.
LABEL_BEARING = frozenset({
    "base.wordnet.word.word", "base.wordnet.word_sense.word",
    "base.wordnet.synset.gloss", "base.wordnet.synset.word",
    "base.skosbase.skos_concept.pref_label", "base.skosbase.skos_concept.alt_label",
    "base.schemastaging.context_name.short_name",
    "biology.organism_classification.scientific_name",
    "medicine.drug_label_section.section_name",
    "common.notable_for.object", "common.topic.notable_types",
})


def classify(relation):
    """-> 'DENY' | 'LABEL' | 'KEEP'. Pure function of the relation string."""
    r = str(relation)
    if r in DENY_RELATION:
        return "DENY"
    for p in DENY_NAMESPACE:
        if r.startswith(p):
            return "DENY"
    if r in LABEL_BEARING:
        return "LABEL"
    return "KEEP"


def leaf(relation):
    """people.marriage.spouse -> 'spouse'. The role is the last path segment."""
    r = str(relation)
    if r.startswith("http://"):
        r = r.rsplit("/", 1)[-1]
    return r.rsplit(".", 1)[-1].replace("_", " ") or "related"
