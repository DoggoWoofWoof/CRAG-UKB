"""Scope addendum to PACKAGE_BYTES_LEDGER: the adoption record's `package_bytes_modified: 0` stands.

WHAT THE LEDGER GOT WRONG

  PACKAGE_BYTES_LEDGER.json measured 43,294,408,297 bytes moved in the package since the freeze
  (31.58 GB deleted in two cleanup passes + 11.71 GB written, mostly the retrieval caches) and
  labelled the adoption record's `package_bytes_modified: 0` as `current_value_is: FALSE`.

  That conflates two questions.  The field sits inside the adoption EXECUTION record
  (`adoption_execution_record_2026_09_08`), beside `verify_not_rebuild.held: true` and
  `sidecar_artifacts_only: true`.  It records what the adoption run itself did -- it read the
  package and wrote nothing -- and that was true on Sep 8 and is true now.  The 43.29 GB were
  moved by later sessions, after adoption closed.  So the 0 is correct in its own scope, the
  tests that pin it stay green, and the ledger's ANSWER is a different quantity: the package
  delta since the freeze, which belongs in a dated supersession record that cites the ledger
  by sha256 -- exactly what the ledger's own `consequence` line prescribes.

  Under the standing rule (a correction is a new record with its own hash, never an edit) the
  ledger is left byte-for-byte as it is, because its sha256 is already cited.  This record
  says which of its claims is withdrawn and which stands.

Run:
  python src/dataset_canonical/package_bytes_ledger_scope_addendum.py
"""
import datetime
import hashlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_manifest import record_hash  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, "data", "final_canonical")
LEDGER = os.path.join(FC, "PACKAGE_BYTES_LEDGER.json")
OUT = os.path.join(FC, "PACKAGE_BYTES_LEDGER_SCOPE_ADDENDUM.json")


def main():
    raw = open(LEDGER, "rb").read()
    led = json.loads(raw)
    sha = hashlib.sha256(raw).hexdigest()
    ans = led["ANSWER"]
    rec = {
        "RECORD": "PACKAGE_BYTES_LEDGER_SCOPE_ADDENDUM",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_what": "narrows the claim of PACKAGE_BYTES_LEDGER: its measurement stands, its verdict "
                 "on the adoption record's field does not",
        "ledger": {"file": "PACKAGE_BYTES_LEDGER.json", "sha256": sha,
                   "left_unedited_because": "its sha256 is cited by the adoption supersession "
                                            "record; a correction is a new record, never an edit"},
        "withdrawn": {
            "claim": "corrects.current_value_is = FALSE (and _what: 'the correct value for the "
                     "adoption record's package_bytes_modified field')",
            "why": "package_bytes_modified is a field of the adoption EXECUTION record, scoped "
                   "to that run's own I/O (beside verify_not_rebuild.held and "
                   "sidecar_artifacts_only). The run read the package and wrote nothing, so 0 "
                   "was and is true. The ledger measured a different quantity.",
        },
        "stands": {
            "measurement": "the ledger's ANSWER and its components, as measured",
            "package_bytes_moved_since_freeze": ans["package_bytes_modified"],
            "bytes_deleted": ans["bytes_deleted"],
            "bytes_written": ans["bytes_written"],
            "moved_by": "sessions after adoption closed: cleanup pass 1 and pass 2, the twelve "
                        "retrieval caches, the sidecar records",
            "where_it_belongs": "a dated adoption SUPERSESSION record in the adopting repository "
                                "that cites the ledger sha256 above -- not the adoption "
                                "execution record's own field",
        },
        "adoption_record_field": {"name": "package_bytes_modified", "value": 0,
                                  "verdict": "TRUE in its scope; unchanged"},
    }
    rec["RECORD_SHA256"] = record_hash(rec)
    rec["RECORD_SHA256_LF"] = record_hash(rec, crlf=False)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1))
    print("ledger sha256      %s" % sha)
    print("withdrawn          corrects.current_value_is=FALSE")
    print("stands             %d bytes moved since freeze (%d deleted + %d written)"
          % (ans["package_bytes_modified"], ans["bytes_deleted"], ans["bytes_written"]))
    print("adoption field     package_bytes_modified = 0  -> TRUE in its scope")
    print("wrote %s  RECORD_SHA256_LF %s" % (os.path.relpath(OUT, ROOT).replace("\\", "/"),
                                            rec["RECORD_SHA256_LF"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
