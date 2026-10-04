"""Apply the CRAG checkpoint fork to a pristine KaHIP/FREIGHT checkout (additive, exact-anchor patches).

    python apply_fork.py <FREIGHT_ROOT>

Every edit replaces one exact upstream anchor (asserted to occur exactly once) or appends; running twice is a
no-op (marker check).  The stock targets (freight_con / freight_cut / converters) keep their behaviour -- the
patches only add accessors, two NULL-by-default hooks, and two new targets freight_con_ckpt / freight_cut_ckpt.
Run on Windows against the \\\\wsl$ share or inside WSL with python3; writes FORK_APPLIED.json next to CMakeLists.txt.
"""
import hashlib
import io
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "CRAG ckpt fork"


def rd(p):
    with io.open(p, "r", encoding="utf-8", newline="") as f:
        return f.read()


def wr(p, s):
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def patch(root, rel, old, new, once=True):
    p = os.path.join(root, rel)
    s = rd(p)
    crlf = s.count("\r\n")
    if crlf and crlf == s.count("\n"):            # upstream graph_io_stream.h is CRLF throughout: match its line endings
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
    if new in s:
        return "already"
    n = s.count(old)
    if n != 1:
        raise RuntimeError("%s: anchor found %d times (expected 1):\n%s" % (rel, n, old))
    wr(p, s.replace(old, new))
    return "patched"


def main(root):
    hg = os.path.join(root, "code_for_hypergraphs")
    res = {}
    # 1. graph_io_stream.h: hooks + shard-aware getline
    res["graph_io_stream.h/hooks"] = patch(root, "code_for_hypergraphs/lib/io/graph_io_stream.h",
        "class graph_io_stream {\n        public:\n                graph_io_stream();\n",
        "class graph_io_stream {\n        public:\n                // %s: shard hooks (NULL = upstream behaviour)\n"
        "                static bool (*ckpt_shard_hook)(PartitionConfig&);\n"
        "                static bool (*ckpt_eval_hook)(std::ifstream&, bool);\n\n                graph_io_stream();\n" % MARK)
    res["graph_io_stream.h/getline"] = patch(root, "code_for_hypergraphs/lib/io/graph_io_stream.h",
        "\t\twhile( node_counter < num_lines) {\n\t\t\tstd::getline(*(partition_config.stream_in),(*lines)[0]);\n\t\t\tif ((*lines)[0][0] == '%') {\n\t\t\t\tcontinue;\n\t\t\t}\n",
        "\t\twhile( node_counter < num_lines) {\n"
        "\t\t\tif (!std::getline(*(partition_config.stream_in),(*lines)[0])) { // %s: end of a shard -> next shard\n"
        "\t\t\t\tif (graph_io_stream::ckpt_shard_hook == NULL || !graph_io_stream::ckpt_shard_hook(partition_config)) {\n"
        "\t\t\t\t\tstd::cerr << \"stream exhausted before all nodes were read\" << std::endl; exit(1);\n"
        "\t\t\t\t}\n\t\t\t\t(*lines)[0].clear();\n\t\t\t\tcontinue;\n\t\t\t}\n"
        "\t\t\tif ((*lines)[0][0] == '%%') {\n\t\t\t\tcontinue;\n\t\t\t}\n" % MARK)
    # 2. graph_io_stream.cpp: hook definitions + shard-aware evaluation
    res["graph_io_stream.cpp/defs"] = patch(root, "code_for_hypergraphs/lib/io/graph_io_stream.cpp",
        "#include \"graph_io_stream.h\"\n#include \"timer.h\"\n",
        "#include \"graph_io_stream.h\"\n#include \"timer.h\"\n\n// %s\nbool (*graph_io_stream::ckpt_shard_hook)(PartitionConfig&) = NULL;\n"
        "bool (*graph_io_stream::ckpt_eval_hook)(std::ifstream&, bool) = NULL;\n" % MARK)
    res["graph_io_stream.cpp/eval"] = patch(root, "code_for_hypergraphs/lib/io/graph_io_stream.cpp",
        "\tfor (long node = 0; node < nmbNodes; node++) {\n\t\tstd::vector<LongNodeID>* line_numbers_ptr;\n\t\tif (use_cached) {\n"
        "\t\t\tline_numbers_ptr = &(*cached_input)[node];\n\t\t} else {\n\t\t\tif (!std::getline(in, (*lines)[0])) break;\n",
        "\tif (!use_cached && graph_io_stream::ckpt_eval_hook != NULL) graph_io_stream::ckpt_eval_hook(in, true); // %s\n"
        "\tfor (long node = 0; node < nmbNodes; node++) {\n\t\tstd::vector<LongNodeID>* line_numbers_ptr;\n\t\tif (use_cached) {\n"
        "\t\t\tline_numbers_ptr = &(*cached_input)[node];\n\t\t} else {\n\t\t\tif (!std::getline(in, (*lines)[0])) {\n"
        "\t\t\t\tif (graph_io_stream::ckpt_eval_hook != NULL && graph_io_stream::ckpt_eval_hook(in, false)) { node--; continue; }\n"
        "\t\t\t\tbreak;\n\t\t\t}\n" % MARK)
    # 3. random_functions.h: PRNG accessors
    res["random_functions.h/include"] = patch(root, "code_for_hypergraphs/lib/tools/random_functions.h",
        "#include <random>\n#include <vector>\n", "#include <random>\n#include <vector>\n#include <sstream>\n#include <string>\n")
    res["random_functions.h/static"] = patch(root, "code_for_hypergraphs/lib/tools/random_functions.h",
        "        private:\n                static int m_seed;\n",
        "\t\t// %s: PRNG state accessors\n\t\tstatic uint32_t ckpt_get_rand_counter() { return rand_counter; }\n"
        "\t\tstatic void ckpt_set_rand_counter(uint32_t v) { rand_counter = v; }\n\t\tstatic int ckpt_get_seed() { return m_seed; }\n"
        "\t\tstatic std::string ckpt_get_mt_state() { std::stringstream ss; ss << m_mt; return ss.str(); }\n"
        "\t\tstatic void ckpt_set_mt_state(const std::string& s) { std::stringstream ss(s); ss >> m_mt; }\n\n"
        "        private:\n                static int m_seed;\n" % MARK)
    res["random_functions.h/fastRandBool"] = patch(root, "code_for_hypergraphs/lib/tools/random_functions.h",
        "\t\t\tprivate:\n\t\t\t\tstatic constexpr const U s_mask_left1",
        "\t\t\t\tU ckpt_get() const { return m_rand; }   // %s\n\t\t\t\tvoid ckpt_set(U v) { m_rand = v; }\n"
        "\t\t\tprivate:\n\t\t\t\tstatic constexpr const U s_mask_left1" % MARK)
    # 4. self_sorting_monotonic_vector.h: state export / import
    res["ssmv.h/include"] = patch(root, "code_for_hypergraphs/lib/data_structure/priority_queues/self_sorting_monotonic_vector.h",
        "#include <vector>\n#include <list>\n", "#include <vector>\n#include <list>\n#include <map>\n#include <iterator>\n")
    res["ssmv.h/methods"] = patch(root, "code_for_hypergraphs/lib/data_structure/priority_queues/self_sorting_monotonic_vector.h",
        "\t\tK size() const;\n",
        "\t\tK size() const;\n"
        "\t\t// %s: exact state export / import (bucket list in list order; every element's bucket by index)\n"
        "\t\tvoid ckpt_export(std::vector<K>& keys, std::vector<K>& elem_bucket, std::vector<K>& posv, std::vector<K>& bsize,\n"
        "\t\t                 std::vector<V>& bvalue, std::vector<K>& bpos) const {\n"
        "\t\t\tstd::map<const BUCKET<K,V>*, K> idx; K i = 0;\n"
        "\t\t\tbsize.clear(); bvalue.clear(); bpos.clear();\n"
        "\t\t\tfor (auto it = bucket_list.begin(); it != bucket_list.end(); ++it, ++i) {\n"
        "\t\t\t\tidx[&*it] = i; bsize.push_back(it->size); bvalue.push_back(it->value); bpos.push_back(it->pos_begin);\n\t\t\t}\n"
        "\t\t\tkeys.resize(ordered_list.size()); elem_bucket.resize(ordered_list.size());\n"
        "\t\t\tfor (size_t j = 0; j < ordered_list.size(); j++) { keys[j] = ordered_list[j].key; elem_bucket[j] = idx.at(&*(ordered_list[j].buck)); }\n"
        "\t\t\tposv = pos;\n\t\t}\n"
        "\t\tvoid ckpt_import(K n, const std::vector<K>& keys, const std::vector<K>& elem_bucket, const std::vector<K>& posv,\n"
        "\t\t                 const std::vector<K>& bsize, const std::vector<V>& bvalue, const std::vector<K>& bpos) {\n"
        "\t\t\tn_elements = n; bucket_list.clear();\n"
        "\t\t\tstd::vector<typename std::list<BUCKET<K,V>>::iterator> its;\n"
        "\t\t\tfor (size_t i = 0; i < bsize.size(); i++) { BUCKET<K,V> b = {bsize[i], bvalue[i], bpos[i]}; bucket_list.push_back(b); its.push_back(std::prev(bucket_list.end())); }\n"
        "\t\t\tordered_list.resize(n); pos = posv;\n"
        "\t\t\tfor (K j = 0; j < n; j++) { ordered_list[j].key = keys[j]; ordered_list[j].buck = its.at(elem_bucket[j]); }\n\t\t}\n" % MARK)
    # 5. vertex_partitioning.h: accessors to protected state
    res["vertex_partitioning.h/accessors"] = patch(root, "code_for_hypergraphs/lib/partition/onepass_partitioning/vertex_partitioning.h",
        "\t\tstd::vector<floating_block> blocks;\n        protected:\n",
        "\t\tstd::vector<floating_block> blocks;\n"
        "\t\t// %s: accessors to the mutable state that lives in this object\n"
        "\t\tself_sorting_monotonic_vector<PartitionID, NodeWeight>& ckpt_sorted_blocks() { return sorted_blocks; }\n"
        "\t\tLongNodeID& ckpt_amortized_rounds() { return amortized_rounds_for_feasibility_sampling; }\n"
        "\t\trandom_functions::fastRandBool<uint64_t>& ckpt_random_obj() { return random_obj; }\n"
        "\t\tint ckpt_sampling() const { return sampling; }\n"
        "        protected:\n" % MARK)
    # 6. new app + targets
    for fn in ("freight_ckpt.cpp", "ckpt.h"):
        dst = os.path.join(hg, "app", fn)
        src = os.path.join(HERE, fn)
        if not os.path.exists(dst) or sha(dst) != sha(src):
            shutil.copyfile(src, dst); res["app/" + fn] = "copied"
        else:
            res["app/" + fn] = "already"
    cm = os.path.join(root, "CMakeLists.txt")
    s = rd(cm)
    if MARK not in s:
        s += ("\n# %s\n"
              "add_executable(freight_con_ckpt ${HG}/app/freight_ckpt.cpp\n"
              "  $<TARGET_OBJECTS:hg_kaffpa> $<TARGET_OBJECTS:hg_mapping> $<TARGET_OBJECTS:hg_streampart>)\n"
              "target_include_directories(freight_con_ckpt PRIVATE ${HG_INCLUDES})\n"
              "target_compile_definitions(freight_con_ckpt PRIVATE \"-DMODE_NETLIST\" \"-DMODE_FREIGHT\" \"-DMODE_CONNECTIVITY\")\n"
              "target_link_libraries(freight_con_ckpt ${OpenMP_CXX_LIBRARIES})\n"
              "install(TARGETS freight_con_ckpt DESTINATION bin)\n"
              "add_executable(freight_cut_ckpt ${HG}/app/freight_ckpt.cpp\n"
              "  $<TARGET_OBJECTS:hg_kaffpa> $<TARGET_OBJECTS:hg_mapping> $<TARGET_OBJECTS:hg_streampart>)\n"
              "target_include_directories(freight_cut_ckpt PRIVATE ${HG_INCLUDES})\n"
              "target_compile_definitions(freight_cut_ckpt PRIVATE \"-DMODE_NETLIST\" \"-DMODE_FREIGHT\")\n"
              "target_link_libraries(freight_cut_ckpt ${OpenMP_CXX_LIBRARIES})\n"
              "install(TARGETS freight_cut_ckpt DESTINATION bin)\n" % MARK)
        wr(cm, s); res["CMakeLists.txt"] = "appended"
    else:
        res["CMakeLists.txt"] = "already"
    rec = {"RECORD": "FREIGHT_FORK_APPLIED", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "root": root, "edits": res,
           "patched_files_sha256": {rel: sha(os.path.join(root, rel)) for rel in (
               "code_for_hypergraphs/lib/io/graph_io_stream.h", "code_for_hypergraphs/lib/io/graph_io_stream.cpp",
               "code_for_hypergraphs/lib/tools/random_functions.h",
               "code_for_hypergraphs/lib/data_structure/priority_queues/self_sorting_monotonic_vector.h",
               "code_for_hypergraphs/lib/partition/onepass_partitioning/vertex_partitioning.h",
               "code_for_hypergraphs/app/freight_ckpt.cpp", "code_for_hypergraphs/app/ckpt.h", "CMakeLists.txt")},
           "fork_sources_sha256": {fn: sha(os.path.join(HERE, fn)) for fn in ("freight_ckpt.cpp", "ckpt.h", "apply_fork.py")}}
    wr(os.path.join(root, "FORK_APPLIED.json"), json.dumps(rec, indent=1))
    for k, v in res.items():
        print("%-40s %s" % (k, v))
    return rec


if __name__ == "__main__":
    main(sys.argv[1])
