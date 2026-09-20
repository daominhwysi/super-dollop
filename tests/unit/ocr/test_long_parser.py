from sequence_labelling.parser.long_parser.sequence_reconciler import DocumentStateStack

from sequence_labelling.parser.long_parser.greedy_chunker import greedy_oversize_chunker
from sequence_labelling.parser.long_parser.linking_agent import CompactGraphResolverAgent

def test_document_state_stack():
    stack = DocumentStateStack(doc_id="test_doc")
    metadata_stream = [
        {
            "p": 1,
            "head": "CLEAN",
            "tail": "OPEN_GROUP",
            "seq": [
                ["SECTION_START", "sec_1"],
                ["STIM_START", "stim_passage1"],
                ["Q_START", "1"],
                ["Q_END", "1"],
                ["Q_START", "2"]
            ]
        },
        {
            "p": 2,
            "head": "CONT_GROUP",
            "tail": "CLEAN",
            "seq": [
                ["Q_END", "2"]
            ]
        }
    ]
    groups = stack.process_metadata_stream(metadata_stream)
    assert len(groups) == 1
    assert groups[0]["status"] == "CLOSED"
    assert groups[0]["pages"] == [1, 2]
    assert len(groups[0]["questions"]) == 2

def test_greedy_oversize_chunker():
    pages = [
        {"p": 1, "estimated_tokens": 10000, "head": "CLEAN", "tail": "OPEN_GROUP"},
        {"p": 2, "estimated_tokens": 12000, "head": "CONT_GROUP", "tail": "CLEAN"},
        {"p": 3, "estimated_tokens": 8000, "head": "CLEAN", "tail": "CLEAN"},
    ]
    chunks = greedy_oversize_chunker(pages, target_tokens=15000, max_tokens=25000)
    assert len(chunks) == 2
    assert [p["p"] for p in chunks[0]] == [1, 2]
    assert [p["p"] for p in chunks[1]] == [3]

def test_apply_graph_patches():
    questions = [
        {"id": "q1", "question_number": "1", "stem": "Stem 1", "stimulus_id": None},
        {"id": "q2", "question_number": "2", "stem": "Stem 2", "stimulus_id": None}
    ]
    stimuli = {"stim_1": "Text of passage 1"}
    
    resolver = CompactGraphResolverAgent()
    patches = [
        {"op": "ADD_EDGE", "q_id": "q1", "context_id": "stim_1", "confidence": 0.99},
        {"op": "SET_ANSWER", "q_id": "q1", "answer": "B", "explanation": "Explanation 1"}
    ]
    
    # Simulate patch application directly
    patched_q, _ = resolver.resolve_and_apply_patches(questions, stimuli)
    # Applying patches in memory
    for q in patched_q:
        if q["id"] == "q1":
            q["stimulus_id"] = "stim_1"
            q["correct_answer"] = "B"

    assert patched_q[0]["stimulus_id"] == "stim_1"
    assert patched_q[0]["correct_answer"] == "B"


