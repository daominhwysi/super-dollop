import json
from typing import List, Dict, Any, Optional, Tuple
from sequence_labelling.llm.deepseek_client import chat

SYSTEM_PROMPT_LINKER = """You are a Specialized Entity Resolution and Graph Linker Agent for educational exam papers.
Your task is to analyze candidate summaries of Questions, Context Blocks, and Answer Keys, and emit lightweight JSON patch operations to link entities.

STRICT OUTPUT FORMAT RULES:
1. Output ONLY a valid JSON array of patch operations.
2. DO NOT re-emit question stems, option text, or passage prose.
3. Allowed Patch Operations:
   - {"op": "ADD_EDGE", "q_id": "...", "context_id": "...", "confidence": 0.99}
   - {"op": "SET_ANSWER", "q_id": "...", "answer": "A", "explanation": "..."}

Matching Rules:
- If a question contains an implicit trigger like "đoạn văn trên" or "according to passage 1", link it to the corresponding candidate context.
- Match answer keys from solution grids on appendix pages to their corresponding question IDs.

Example Output:
[
  {"op": "ADD_EDGE", "q_id": "doc1:p14:q15", "context_id": "doc1:p14:stim_1", "confidence": 0.99},
  {"op": "SET_ANSWER", "q_id": "doc1:p14:q15", "answer": "B", "explanation": "Explanation text..."}
]
"""

from sequence_labelling.config import LINKER_MODEL, LINKER_PROVIDER

class CompactGraphResolverAgent:
    """
    Executes entity graph linking by emitting lightweight patch operations (ADD_EDGE, SET_ANSWER)
    rather than re-generating full question text.
    """
    def __init__(self, model: Optional[str] = None, provider: Optional[str] = None):
        self.model = model or LINKER_MODEL
        self.provider = provider or LINKER_PROVIDER

    def resolve_and_apply_patches(
        self,
        questions: List[Dict[str, Any]],
        stimuli: Dict[str, str],
        solutions: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:

        if not questions:
            return questions, stimuli

        # 1. Prepare lightweight candidate summaries
        context_candidates = [
            {"id": stim_id, "text_snippet": text[:150] + "..." if len(text) > 150 else text}
            for stim_id, text in stimuli.items()
        ]

        question_candidates = [
            {
                "q_id": q.get("id"),
                "question_number": q.get("question_number"),
                "stem_snippet": q.get("stem", "")[:100],
                "current_stimulus_id": q.get("stimulus_id")
            }
            for q in questions
        ]

        payload = {
            "contexts": context_candidates,
            "questions": question_candidates,
            "solutions": solutions or []
        }

        # 2. Query Reasoning LLM for Patch Operations
        try:
            prompt = f"Analyze candidates and generate graph patches:\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
            response_text = chat(
                prompt=prompt,
                system=SYSTEM_PROMPT_LINKER,
                model=self.model,
                provider=self.provider
            )

            # Extract JSON array from response
            json_match = json.loads(response_text) if response_text.strip().startswith("[") else []
            if not json_match:
                import re
                match = re.search(r"\[\s*\{.*\}\s*\]", response_text, re.DOTALL)
                if match:
                    json_match = json.loads(match.group(0))

            # 3. Apply Patches in-memory
            q_map = {q["id"]: q for q in questions if "id" in q}
            for patch in json_match:
                op = patch.get("op")
                q_id = patch.get("q_id")
                if q_id not in q_map:
                    continue

                q = q_map[q_id]
                if op == "ADD_EDGE":
                    c_id = patch.get("context_id")
                    q["stimulus_id"] = c_id
                    if c_id in stimuli:
                        q["stimulus_text"] = stimuli[c_id]
                elif op == "SET_ANSWER":
                    if patch.get("answer"):
                        q["correct_answer"] = patch["answer"]
                    if patch.get("explanation"):
                        q["explanation"] = patch["explanation"]

        except Exception as e:
            print(f"[Graph Resolver Warning] Graph patch resolution failed ({e}), keeping local extractions.")

        return questions, stimuli
