#!/usr/bin/env python3
import json
from pathlib import Path

def main():
    data_path = Path("backend/logs/artifact/passed_documents_token_det_f1_benchmark.json")
    data = json.loads(data_path.read_text(encoding="utf-8"))

    pass_data = data["PASS"]
    print("=" * 90)
    print("1. TOKEN STATISTICS IN PASSED DOCUMENTS (409 Docs)")
    print("=" * 90)
    tok = pass_data["token_statistics"]
    for metric, stats in tok.items():
        t = stats["total"]
        m = stats["mean"]
        med = stats["median"]
        mi = stats["min"]
        ma = stats["max"]
        print(f"  {metric:<15}: Total={t:>12,d} | Mean={m:>9.1f} | Median={med:>9.1f} | Min={mi:>7,d} | Max={ma:>9,d}")

    print("\n" + "=" * 90)
    print("2. DETERMINISTIC PARSER (DET) F1 BENCHMARK ON PASSED DOCUMENTS (409 Docs)")
    print("=" * 90)
    h_cnt = pass_data["high_confidence_count"]
    d_cnt = pass_data["doc_count"]
    h_rate = pass_data["high_confidence_rate"]
    print(f"High Confidence Pass Rate: {h_cnt}/{d_cnt} ({h_rate:.2f}%)\n")
    print(f"{'Tag / Label':<20} | {'TP':>7} | {'FP':>6} | {'FN':>6} | {'Gold':>7} | {'Pred':>7} | {'Prec (%)':>9} | {'Rec (%)':>8} | {'F1 (%)':>7}")
    print("-" * 90)
    for lbl, s in pass_data["span_f1_metrics"].items():
        tp = s["tp"]
        fp = s["fp"]
        fn = s["fn"]
        g_cnt = s["gold_cnt"]
        p_cnt = s["pred_cnt"]
        prec = s["precision"]
        rec = s["recall"]
        f1 = s["f1"]
        print(f"{lbl:<20} | {tp:>7d} | {fp:>6d} | {fn:>6d} | {g_cnt:>7d} | {p_cnt:>7d} | {prec:>9.2f} | {rec:>8.2f} | {f1:>7.2f}")

    print("\n" + "=" * 90)
    print("3. CATEGORY-BY-CATEGORY BREAKDOWN (PASS SUBSET)")
    print("=" * 90)
    print(f"{'Category':<32} | {'Docs':>4} | {'Raw Tok (cl100k)':>16} | {'HighConf (%)':>12} | {'QL F1':>7} | {'OL F1':>7} | {'Stem F1':>8} | {'OT F1':>7}")
    print("-" * 110)
    for cat, cdata in pass_data["category_breakdown"].items():
        d_c = cdata["doc_count"]
        r_t = cdata["raw_tokens_cl100k"]
        hc = cdata["high_conf_rate"]
        ql_f1 = cdata["spans"]["question_label"]["f1"]
        ol_f1 = cdata["spans"]["option_label"]["f1"]
        s_f1 = cdata["spans"]["stem"]["f1"]
        ot_f1 = cdata["spans"]["option_text"]["f1"]
        print(f"{cat:<32} | {d_c:>4d} | {r_t:>16,d} | {hc:>11.1f}% | {ql_f1:>6.2f}% | {ol_f1:>6.2f}% | {s_f1:>7.2f}% | {ot_f1:>6.2f}%")

    print("\n" + "=" * 90)
    print("4. COMPARISON ACROSS DATASET QUALITY TIERS")
    print("=" * 90)
    for sname, sdata in data.items():
        cnt = sdata["doc_count"]
        raw_tok = sdata["token_statistics"]["raw_cl100k"]["total"]
        xml_tok = sdata["token_statistics"]["xml_cl100k"]["total"]
        raw_words = sdata["token_statistics"]["raw_words"]["total"]
        h_r = sdata["high_confidence_rate"]
        ql_f1 = sdata["span_f1_metrics"]["question_label"]["f1"]
        ol_f1 = sdata["span_f1_metrics"]["option_label"]["f1"]
        stem_f1 = sdata["span_f1_metrics"]["stem"]["f1"]
        print(f"[{sname:<35}] Docs: {cnt:>3d} | Raw Tok: {raw_tok:>9,d} | XML Tok: {xml_tok:>9,d} | Words: {raw_words:>9,d} | HighConf: {h_r:>5.1f}% | QL F1: {ql_f1:>5.2f}% | OL F1: {ol_f1:>5.2f}% | Stem F1: {stem_f1:>5.2f}%")

if __name__ == "__main__":
    main()
