"""Command-line interface for synthetic exam generation and reconstruction."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from synthetic_exam_generator.generator import Subject
from synthetic_exam_generator.curriculum import generate_curriculum, generate_all_curricula
from synthetic_exam_generator.exam_compiler import run_batch_exams_generator
from synthetic_exam_generator.reconstructor import (
    reconstruct_exam,
    ReconstructorConfig,
    OPTION_PREFIX_STYLES,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthetic_exam_generator",
        description="Synthetic Exam Generator & Sequence Labelling Reconstructor for Vietnamese Education",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # 1. curriculum
    p_curr = subparsers.add_parser("curriculum", help="Generate curriculum JSON files")
    p_curr.add_argument("--all", action="store_true", help="Generate curricula for all subjects & grades concurrently")
    p_curr.add_argument("--subject", type=str, help="Subject slug (e.g. 'physics', 'biology')")
    p_curr.add_argument("--grade", type=int, choices=[8, 9, 10, 11, 12], help="Grade level (e.g. 10, 11, 12)")
    p_curr.add_argument("--model", type=str, default=None, help="LLM model to use")
    p_curr.add_argument("--provider", type=str, choices=["codex", "deepseek", "nvidia", "vilao", "xah", "commandcode", "agy"], default=None, help="LLM provider")
    p_curr.add_argument("--thinking", type=str, default="low", help="Thinking effort level")
    p_curr.add_argument("-c", "--concurrency", type=int, default=4, help="Number of parallel workers")

    # 2. exam
    p_exam = subparsers.add_parser("exam", help="Generate mock exams as compiled JSON")
    p_exam.add_argument("-n", "--num-exams", type=int, default=10, help="Number of exams to generate")
    p_exam.add_argument("-o", "--output-dir", type=str, default="data/synthetic_exams", help="Output directory path")
    p_exam.add_argument("--model", type=str, default=None, help="LLM model to use")
    p_exam.add_argument("--provider", type=str, choices=["codex", "deepseek", "nvidia", "vilao", "xah", "commandcode", "agy"], default=None, help="LLM provider")
    p_exam.add_argument("--thinking", type=str, default="low", help="Thinking effort level")
    p_exam.add_argument("-c", "--concurrency", type=int, default=2, help="Number of concurrent threads per exam")
    p_exam.add_argument("--subject", type=str, help="Filter generation for a specific subject")
    p_exam.add_argument("--grade", type=int, help="Filter generation for a specific grade")

    # 3. reconstruct
    p_rec = subparsers.add_parser("reconstruct", help="Reconstruct raw text and track spans from exam JSONs")
    p_rec.add_argument("-i", "--input-dir", type=str, default="data/synthetic_exams", help="Input directory containing exam JSON files")
    p_rec.add_argument("-o", "--output-dir", type=str, default="data/synthetic_exams/xml", help="Destination folder for reconstructed XML files")
    p_rec.add_argument("--opt-style", type=str, choices=list(OPTION_PREFIX_STYLES.keys()), default=None, help="Override option prefix style")
    p_rec.add_argument("--typo-rate", type=float, default=0.0, help="Typo noise injection rate")
    p_rec.add_argument("--space-noise-rate", type=float, default=0.0, help="Spacing noise injection rate")
    p_rec.add_argument("--latex-mask-prob", type=float, default=0.0, help="LaTeX masking probability")
    p_rec.add_argument("--latex-placeholder", type=str, default="[LATEX]", help="LaTeX masking placeholder")
    p_rec.add_argument("--option-drop-prob", type=float, default=0.0, help="Option drop probability")
    p_rec.add_argument("--casing-noise-prob", type=float, default=0.0, help="Casing noise probability")
    p_rec.add_argument("--synonym-swap-prob", type=float, default=0.0, help="Synonym swap probability")
    p_rec.add_argument("--formatting-noise-prob", type=float, default=0.0, help="Formatting tag noise probability")

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "curriculum":
        if args.all:
            generate_all_curricula(
                model=args.model,
                thinking=args.thinking,
                concurrency=args.concurrency,
                provider=args.provider,
            )
        else:
            if not args.subject or not args.grade:
                print("Error: Single curriculum generation requires both --subject and --grade, or use --all.")
                return 1
            generate_curriculum(
                subject=args.subject,
                grade=args.grade,
                model=args.model,
                thinking=args.thinking,
                provider=args.provider,
            )
        return 0

    elif args.command == "exam":
        run_batch_exams_generator(
            num_exams=args.num_exams,
            output_dir=args.output_dir,
            model=args.model,
            thinking=args.thinking,
            concurrency=args.concurrency,
            subject=args.subject,
            grade=args.grade,
            provider=args.provider,
        )
        return 0

    elif args.command == "reconstruct":
        import json
        in_path = Path(args.input_dir)
        out_path = Path(args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        cfg = ReconstructorConfig(
            option_prefix_style=args.opt_style,
            typo_rate=args.typo_rate,
            space_noise_rate=args.space_noise_rate,
            latex_mask_prob=args.latex_mask_prob,
            latex_placeholder=args.latex_placeholder,
            option_drop_prob=args.option_drop_prob,
            casing_noise_prob=args.casing_noise_prob,
            synonym_swap_prob=args.synonym_swap_prob,
            formatting_noise_prob=args.formatting_noise_prob,
        )

        exam_files = list(in_path.glob("*.json"))
        print(f"Reconstructing {len(exam_files)} exams from {in_path} -> {out_path}...")
        success = 0
        for f in exam_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if "sections" not in data:
                    continue
                result = reconstruct_exam(data, cfg)
                out_file = out_path / f"{f.stem}_annotated.xml"
                out_file.write_text(result["raw_xml"], encoding="utf-8")
                success += 1
            except Exception as e:
                print(f"Error reconstructing {f.name}: {e}")

        print(f"Successfully reconstructed {success}/{len(exam_files)} exams.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
