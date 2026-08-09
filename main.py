"""CLI for running a question and saving the complete trajectory."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent import Agent
from llm import configured_reasoner
from models import AgentTrace, IterationTrace
from retriever import Retriever


class DemoPrinter:
    def __init__(self, display: str) -> None:
        self.display = display
        self.lines: list[str] = []

    def emit(self, line: str = "") -> None:
        self.lines.append(line)
        print(line)

    def rule(self, char: str = "-") -> None:
        self.emit(char * 60)

    def header(self, question: str) -> None:
        if self.display == "client":
            self.rule("=")
            self.emit("AGENTIC DOCUMENT SEARCH")
            self.rule("=")
            self.emit()
            self.emit("QUESTION")
            self.emit(question)
        else:
            self.emit("QUESTION\n" + question)

    def iteration(self, item: IterationTrace) -> None:
        if self.display != "client":
            print_iteration(item, self.emit)
            return

        self.emit()
        self.rule()
        self.emit(f"ITERATION {item.iteration}")
        self.rule()
        self.emit()
        self.emit("SEARCH")
        self.emit(item.search_decision.search_query)
        self.emit()
        self.emit("WHY")
        self.emit(item.search_decision.reason)
        self.emit()
        self.emit("EVIDENCE")
        for section, page in _evidence_lines(item):
            self.emit(f"+ {section} - AWS-WAF p.{page}")
        result = "SUFFICIENT" if item.assessment.sufficient else "INSUFFICIENT"
        marker = "+" if item.assessment.sufficient else "x"
        self.emit()
        self.emit(f"ASSESSMENT  {marker} {result}")
        self.emit()
        self.emit("SUPPORTED")
        for supported in item.assessment.supported_information:
            self.emit(f"+ {_short_label(supported)}")
        if not item.assessment.supported_information:
            self.emit("None")
        self.emit()
        self.emit("MISSING")
        if item.assessment.missing_information:
            for missing in item.assessment.missing_information:
                self.emit(f"- {_short_label(missing)}")
        else:
            self.emit("None")
        if item.assessment.suggested_next_search:
            self.emit()
            self.emit("NEXT SEARCH")
            self.emit(item.assessment.suggested_next_search)

    def footer(self, trace: AgentTrace, destination: Path, tests_result: str | None) -> None:
        if self.display != "client":
            self.emit(f"\nSTOP REASON\n{trace.stop_reason}\n\nFINAL CITED ANSWER\n{trace.final_answer}")
            self.emit(f"\nTRACE SAVED: {destination}")
            return

        self.emit()
        self.rule("=")
        self.emit("STOP")
        self.emit(trace.stop_reason or "")
        self.rule("=")
        self.emit()
        self.emit("FINAL ANSWER")
        self.emit()
        self.emit(trace.final_answer or "")
        self.emit()
        self.rule()
        self.emit("TRACE")
        self.emit(f"Saved: {destination}")
        self.emit()
        self.emit("CITATIONS")
        if trace.citation_validation.valid:
            self.emit("+ All final citations validated against retrieved evidence")
        else:
            self.emit(f"x Citation validation found uncited claims: {trace.citation_validation.uncited_claims}")
        if tests_result:
            self.emit()
            self.emit("TESTS")
            self.emit(f"+ {tests_result}")
        self.rule("=")


def print_iteration(item: IterationTrace, emit=print) -> None:
    emit(f"\nITERATION {item.iteration}")
    emit(f"Search: {item.search_decision.search_query}")
    result = "SUFFICIENT" if item.assessment.sufficient else "INSUFFICIENT"
    emit(f"Assessment: {result} - {item.assessment.reason}")
    missing = "; ".join(item.assessment.missing_information) or "None"
    emit(f"Missing: {missing}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--traces-dir", default="traces")
    parser.add_argument("--trace-file")
    parser.add_argument("--display", choices=["standard", "client"], default="standard")
    parser.add_argument("--output-file")
    parser.add_argument("--tests-result")
    args = parser.parse_args()
    if not args.question.strip():
        raise SystemExit("ERROR: Question cannot be empty.")

    printer = DemoPrinter(args.display)
    printer.header(args.question)
    try:
        agent = Agent(Retriever(), configured_reasoner(), args.max_iterations, args.k)
        _, trace = agent.run(args.question, printer.iteration)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}") from error

    destination = _trace_destination(args)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(trace.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
    printer.footer(trace, destination, args.tests_result)

    if args.output_file:
        output = Path(args.output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(printer.lines) + "\n", encoding="utf-8")


def _trace_destination(args) -> Path:
    if args.trace_file:
        return Path(args.trace_file)
    traces_dir = Path(args.traces_dir)
    slug = re.sub(r"[^a-z0-9]+", "-", args.question.lower()).strip("-")[:55]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return traces_dir / f"{stamp}-{slug}.json"


def _evidence_lines(item: IterationTrace) -> list[tuple[str, int]]:
    seen = set()
    lines = []
    for chunk in item.retrieved:
        key = (chunk.section or "Unknown", chunk.page)
        if key in seen:
            continue
        seen.add(key)
        lines.append(key)
        if len(lines) == 4:
            break
    return lines


def _short_label(text: str) -> str:
    labels = {
        "reliability and failure preparation": "Reliability / failure preparation",
        "cost optimization and avoiding unnecessary spend": "Cost Optimization",
        "security": "Security",
    }
    return labels.get(text, text)


if __name__ == "__main__":
    main()
