"""
run_eval.py: run the Group 8 pun test set through PunBot and fill in the workbook.

What it does
  1. Detection: sends each sentence on the "Test Set" tab to PunBot exactly as a
     user would, reads back the Pun / Non-pun classification, and writes it to
     the "Pipeline says" column (J by default). Column K then scores itself.
  2. Dialog: for each row on the "Dialog Questions" tab, replays the pun's
     analysis as conversation history and asks the question as a follow-up.
     The answer goes into a new "Chatbot answer" column for the raters.
  3. Adds a "Chatbot Output" tab with the full analysis, parsed pivot word and
     classification for every sentence, so failures can be inspected.

The input workbook is never modified; results go to a new file.

Usage (from the project root, with ANTHROPIC_API_KEY set in .env):
    python -m evaluation.run_eval
    python -m evaluation.run_eval --only detection --limit 3     # quick smoke test
    python -m evaluation.run_eval --column M                      # write predictions elsewhere
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import column_index_from_string, get_column_letter

from backend.analysis import parse_classification, parse_pivot
from backend.config import config
from backend.llm_adapter import LLMAdapter
from backend.prompts import SYSTEM_PROMPT

HERE = Path(__file__).parent
DEFAULT_INPUT = HERE / "Group8_Pun_Test_Set.xlsx"
DEFAULT_OUTPUT_DIR = HERE / "results"

TEST_SHEET = "Test Set"
DIALOG_SHEET = "Dialog Questions"
OUTPUT_SHEET = "Chatbot Output"
EVAL_TEMPERATURE = 0.0  # as deterministic as the model allows, so reruns are comparable

HEADER_FONT = Font(bold=True)
HEADER_FILL = PatternFill("solid", fgColor="D9D9D9")
WRAP_TOP = Alignment(wrap_text=True, vertical="top")


def find_column(ws, header_prefix: str) -> Optional[int]:
    """1-based index of the first header cell starting with header_prefix (case-insensitive)."""
    for cell in ws[1]:
        if isinstance(cell.value, str) and cell.value.strip().lower().startswith(header_prefix.lower()):
            return cell.column
    return None


def read_test_set(ws) -> List[Dict]:
    id_col, sent_col = find_column(ws, "ID"), find_column(ws, "Sentence")
    label_col, target_col = find_column(ws, "Label"), find_column(ws, "Target word")
    rows = []
    for r in range(2, ws.max_row + 1):
        item_id = ws.cell(r, id_col).value
        sentence = ws.cell(r, sent_col).value
        if not item_id or not sentence:
            continue
        rows.append({
            "row": r,
            "id": str(item_id).strip(),
            "sentence": str(sentence).strip(),
            "label": ws.cell(r, label_col).value if label_col else None,
            "target": ws.cell(r, target_col).value if target_col else None,
        })
    return rows


def read_dialog_questions(ws) -> List[Dict]:
    qid_col, pid_col = find_column(ws, "QID"), find_column(ws, "Pun ID")
    q_col = find_column(ws, "Question")
    rows = []
    for r in range(2, ws.max_row + 1):
        qid, pid, question = ws.cell(r, qid_col).value, ws.cell(r, pid_col).value, ws.cell(r, q_col).value
        if not (qid and pid and question):
            continue  # skips the blank and totals rows under the table
        rows.append({"row": r, "qid": str(qid).strip(), "pun_id": str(pid).strip(), "question": str(question).strip()})
    return rows


def run_detection(llm: LLMAdapter, items: List[Dict]) -> Dict[str, Dict]:
    results = {}
    for n, item in enumerate(items, 1):
        answer = llm.generate(SYSTEM_PROMPT, history=[], user_message=item["sentence"], temperature=EVAL_TEMPERATURE)
        prediction = parse_classification(answer)
        results[item["id"]] = {
            "sentence": item["sentence"],
            "label": item["label"],
            "target": item["target"],
            "prediction": prediction,
            "pivot": parse_pivot(answer),
            "analysis": answer,
        }
        mark = "ok" if prediction == item["label"] else ("??" if prediction is None else "XX")
        print(f"  [{n:>2}/{len(items)}] {item['id']}  truth={item['label']:<8} bot={prediction or 'UNPARSED':<8} {mark}")
    return results


def run_dialog(llm: LLMAdapter, questions: List[Dict], test_items: Dict[str, Dict], analyses: Dict[str, Dict]) -> Dict[str, str]:
    answers = {}
    for n, q in enumerate(questions, 1):
        item = test_items.get(q["pun_id"])
        if item is None:
            answers[q["qid"]] = f"(Pun ID {q['pun_id']} not found on the Test Set tab)"
            continue
        if q["pun_id"] not in analyses:  # dialog-only run: analyze the pun first
            first = llm.generate(SYSTEM_PROMPT, history=[], user_message=item["sentence"], temperature=EVAL_TEMPERATURE)
            analyses[q["pun_id"]] = {"analysis": first}
        history = [
            {"role": "user", "content": item["sentence"]},
            {"role": "assistant", "content": analyses[q["pun_id"]]["analysis"]},
        ]
        answers[q["qid"]] = llm.generate(SYSTEM_PROMPT, history=history, user_message=q["question"], temperature=EVAL_TEMPERATURE)
        print(f"  [{n:>2}/{len(questions)}] {q['qid']} ({q['pun_id']}): {q['question'][:60]}")
    return answers


def write_detection(wb, items: List[Dict], results: Dict[str, Dict], column: str) -> None:
    ws = wb[TEST_SHEET]
    col = column_index_from_string(column)
    for item in items:
        res = results.get(item["id"])
        if res and res["prediction"]:
            ws.cell(item["row"], col, res["prediction"])

    # Full output tab for inspecting failures
    if OUTPUT_SHEET in wb.sheetnames:
        del wb[OUTPUT_SHEET]
    out = wb.create_sheet(OUTPUT_SHEET)
    headers = ["ID", "Sentence", "Label (ground truth)", "Chatbot classification", "Match?",
               "Target word (ground truth)", "Chatbot pivot", "Full chatbot analysis"]
    out.append(headers)
    for item in items:
        res = results.get(item["id"])
        if not res:
            continue
        match = "" if res["prediction"] is None else ("Yes" if res["prediction"] == res["label"] else "No")
        out.append([item["id"], res["sentence"], res["label"], res["prediction"] or "UNPARSED", match,
                    res["target"], res["pivot"], res["analysis"]])
    style_sheet(out, widths=[7, 50, 14, 14, 8, 16, 16, 90])


def write_dialog(wb, questions: List[Dict], answers: Dict[str, str]) -> None:
    ws = wb[DIALOG_SHEET]
    col = find_column(ws, "Chatbot answer") or ws.max_column + 1
    header = ws.cell(1, col, "Chatbot answer")
    header.font, header.fill, header.alignment = HEADER_FONT, HEADER_FILL, WRAP_TOP
    ws.column_dimensions[get_column_letter(col)].width = 90
    for q in questions:
        if q["qid"] in answers:
            cell = ws.cell(q["row"], col, answers[q["qid"]])
            cell.alignment = WRAP_TOP


def write_run_info(wb, model: str, ran: List[str]) -> None:
    out = wb[OUTPUT_SHEET] if OUTPUT_SHEET in wb.sheetnames else wb.create_sheet(OUTPUT_SHEET)
    note_col = out.max_column + 2
    out.cell(1, note_col, "Run info").font = HEADER_FONT
    out.cell(2, note_col, f"Model: {model}")
    out.cell(3, note_col, f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    out.cell(4, note_col, f"Steps: {', '.join(ran)}")
    out.column_dimensions[get_column_letter(note_col)].width = 34


def style_sheet(ws, widths: List[int]) -> None:
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for cell in ws[1]:
        cell.font, cell.fill = HEADER_FONT, HEADER_FILL
    for row in ws.iter_rows(min_row=1):
        for cell in row:
            cell.alignment = WRAP_TOP
    ws.freeze_panes = "C2"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the pun test set through PunBot.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Test set workbook")
    parser.add_argument("--output", type=Path, help="Where to save results (default: evaluation/results/<name>_results_<time>.xlsx)")
    parser.add_argument("--only", choices=["detection", "dialog"], help="Run just one part")
    parser.add_argument("--limit", type=int, help="Only run the first N items of each part (for a quick test)")
    parser.add_argument("--column", default="J", help='Column for the Pun/Non-pun predictions (default "J", Pipeline says)')
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input workbook not found: {args.input}")
        return 1
    output = args.output or DEFAULT_OUTPUT_DIR / f"{args.input.stem}_results_{datetime.now():%Y%m%d_%H%M}.xlsx"
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        llm = LLMAdapter()
    except EnvironmentError as e:
        print(e)
        return 1

    wb = openpyxl.load_workbook(args.input)
    items = read_test_set(wb[TEST_SHEET])
    by_id = {i["id"]: i for i in items}
    questions = read_dialog_questions(wb[DIALOG_SHEET]) if DIALOG_SHEET in wb.sheetnames else []
    if args.limit:
        items, questions = items[: args.limit], questions[: args.limit]

    print(f"Model: {config.CLAUDE_MODEL}\nInput: {args.input}")
    ran, results, answers = [], {}, {}

    if args.only != "dialog":
        print(f"\nDetection: {len(items)} sentences")
        results = run_detection(llm, items)
        write_detection(wb, items, results, args.column)
        ran.append("detection")

    if args.only != "detection" and questions:
        print(f"\nDialog: {len(questions)} questions")
        answers = run_dialog(llm, questions, by_id, dict(results))
        write_dialog(wb, questions, answers)
        ran.append("dialog")

    write_run_info(wb, config.CLAUDE_MODEL, ran)
    wb.calculation.fullCalcOnLoad = True  # make Excel recompute the scoring formulas on open
    wb.save(output)
    output.with_suffix(".json").write_text(json.dumps(
        {"model": config.CLAUDE_MODEL, "detection": results, "dialog": answers}, indent=2, ensure_ascii=False))

    if results:
        scored = [r for r in results.values() if r["prediction"]]
        tp = sum(r["label"] == "Pun" and r["prediction"] == "Pun" for r in scored)
        fp = sum(r["label"] == "Non-pun" and r["prediction"] == "Pun" for r in scored)
        fn = sum(r["label"] == "Pun" and r["prediction"] == "Non-pun" for r in scored)
        tn = sum(r["label"] == "Non-pun" and r["prediction"] == "Non-pun" for r in scored)
        unparsed = len(results) - len(scored)
        print(f"\nTP={tp} FP={fp} FN={fn} TN={tn}" + (f"  unparsed={unparsed}" if unparsed else ""))
        if tp + fp and tp + fn:
            p, r = tp / (tp + fp), tp / (tp + fn)
            print(f"Precision={p:.2f} Recall={r:.2f} F1={(2 * p * r / (p + r)) if p + r else 0:.2f}")
    print(f"\nSaved: {output}\n       {output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
