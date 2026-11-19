import argparse
import json
from pathlib import Path

def read_optional(path):
    if not path.is_file():
        return None  # 파일 없음은 성공 판정이 아니다. 더 이른 실패일 수 있다.
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work", type=Path)  # 실패 메시지 또는 출력 이름에서 찾은 .plan 경로
    work = parser.parse_args().work
    summary = read_optional(work / "preflight.json")
    groups = read_optional(work / "failure-groups.json")
    inventory = read_optional(work / "operator-inventory.json")

    if summary is None:
        print("preflight.json 없음: 원래 예외와 캡처·환경 단계를 먼저 확인하세요.")
    else:
        print("STATUS", summary.get("status"), "CODE", summary.get("code"))
        for row in summary.get("first_failures", []):
            print("FIRST", row["node"], row["op"], row["category"], row["reason"])

    if groups is None:
        print("failure-groups.json 없음: 이 파일만으로 규칙 통과 여부를 판단하지 않습니다.")
    else:
        # 원인 후보를 먼저 읽고, 입력을 받지 못해 따라 실패한 항목은 별도로 읽는다.
        for category in ("rule_or_contract_gap", "binding_dependency"):
            for row in groups:
                if row.get("category") == category:
                    print("GROUP", category, row["op"], row["occurrences"], row["reason"])

    if inventory is not None:
        for row in inventory.get("nodes", []):
            if row.get("admission") in ("rule_or_contract_gap", "binding_dependency"):
                print("NODE", row["node"], row["operator"], row["admission"])
                print("  inputs=", row.get("inputs"), "reason=", row.get("admission_reason"))

if __name__ == "__main__":
    main()
