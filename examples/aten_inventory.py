"""ATen 목록을 버전·집계 단위와 함께 기록하는 진단 스크립트.

소스 선언 수와 설치본의 등록 수를 구분한다. NPU 지원·실행 검증은 하지 않는다.
Torch 내부 진단 API를 사용하므로 설치 버전이 바뀌면 API도 다시 확인해야 한다.
"""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import re


def declared_inventory(path):
    text = Path(path).read_text(encoding="utf-8")
    # YAML에서 명시적인 함수 선언만 읽는다. 생성된 코드·실행 성공 수와 다르다.
    declarations = re.findall(r"^- func: (.+)$", text, re.MULTILINE)
    # add.Tensor(...)라면 '(' 앞의 add.Tensor를 overload 이름으로 센다.
    names = [row.split("(", 1)[0] for row in declarations]
    return {
        "scope": "explicit native_functions.yaml declarations, not runtime coverage",
        "file_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "declared_schema_count": len(declarations),
        "unique_overload_names": len(set(names)),
        # add.Tensor와 add.Scalar는 같은 기본 이름 add(packet)로 묶는다.
        "operator_packet_count": len({name.split(".", 1)[0] for name in names}),
    }


def runtime_inventory(modules):
    import torch
    for name in modules:
        # import가 operator 등록에 영향을 줄 수 있으므로 어떤 모듈을 읽었는지 기록한다.
        importlib.import_module(name)
    names = sorted(
        name for name in torch._C._dispatch_get_all_op_names()
        if name.startswith("aten::")
    )
    packets = sorted({name[6:].split(".", 1)[0] for name in names})
    all_schemas = [
        schema for schema in torch._C._jit_get_all_schemas()
        if schema.name.startswith("aten::")
    ]
    schemas = sorted({str(schema) for schema in all_schemas})
    core = {}
    tag = getattr(torch.Tag, "core", None)
    # Core는 전체 ATen과 다른 부분집합이다. 태그가 없는 설치본은 '0개'와 구별한다.
    if tag is not None:
        for packet_name in packets:
            packet = getattr(torch.ops.aten, packet_name)
            for overload_name in packet.overloads():
                overload = getattr(packet, overload_name)
                if tag in overload.tags:
                    core[str(overload)] = str(overload._schema)
    versions = {"torch": torch.__version__}
    for package in ("furiosa-torch", "furiosa-tcc", "fxbx", "tplmachine"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    rows = [{"operator": name, "schema": core[name]} for name in sorted(core)]
    report = {
        "scope": "installed registry, not NPU support or numerical acceptance",
        "versions": versions,
        "imported_modules": modules,
        "privateuse1_backend_name": torch._C._get_privateuse1_backend_name(),
        "registered_aten_overload_count": len(names),
        "registered_aten_packet_count": len(packets),
        "unique_aten_schema_count": len(schemas),
        "core_tag_available": tag is not None,
        "core_overload_count": len(core) if tag is not None else None,
        "core_packet_count": len({name.rsplit(".", 1)[0] for name in core}) if tag is not None else None,
        "registered_names": names,
        "schemas": schemas,
        "core_overloads": rows,
    }
    # 같은 버전·등록 목록인지 비교할 수 있도록 정렬한 보고서의 해시를 남긴다.
    encoded = json.dumps(report, sort_keys=True, ensure_ascii=False).encode()
    report["inventory_sha256"] = hashlib.sha256(encoded).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--import-module", action="append", default=[])
    parser.add_argument("--native-yaml", type=Path)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    if args.source_only and args.native_yaml is None:
        parser.error("--source-only requires --native-yaml")
    report = {"schema": "furyno5-handbook.aten-inventory.v1"}
    if not args.source_only:
        report["runtime"] = runtime_inventory(args.import_module)
    if args.native_yaml:
        report["source"] = declared_inventory(args.native_yaml)
    # x 모드로 기존 보고서를 덮어쓰지 않고 새 버전의 결과 파일만 생성한다.
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    summary = report.get("runtime", report.get("source"))
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("registered_names", "schemas", "core_overloads")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

