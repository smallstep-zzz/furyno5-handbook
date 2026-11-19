"""Versioned ATen inventory. Diagnostic internal APIs; no device execution."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import re


def declared_inventory(path):
    text = Path(path).read_text(encoding="utf-8")
    declarations = re.findall(r"^- func: (.+)$", text, re.MULTILINE)
    names = [row.split("(", 1)[0] for row in declarations]
    return {
        "scope": "explicit native_functions.yaml declarations, not runtime coverage",
        "file_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "declared_schema_count": len(declarations),
        "unique_overload_names": len(set(names)),
        "operator_packet_count": len({name.split(".", 1)[0] for name in names}),
    }


def runtime_inventory(modules):
    import torch
    for name in modules:
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
    # Refuse to silently replace a previous versioned inventory.
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    summary = report.get("runtime", report.get("source"))
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("registered_names", "schemas", "core_overloads")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

