import argparse
import json
import zipfile

def read_object(archive, name, limit):
    info = archive.getinfo(name)  # 없으면 여기에서 중단; 임의로 형식을 추측하지 않는다.
    if info.file_size > limit:
        raise ValueError(f"JSON size limit exceeded: {name}")
    value = json.loads(archive.read(name))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {name}")
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package")  # 처음 받은 .fxb/.fxbx/.fxbex 파일 경로
    path = parser.parse_args().package
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        print("members:", names[:12], "count=", len(names))
        manifest = read_object(archive, "manifest.json", 4 * 1024**2)
        print("manifest markers:", {key: manifest.get(key) for key in
                                    ("format", "schema_version", "format_version")})

        if (manifest.get("format") == "FXB_EX"
                and type(manifest.get("schema_version")) is int
                and manifest["schema_version"] == 1):
            print("reader: 초기 FXB_EX용 fxbx inspect / validate --integrity-only")
            return
        if manifest.get("format_version") != 2:
            print("reader: 이 예제에서 판별하지 않는 형식; 생산자 계약을 확인하세요.")
            return
        if "fxbex.provenance.json" not in names:
            print("native v2 표식만 확인: 커스텀 실행계획이나 공식 모델 지원은 아직 모릅니다.")
            return

        # 고정 소비자와 같은 상한: 계획 64 MiB + 그 밖의 producer 정보 4 MiB.
        producer = read_object(archive, "fxbex.provenance.json", 68 * 1024**2)
        plan = producer.get("execution_plan")
        graph = producer.get("graph_plan")
        if producer.get("producer") != "fxbex":
            print("reader: 알 수 없는 producer")
        elif isinstance(plan, dict) and plan.get("schema") in (
            "tclmachine.execution-plan.v1", "tclmachine.execution-plan.v2",
        ):
            print("plan schema:", plan["schema"])
            print("reader: fxbex.execution_fxb.read_execution_plan(path)")
        elif isinstance(graph, dict) and graph.get("schema") == "fxbex.graph-plan.v1":
            print("reader: fxbex.graph_fxb.read_graph_plan(path); 이전 단일 EDF graph 경로")
        else:
            print("reader: 지원하는 커스텀 계획 표식을 찾지 못했습니다.")

if __name__ == "__main__":
    main()
