# 실행 행 → (요청 ID, 요청 안 위치, 유효 여부)의 교육용 대응표
rows = [("A", 0, True), ("A", 1, True), ("B", 0, True), (None, 0, False)]
logits = [
    [3.0, 1.0, 0.0],  # 행 0: A의 중간 위치이므로 다음 토큰 선택에는 쓰지 않는다.
    [0.0, 4.0, 1.0],  # 행 1: A의 마지막 입력 위치 → 후보 번호 1
    [5.0, 1.0, 0.0],  # 행 2: B의 마지막 입력 위치 → 후보 번호 0
    [99.0, 99.0, 99.0],  # 행 3: 큰 숫자가 있어도 padding이므로 제외한다.
]
last_row = {}
for row_index, (request_id, position, valid) in enumerate(rows):
    if not valid:
        continue
    previous = last_row.get(request_id)
    # 실행 배열의 마지막 행이 아니라, 요청별 가장 큰 논리 위치를 선택한다.
    if previous is None or position > rows[previous][1]:
        last_row[request_id] = row_index
next_candidate = {
    request_id: max(range(len(logits[row])), key=lambda j: logits[row][j])
    for request_id, row in last_row.items()
}
assert last_row == {"A": 1, "B": 2}
assert next_candidate == {"A": 1, "B": 0}
print(last_row, next_candidate)
