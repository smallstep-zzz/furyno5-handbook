from time import perf_counter_ns

state = {"counter": 0}

def reset_request():
    state["counter"] = 0  # 이 예제가 소유한 요청 상태만 초기화

def completed_call():
    state["counter"] += 1  # 반환 전에 상태 갱신이 완료되는 동기 함수
    return state["counter"]

def run_case(independent):
    reset_request()
    warmup_outputs = [completed_call() for _ in range(2)]  # 결과 1,2와 상태 2를 만듦
    samples = []
    for _ in range(3):
        if independent:
            reset_request()  # 측정 밖: 이 예는 초기화가 끝난 뒤의 실행만 잰다.
        before = state["counter"]  # 표본마다 실제 출발 상태를 기록한다.
        start = perf_counter_ns()
        output = completed_call()
        elapsed_ns = perf_counter_ns() - start  # completed_call 완료까지의 구간
        samples.append((before, output, elapsed_ns))
    return warmup_outputs, samples

for independent in (True, False):
    warmup, samples = run_case(independent)
    transitions = [(before, output) for before, output, _ in samples]
    expected = [(0, 1), (0, 1), (0, 1)] if independent else [(2, 3), (3, 4), (4, 5)]
    assert transitions == expected
    print("independent" if independent else "continuous", warmup, transitions)
# independent [1, 2] [(0, 1), (0, 1), (0, 1)]
# continuous [1, 2] [(2, 3), (3, 4), (4, 5)]
