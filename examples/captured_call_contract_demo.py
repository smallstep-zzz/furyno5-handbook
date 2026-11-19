weight, bias = 2, 1  # 원래 모델 객체가 보관한다고 가정한 스칼라 상태

def original_model(x):
    return x * weight + bias  # 사용자 계약: 인수 하나, 숫자 하나 반환

def captured_forward(weight_arg, bias_arg, x_arg):
    y = x_arg * weight_arg + bias_arg
    return (y,)  # 이 설명용 캡처 그래프의 계약: 인수 셋, 한 원소 tuple 반환

def example_backend(graph_forward):
    # 실제 backend 함수의 (GraphModule, example_inputs) API가 아닌 역할 설명용 함수다.
    return graph_forward  # 실제 backend도 받은 그래프의 호출 계약을 보존해야 한다.

backend_callable = example_backend(captured_forward)

def user_wrapper(x):
    internal_outputs = backend_callable(weight, bias, x)  # 바깥 상태를 내부 인수로 연결
    (y,) = internal_outputs  # 내부 반환 구조를 풀어 원래 사용자 출력으로 연결
    return y

assert captured_forward(2, 1, 3) == (7,)
assert user_wrapper(3) == original_model(3) == 7
print(captured_forward(2, 1, 3), user_wrapper(3))  # (7,) 7
