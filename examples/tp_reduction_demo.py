def relu(value):
    return max(0, value)  # 음수만 0으로 바꾸는 원소별 함수

partial_0, partial_1 = -2, 3  # 서로 다른 출력이 아니라 같은 원소에 더할 두 부분
correct = relu(partial_0 + partial_1)  # 먼저 완성한 출력 1에 ReLU 적용
wrong = relu(partial_0) + relu(partial_1)  # 음수 부분을 일찍 버려 결과가 3이 됨
assert (correct, wrong) == (1, 3)
print(correct, wrong)

# 반대로 두 값이 이미 완성된 서로 다른 출력 원소라면 원소별 처리 뒤 연결 가능
separate_outputs = [-2, 3]
processed_slices = [relu(separate_outputs[0]), relu(separate_outputs[1])]
assert processed_slices == [0, 3]  # 전체 벡터 [-2,3]에 원소별 ReLU한 것과 같다.
print(processed_slices)
