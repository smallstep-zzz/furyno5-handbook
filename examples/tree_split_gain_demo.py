x = [0.1, 0.4, 0.6, 0.9]
targets = [1.0, 1.0, 3.0, 3.0]
gradients = [-y for y in targets]  # 현재 예측 0, 제곱 손실이므로 g=0-y
hessians = [1.0] * len(x)

def leaf_value(indices):
    g = sum(gradients[i] for i in indices)
    h = sum(hessians[i] for i in indices)
    return -g / h  # 이 예의 손실을 최소화하는 같은 leaf의 보정값

def loss(predictions):
    return sum((p - y) ** 2 / 2 for p, y in zip(predictions, targets))

parent = [leaf_value(range(len(x)))] * len(x)
parent_loss = loss(parent)  # 분할 없이 한 leaf로 보정한 손실 2
results = []
for threshold in (0.2, 0.5):
    left = [i for i, value in enumerate(x) if value <= threshold]
    right = [i for i, value in enumerate(x) if value > threshold]
    # 두 후보 모두 빈 leaf가 없다는 예제 조건 아래 계산한다.
    lv, rv = leaf_value(left), leaf_value(right)
    predictions = [lv if value <= threshold else rv for value in x]
    candidate_loss = loss(predictions)
    results.append((threshold, candidate_loss))
    print(threshold, round(candidate_loss, 6), round(parent_loss - candidate_loss, 6))
assert min(results, key=lambda item: item[1])[0] == 0.5
# 0.2 1.333333 0.666667
# 0.5 0.0 2.0
