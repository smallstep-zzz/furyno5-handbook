def completion_us(duration_a, duration_b):
    start_a, start_b = 10, 20  # 시작 시각은 세 비교에서 같다고 가정
    copy_us = 10  # 두 결과가 준비된 뒤에만 시작하는 마지막 복사 시간
    end_a = start_a + duration_a
    end_b = start_b + duration_b
    return max(end_a, end_b) + copy_us

original = completion_us(40, 40)
faster_a = completion_us(20, 40)
faster_b = completion_us(40, 20)
assert (original, faster_a, faster_b) == (70, 70, 60)
print(original, faster_a, faster_b)  # 70 70 60
