# 10. 작은 모델 하나로 읽는 FuryNo5의 구조

> **전체 경로에서의 위치**
>
> 작은 PyTorch 모델 하나가 FuryNo5 안에서 어떤 객체와 파일로 바뀌는지 따라간다. 자세한 컴파일은 [11장](11-furyno5-compiler.md), 파일을 받아 실행하는 쪽은 [12장](12-fxbx.md)에서 다룬다.

Python의 모델은 계산 방법과 그 계산에 쓸 가중치를 알고 있다. NPU에서 실행하려면 이 계산을 장치 프로그램으로 바꾸고 실제 입력을 넣을 수 있게 준비해야 한다. **FuryNo5는 이 준비를 맡는다.**

이 책에서는 파일을 만드는 쪽을 생산자(producer), 만들어진 파일을 읽어 실행하는 쪽을 소비자(consumer)라고 부른다. 최종 NPU 코드 생성은 Furiosa 컴파일러가 맡고, 소비자는 선택한 경로의 SDK 실행기나 fxbX가 된다.

먼저 작은 모델이 어떤 자료구조와 파일로 바뀌는지 따라가자. 그다음 실제 파일·함수와 지원 범위를 읽으면, “후보를 만들었다”, “컴파일했다”, “NPU에서 정답을 확인했다”가 서로 다른 결과라는 점도 자연스럽게 드러난다. FuryNo5는 모든 ATen(A Tensor Library) 연산을 완성한 범용 backend가 아니며 README도 전체 모델 자동 이식 완료를 주장하지 않는다. [프로젝트의 현재 범위](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/README.md)

> **코드와 검증 기준** · FuryNo5 commit `bba9efea8201aec48150514a35d0b637d29d107e`를 2026-11-19에 확인했다. 소스 링크는 비공개 저장소 열람 권한이 필요할 수 있다. 지원 여부는 아래의 등록·컴파일·수치·장치 검증 범위를 나누어 읽는다.

## 공통 예제: 8개 특징을 받아 8개 값을 돌려주는 모델

먼저 숫자 두 개를 받아 새 숫자 세 개를 만드는 층을 생각하자. 입력이 `[1,2]`일 때 첫 출력은 첫 값만 골라 1, 둘째는 둘째 값만 골라 2, 셋째는 둘을 더하고 1을 더해 4로 만들 수 있다. 각 출력에 곱할 수와 더할 수를 따로 둔다. 이것이 **Linear 층**의 작은 예다.

이 장의 실제 모델은 그 크기를 늘린 `Linear(8,12) → ReLU → Linear(12,8)`이다. 한 입력의 숫자 8개를 12개로 바꾸고, 음수를 0으로 바꾸는 ReLU를 적용한 뒤 다시 8개로 바꾼다. 입력을 네 개 모으므로 `x`의 shape는 `[4,8]`이다. 숫자 저장 형식은 BF16을 사용한다.

Linear의 출력 하나는 입력의 각 수에 계수를 곱해 더하고 보정값을 더한 것이다. 곱할 계수를 모은 것이 **weight**, 마지막에 더할 값이 **bias**다.

`Linear(8,12)`는 입력 여덟 개로 출력 하나를 만드는 계수 줄을 열두 개 가진다. 그래서 weight의 shape는 `[12,8]`이다. 코드에서는 입력 `[4,8]`과 곱하기 위해 weight의 행·열을 바꾼 `W.T[8,12]`를 사용하고, 결과는 `[4,12]`가 된다.

보정값 열두 개는 네 입력 행에 각각 반복해서 더한다. 이 반복 적용이 broadcasting이다. 행·열 교환은 transpose이며 원소 수만 유지해 모양을 다시 읽는 reshape로 대체할 수 없다. [Linear의 규격](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Linear.html)

| 논리 값 | shape | 계산 또는 공급자 | 파일을 다시 읽을 때 필요한가 |
|---|---|---|---|
| `x` | `[4,8]` | 호출자가 매번 전달 | 예, 요청 입력 |
| 첫 weight·bias | `[12,8]`, `[12]` | `model.state_dict()`의 `0.weight`, `0.bias` | 예, 외부 가중치 |
| `h` | `[4,12]` | 첫 Linear 결과 | 실행 중 생성 |
| `a` | `[4,12]` | `relu(h)` | 실행 중 생성 |
| 마지막 weight·bias | `[8,12]`, `[8]` | `2.weight`, `2.bias` | 예, 외부 가중치 |
| `y` | `[4,8]` | 마지막 Linear 결과 | 호출자에게 반환 |

모델이 가지고 있는 weight·bias 같은 조정 가능한 배열을 **parameter(매개변수)라고** 한다. 이 실습은 학습이 아니라 이미 정한 값으로 답을 구하므로 값을 갱신하는 데 필요한 계산 기록을 요구하지 않는다. `requires_grad_(False)`가 이 설정에 해당한다.

`eval()`은 학습할 때와 추론할 때 동작이 다른 층을 평가 모드로 바꾼다. 미분 기록을 끄는 설정과 목적이 다르다. `.bfloat16()`은 숫자의 저장 형식을 바꾸므로, 실제 모델에서는 이 변경 뒤 답의 오차도 확인해야 한다.

같은 shape의 tensor라도 메모리 해석이 다를 수 있다. **stride**는 한 축에서 원소 하나 이동할 때 저장소에서 건너뛰는 원소 수다. 행이 연속으로 저장된 `[4,8]` 입력의 stride는 `[8,1]`이다. **contiguous**는 해당 순서로 빈틈없이 이어진 배치를 뜻한다. 전치된 view처럼 메모리를 공유하되 stride가 다른 입력을 같은 커널에 무조건 넣을 수는 없다. FuryNo5 규칙은 이런 조건도 함께 검사한다. [입력 계약 검사](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/edf_export.py#L33-L109)

## 이 모델이 어떤 객체와 파일로 바뀌는가

모델의 연결을 종이에 적는다고 생각하자. ‘입력을 첫 Linear에 넣고, 그 결과를 ReLU에 넣고, 다음 Linear의 답을 반환한다’는 관계를 프로그램 안의 자료구조로 얻는 일이 **캡처(capture)다**.

이 구조에서 계산·입력 하나를 표시한 점이 node, 앞 결과가 다음 계산으로 간다는 선이 연결이다. 함께 컴파일할 계산 묶음은 region(영역)이라고 한다. 아래 표는 이 연결에서 파일까지 무엇을 만드는지 보여 준다. 연산 세 개가 반드시 EDF 세 개가 된다는 뜻은 아니다.

| 단계 | 입력 → 결과 | 이번 모델에서 의미하는 것 |
|---|---|---|
| 캡처 | Python module·예제 입력 → `ExportedProgram` | Linear·ReLU의 연결과 입력 조건을 추출 |
| 규칙·영역 선택 | graph·tensor 정보 → 컴파일 요청 | 지원하는 shape·dtype인지, 어디까지 묶을지 결정 |
| 계산 표현 생성 | 요청 → TCL 문자열 또는 SDK compiler 입력 | `h`, `a`, `y`를 계산하는 의미를 표현 |
| 컴파일 | 계산 표현 → EDF bytes | RNGD에서 실행할 장치 프로그램 생성 |
| 계획 조립 | EDF 계약·값 연결 → 실행계획 JSON | 어떤 weight·입력·중간값으로 각 프로그램을 호출할지 기록 |
| 포장 | 실행계획·EDF → `.fxbx` 패키지 | 다른 프로세스에 전달할 파일 생성 |
| 소비 | 패키지·외부 weight·`x` → `y` | fxbX가 검사·적재·실행 |

TCL(Tensor Contraction Language)은 텐서 계산을 compiler에 설명하는 언어이고, EDF(Executable Device Format)는 컴파일된 장치 프로그램이다. JSON(JavaScript Object Notation) 실행계획에는 코드 대신 값의 이름과 연결이 들어간다. **export**는 여기서 메모리의 모델을 다른 단계가 사용할 객체·파일로 내보내는 작업을 뜻한다. `torch.export`가 만드는 객체와 FuryNo5 `export_model`이 만드는 패키지는 서로 다른 단계의 export다.

## 이름과 import 경로

제품명은 **FuryNo5**다. 과거 TCLMachine과의 호환 때문에 배포 패키지와 내부 폴더에 `tplmachine`, 외부에서 부르는 얇은 함수 계층인 공개 facade에 `tclmachine`이 남아 있다. 이것은 문서에서 다른 프로젝트를 의미하는 것이 아니다. 코드 예제를 실제로 실행하려면 현재 존재하는 import 경로를 그대로 써야 한다. [공개 facade](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tclmachine/__init__.py)

공개 진입점의 중요한 차이는 다음과 같다.

| API | 사용하는 때 | 현재 성격 |
|---|---|---|
| `tclmachine.optimize` | 작은 정적 BF16 수축 그래프를 현재 프로세스에서 준비 | 사용자가 선택한 graph에 새 연산을 연결해 호출 |
| `tclmachine.export_model` | 모델 그래프를 파일로 생산 | 다중 EDF·tensor 연결·상태 계약 |
| `tclmachine.export_program` | `forward` 등 이름 붙은 실행 입구(entry) 여러 개를 묶음 | 공유 불변 가중치와 명시적 상태 |
| `tclmachine.build_verified_model` | 파일 생성 뒤 독립 소비자 검증까지 요구 | 변경 입력·CPU 기준·공식 trace·새 프로세스 replay |
| `tclmachine.load_plan` | 측정 선택 경로의 결정·준비 기록·환경을 연결 | `SelectedRuntime` 호출 객체; `.fxbx` reader가 아님 |
| `tclmachine.plan_regions` | 분석용 graph dictionary와 비용표로 영역 조합 탐색 | 후보 제안 dictionary; 실행 가능한 패키지가 아님 |
| `tclmachine.prepare_model_direct` | 공개 SDK 컴파일 경로를 직접 활용 | 별도 직접/분할 실행 경로 |

사용자가 결정할 첫 질문은 **지금 프로세스 안에서 호출할 객체가 필요한가, 다른 프로세스로 옮길 파일이 필요한가**다. 전자는 `optimize`, 후자는 `export_model`을 살펴본다. 각각 다른 구현 경로이므로 `export_model`이 더 많은 의미 계열을 처리한다고 해서 `optimize`의 등록 범위가 자동으로 늘지는 않는다. 어떤 경로를 고쳤는지를 PR(Pull Request)과 문서에 반드시 적는다.

여기서 `plan`이라는 이름이 두 계통에 쓰이므로 파일을 먼저 구분한다. 05장과 11장의 실행계획은 **tensor를 어느 EDF에 전달할지** 기록한다. 반면 `load_plan`의 결정 기록은 **이미 측정한 후보 중 무엇을 선택했는지** 기록한다. 이름이 비슷해도 함수의 인수와 실행 파일 형식이 다르다.

| 생산자와 소속 경로 | 다음 API에 넘기는 실제 자료 | 얻는 것과 사용 범위 |
|---|---|---|
| `plan_regions(graph, cost_book)` — 오프라인 분석 | 분석용 graph dictionary, 비용·후보 정보를 담은 dictionary | 비용 기반 제안. 바로 `load_plan`에 주는 확정 결정이 아니다. |
| 측정 선택 모듈 `select_measured_plan` — 측정 후 선택 | `load_plan(decision, prepared, environment)`의 세 dictionary 또는 각각의 JSON 경로. `decision.schema`는 `tplmachine.measured-plan-selection.v1`이어야 한다. | `SelectedRuntime`. 준비 기록이 지정한 serialized Runnable과 인수 계약을 사용한다. |
| `export_model` — 다중 EDF 생산 | `tiny.fxbx`; 생산 중 따로 남는 `tiny.fxbx.plan/execution-plan.json`은 같은 계획의 조사 자료 | `read_execution_plan("tiny.fxbx")`는 계획 dictionary, `GraphFXB("tiny.fxbx", weights=...)`는 실행 소비자 객체를 만든다. |

예를 들어 앞의 MLP를 `export_model`로 만들었다면 12장의 `GraphFXB` 경로를 고른다. 그 패키지나 옆의 `execution-plan.json`을 `load_plan`의 첫 인수로 바꾸어 넣지 않는다. 반대로 측정 선택 경로는 준비 기록의 `runtime_input_specs`와 artifact 해시를 확인하며, 필요하면 이미 포장된 가중치까지 그 인수 계약대로 제공해야 한다. 이런 변환을 `load_plan`이 원본 모델에서 자동 추론하지는 않는다. [공개 함수의 세 입력·schema 검사](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tclmachine/__init__.py#L33-L51) · [측정 선택 실행기의 입력·loader](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/research/selected_runtime.py#L79-L145) · [분석용 영역 계획기](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/region_planner.py#L212-L227)

## 첫 단계: 후보 생성은 실행이 아니다

**규칙(rule)은** “이 연산의 이 입력 조건을 어떤 계산으로 표현할 수 있는가”를 판정하는 코드다. 규칙이 통과하면 compiler에 시도할 후보를 만들 수 있지만, 아직 compiler 성공이나 장치 정답은 모른다. **수축(contraction)은** 행렬곱처럼 원소를 곱한 뒤 공통 축을 따라 더하는 계산이다. `aten.py`의 현재 `RULES`에는 `mm`, `bmm`, `matmul`, `linear`의 제한된 수축 후보가 들어 있다. 동일 파일은 BF16, 정적 양의 shape, 지원 rank(축의 개수), contiguous stride, 정확한 축 크기를 검사한다. 규칙이 맞으면 `AtenCandidate`에 의미·입출력·TCL source를 담는다. 파일 상단도 이것이 하드웨어 지원 인증표가 아니라고 명시한다. [실제 규칙](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/aten.py#L40-L66)

```python
# 후보 분석 경로다. EDF 컴파일이나 NPU 추론을 요청하지 않는다.
import torch
from tplmachine.aten import plan_exported

class Product(torch.nn.Module):
    def forward(self, a, b):
        return torch.mm(a, b)  # [2,3] @ [3,2] → [2,2]; 공통 축 3에서 곱을 합한다.

a = torch.tensor([[1, -2, 3], [-4, 5, -6]], dtype=torch.bfloat16)  # 왼쪽 [2,3]
b = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.bfloat16)    # 오른쪽 [3,2]
ep = torch.export.export(Product().eval(), (a, b), strict=False)  # 위치 인수 2개로 캡처
report = plan_exported(ep)  # 그래프의 연산·입력 조건을 규칙과 대조한 보고서
print(report["candidate_count"])       # 컴파일을 시도할 수 있는 후보 수
print(report["npu_execution"])          # not_run: 실제 장치 출력은 아직 없음
print(report["full_graph_executable"])  # False: 전체 그래프 실행 승인이 아님
```

이 코드는 전체 MLP(Multilayer Perceptron)를 준비하기 전에 행렬곱 하나만 조사한다. `a`는 `[2,3]`, `b`는 `[3,2]`이고 식의 예상 출력은 `[[4,1],[-10,-1]]`이다. 첫 원소는 `1*1 + (-2)*0 + 3*1 = 4`, 둘째 원소는 `1*0 + (-2)*1 + 3*1 = 1`로 구한다. 하지만 코드에서는 `a @ b`의 NPU 결과를 구하지 않는다. `torch.export.export`가 만든 객체를 `plan_exported`에 주어 후보 수와 분석 상태를 읽는다. `npu_execution=not_run`과 `full_graph_executable=False`는 바로 그 경계를 나타낸다.

여기에서 실패했다고 다중 EDF producer 전체가 같은 연산을 처리하지 못한다고 단정하지 않는다. `execution_plan.py`는 원소별·축약·layout·공간 연산 등 다른 의미 계열 생성기도 사용한다. 반대로 후보 source가 생겨도 컴파일 성공, ABI(Application Binary Interface: 실행 코드와 호출자가 합의한 인수 순서·자료형·배치) 일치, 수치 통과는 아직 남아 있다.

## 두 번째 단계: 준비한 그래프만 dispatch한다

이제 후보를 실제로 컴파일해 현재 프로세스에서 부를 함수로 만들려면 준비 단계가 필요하다. **준비된 callable**은 shape·dtype 계약과 컴파일 결과를 이미 가진 호출 객체다. **dispatch**는 연산 호출을 그 구현 함수에 연결하는 일이다. `register_prepared`의 핵심은 **기존 `aten` 전역 구현을 덮어쓰지 않는 것**이다. 새로운 `tclmachine` namespace에 이름이 고유한 연산을 만들고, 준비된 실행 함수를 외부 장치용 선택 표식인 `PrivateUse1`에 연결한다. `Meta`에는 실제 계산 없이 출력 shape·dtype를 알려주는 함수를 연결한다.

```python
# aten_runtime.py의 핵심: 설명 주석을 추가한 발췌, 주변 함수·변수 정의는 생략.
library = torch.library.Library('tclmachine', 'FRAGMENT')  # 새 연산을 붙일 namespace
library.define(f'{name}({schema_args}) -> Tensor')  # 고유 이름과 입력·출력 함수 선언
library.impl(name, implementation, 'PrivateUse1')  # 실제 NPU 실행 callable 연결
library.impl(name, meta, 'Meta')  # 계산 없이 출력 shape·dtype만 설명하는 함수 연결
```

위 네 줄은 내부 함수의 발췌이며 독립 실행 예제가 아니다. `name`은 준비 과정에서 만든 고유 연산명, `schema_args`는 입력 tensor들의 함수 선언, `implementation`은 준비된 장치 프로그램을 호출하는 함수, `meta`는 출력 정보만 만드는 함수다. `Library(..., 'FRAGMENT')`는 같은 namespace에 구현 조각을 추가할 때 사용하는 등록 객체다.

준비가 끝나면 `optimize`는 모델 그래프의 복사본에서 호출할 함수를 바꾼다. 예를 들어 원래 `aten.mm(a,b)`를 호출하던 점이, 이 입력 조건으로 미리 준비한 `tclmachine` 연산을 호출하도록 바뀐다. 여기서 **target**은 그 점이 호출하는 함수다.

프로그램 안의 모든 `torch.mm`를 전역으로 바꾸는 것은 아니다. 선택한 그래프와 지원 입력 조건의 호출을 준비하는 경로다. 실제 실행 때는 저장된 실행 객체에 현재 입력을 전달한다. [등록과 rewrite 구현](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/aten_runtime.py#L104-L119) · [준비 과정](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/aten_runtime.py#L210-L276)

이 방식은 “일반적인 Torch의 모든 함수를 호출하기만 하면 FuryNo5가 자동 개입한다”와 다르다. 사용자가 지원 범위의 그래프에 `optimize`를 명시적으로 적용해야 한다.

### Product를 준비하고 한 번 실행하기

앞의 `Product`를 실제 callable로 만드는 완성 예제는 [run_prepared_product.py](../examples/run_prepared_product.py)다. 호환 SDK, FuryNo5와 NPU가 필요하다. **이 shape의 실제 지원 여부는 컴파일과 장치 수치 비교로 확인해야 한다.** 사전 규칙을 통과하는 입력이라는 사실과 설치된 compiler의 성공은 별도다.

```python
import torch
import furiosa.torch as ft  # furiosa 장치 backend와 완료 대기 API를 제공한다.
from tclmachine import optimize

class Product(torch.nn.Module):
    def forward(self, a, b):
        return torch.mm(a, b)  # [2,3] @ [3,2] → [2,2]

def main():
    model = Product().eval().requires_grad_(False)
    a = torch.tensor([[1, -2, 3], [-4, 5, -6]], dtype=torch.bfloat16)
    b = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.bfloat16)
    with torch.no_grad():
        expected = model(a, b)  # CPU 기준값: [[4,1],[-10,-1]], shape [2,2]

    prepared = optimize(
        model,
        example_inputs=(a, b),  # CPU 대표 입력으로 shape·dtype·stride와 graph를 준비한다.
        target="furiosa:0",     # 실제 호출 입력도 이 장치에 있어야 한다.
        fallback="error",       # 이 facade가 허용하는 정책; CPU 모델 재실행 옵션이 아니다.
    )
    try:
        a_device = a.to("furiosa:0")  # 이번 호출의 실제 원소 값을 장치로 복사한다.
        b_device = b.to("furiosa:0")
        with torch.no_grad():
            output_device = prepared(a_device, b_device)  # Product의 반환형대로 tensor 하나
            actual = output_device.cpu()                # 비교할 CPU tensor [2,2]로 받는다.
        ft.current_stream(torch.device("furiosa:0")).synchronize()  # 완료를 확인한 뒤 해제
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        print(actual)  # 위 비교를 실제 통과한 결과만 출력한다.
    finally:
        prepared.close()  # 소유한 custom-op 등록을 해제한다. 이후 prepared 호출은 오류다.

if __name__ == "__main__":
    main()
```

| 코드의 값·호출 | 장치와 역할 | 이 시점의 상태 |
|---|---|---|
| `(a, b)` | CPU BF16 대표 입력, contiguous `[2,3]`·`[3,2]` | 준비에 사용할 계약과 원본 CPU 기준값 |
| `optimize(...)` 반환 | `PreparedGraph` Python 객체 | 지원 그래프의 커널 컴파일·등록 완료. 사용자 입력의 NPU 추론 결과는 아직 없음 |
| `prepared(a_device, b_device)` | `furiosa:0`의 같은 shape·dtype·stride 인수 | 이미 준비한 커널을 호출하고 NPU tensor `[2,2]` 반환 |
| `.cpu()`와 `synchronize()` | 결과를 CPU로 받고 장치 완료 확인 | CPU 기준과 숫자를 비교할 수 있음 |
| `prepared.close()` | Python 등록과 실행 객체의 수명 종료 | 같은 객체로 다시 호출할 수 없음 |

여기서는 `prepared`가 `Product.forward`의 반환 구조를 유지하므로 tensor 하나를 받는다. 12장의 `GraphFXB.predict()`는 별도 패키지 API로 항상 출력 tuple을 돌려주는 경로이므로 `actual, = ...`를 사용한다. `example_inputs`를 넘겼다고 같은 CPU tensor로 `prepared(a, b)`를 실행할 수 있는 것은 아니다. 준비된 PrivateUse1 구현은 지정 장치의 인수를 검사하며, 공개 `optimize`는 CPU reference를 등록하지 않는다. 또한 `prepared.report`는 준비 영수증이므로 호출 후 자동으로 NPU 검증 보고서로 변하지 않는다. [facade 인수](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tclmachine/__init__.py#L19-L24) · [입력·장치 검사](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/aten_runtime.py#L76-L100) · [수명과 반환 보고서](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/aten_runtime.py#L181-L283)

## 세 번째 단계: 전체 파일을 생산하는 경로

이번에는 다시 `[4,8]` 입력을 가진 전체 MLP로 돌아온다. `model`은 계산 방법과 weight를 가진 Python 객체, `(x,)`는 입력이 한 개임을 나타내는 tuple, `"tiny.fxbx"`는 출력 파일 경로다. 출력 패키지에 외부 weight의 연결 계약을 담는 것과 weight 값 자체를 모두 넣는 것은 다르므로 실행 때는 해당 weight를 별도로 제공한다.

`export_model`은 모델을 캡처하고, 입력·상태를 연결하고, 의미 규칙으로 영역을 정하고, 고유 커널만 컴파일한다. 성공하면 `execution-plan.json`과 패키지를 만들지만 반환 상태는 **`built_not_runtime_validated`다**. 이 이름 그대로 받아들이는 것이 중요하다. [결과 생성부](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/execution_plan.py#L1287-L1300)

```python
# 호환 SDK와 FuryNo5/fxbX 설치본에서 실행할 생산 예제.
import torch
from tclmachine import export_model

def main():
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 12),  # x[4,8] → h[4,12]; W[12,8]을 전치해 곱한다.
        torch.nn.ReLU(),        # 음수만 0으로 바꾸고 shape [4,12]는 유지한다.
        torch.nn.Linear(12, 8), # [4,12] → y[4,8]
    ).bfloat16().eval().requires_grad_(False)  # BF16 가중치, 추론 모드, 학습 제외
    x = torch.randn(4, 8, dtype=torch.bfloat16)  # 가중치와 dtype을 맞춘 CPU 대표 입력

    result = export_model(
        model, (x,), "tiny.fxbx",  # 모델·입력 하나의 tuple·새 출력 경로
        compile_workers=2,  # CPU 컴파일 worker 수; 장치 개수가 아니다.
    )
    print(result["status"])  # 생성 상태이며 실제 추론 결과가 아니다.
    print(result["regions"], result["unique_kernels"])  # 영역 수와 고유 컴파일 결과 수
    print(result["timings_s"])  # 생산 단계별 소요 시간(초); warm 추론 지연이 아니다.

if __name__ == "__main__":  # spawn한 worker에서 전체 작업이 재실행되는 것을 막는다.
    main()
```

이 예제는 `.py` 파일로 저장해 실행한다. 병렬 컴파일은 `spawn`으로 자식 프로세스를 만들 수 있으므로 `__main__` 보호가 필요하다. 성공한 호출의 `result`는 출력 tensor가 아니라 생산 보고서다. `regions`는 계획의 영역 수, `unique_kernels`는 서로 다른 컴파일 결과 수, `timings_s`는 생산 단계별 시간이다. 같은 kernel을 여러 번 사용할 수 있으므로 이 수들을 같다고 가정하지 않는다. `tiny.fxbx.plan`은 준비와 진단 산출물을 보관하는 디렉터리이며 `tiny.fxbx`는 전달용 패키지다. 이미 출력 파일이나 `.plan` 디렉터리가 있는 경로를 덮어쓰지 않는다. 수정 후 비교 실험은 새로운 출력 경로를 쓴다. 생성된 파일을 검증하는 완전한 과정은 포팅 장에서 다룬다. [컴파일 작업 실행](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/execution_plan.py)

## “모든 ATen 지원”을 어떻게 판단할 것인가

**지원 범위**는 연산 이름 목록만으로 나타낼 수 없다. overload는 같은 기본 이름 아래 인수 형태가 다른 함수이고, 같은 overload도 shape·dtype·stride·속성에 따라 다른 구현이 필요하다. 아래 **Core**는 PyTorch가 작은 공통 집합으로 구분한 연산들이며 전체 ATen과 같지 않다.

소스에 남은 **2026-11-10 Core 기준선**은 다음과 같다. 이 표는 현재 모든 코드의 최종 지원율이 아니라, 당시 동일한 분모와 증거 규칙으로 만든 상태표다. 후속 수정으로 성공한 사례를 이 표에 임의로 더하지 않는다.

| Core overload 상태 | 개수 | 올바른 해석 |
|---|---:|---|
| 작은 그래프 NPU·수치 증거 있음 | 13 | 검증된 특정 입력 조건이 있음 |
| metadata 승인만 있음 | 7 | compiler·NPU·정확성 성공은 미확인 |
| 직접 규칙·계약 차단 관측 | 20 | 현재 producer의 조건/규칙에서 실패 |
| 미확인 | 145 | 관측·검증하지 않은 계약 |
| 역전파 전용 제외 | 7 | 프로젝트의 추론 범위 밖 |
| 합계 | 192 | 고정 Core overload 목록 |

13개 중 6개는 다른 입력 조건에서는 실패도 관측되었다. 이 사실은 모순이 아니다. 연산 이름 하나가 아니라 **overload × shape × dtype × stride × 속성 × 상태 × SDK**가 지원 계약이기 때문이다. 예를 들어 contiguous BF16 행렬곱 성공은 비연속 FP32(Floating Point 32-bit) broadcast 행렬곱 성공을 보장하지 않는다. [기계식 coverage.json](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/docs/validation/core-aten-20261110/coverage.json)

최신 지원 상태를 만들 때는 `core_aten snapshot/report`로 분모와 corpus를 고정하고, trace·수치·반복 실행이 있는 증거만 승계한다. 단순 `metadata_admitted`를 NPU 지원으로 승격하지 않는 코드가 이미 있다. [지원표 생성기](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/research/core_aten.py)

## 프로젝트를 세 부분으로 읽자

| 부분 | 입력 | 결과 | 보장하지 않는 것 |
|---|---|---|---|
| 분석·탐색 코어 | 연산 DAG(Directed Acyclic Graph), 자원·메모리 모델, 비용 근거 | 합법성 검사, 후보 순위, 예측 보고서 | 실제 RNGD instruction 실행·전체 물리 비용 정확도 |
| 모델 생산 경로 | frozen PyTorch 모델, 예제 입력, 선택적 상태 계약 | TCL, genuine EDF, 명시적 실행 계획, 패키지 | 생성만으로 전체 모델 수치·성능 승인 |
| 명시적 실행 연결 | 준비된 callable 또는 검증된 계획·산출물 | SDK 실행, fxbX 소비 | forward 중 무제한 탐색·자동 CPU 모델 폴백 |

이제 파일을 만드는 흐름을 보았으므로, 같은 저장소 안의 분석 기능과 실행 기능을 구분할 수 있다. 표의 DAG(Directed Acyclic Graph)는 순환 없는 의존 그래프이며, 비용 예측은 실제 장치 실행을 대신하는 모델이다. 분석 코어에서 합성 자원 충돌을 줄여 예측 시간이 절반이 되더라도 실기기 속도가 두 배가 된 것은 아니다. 반대로 실행 경로가 genuine EDF를 만들었다고 시뮬레이터의 비용 예측까지 검증된 것은 아니다. `hardware_measured`는 장치에서 측정한 값인지, `source_kind`는 예측·기록 중 어디에서 얻었는지를 알려준다. 환경 지문은 SDK·compiler 등 결과의 조건을 식별하는 정보다. 이를 함께 읽어야 예측치를 실측으로 오해하지 않는다. [설계](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/docs/ARCHITECTURE.md) · [IR(Intermediate Representation) 정의](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/ir.py)

```mermaid
flowchart TB
    A["모델 + 예제 입력 + 상태 계약"] --> B["capture와 의미 정규화"]
    B --> C["shape · dtype · layout 규칙"]
    C --> D["영역 선택과 TCL 생성"]
    D --> E["공식 compiler와 EDF ABI 확인"]
    E --> F["tensor 연결 + 실행 계획"]
    F --> G["fxbX: 검증 · 적재 · 실행"]
    C --> H["지원 gap과 재현 recipe"]
    D --> I["비용 예측과 후보 탐색"]
    I --> J["별도 수치 · trace · 성능 검증"]
    J --> D
```

그림에서 fxbX의 역할은 준비된 계약을 소비하는 것이다. FuryNo5가 누락된 연산 규칙을 정의하고 컴파일하므로, fxbX 실행 중 다시 생산자를 불러 미지원 연산을 고치는 구조로 설명하면 안 된다. [Native 실행 책임](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/docs/NATIVE_EXECUTION_PLAN.md)

## 개발자가 따라갈 파일 지도

아래 표는 앞의 값·객체·파일 변환을 구현한 위치다. 처음부터 모두 열 필요는 없다. 예를 들어 입력 계약이 막히면 `edf_export.py`, 준비된 호출이 잘못 연결되면 `aten_runtime.py`를 확인한다. ABI(Application Binary Interface)는 컴파일된 프로그램과 호출자가 합의한 인수 순서·자료형·배치의 계약이다. 각 파일의 실제 위치는 고정 commit 링크에서 확인할 수 있다.

| 파일과 주요 진입점 | 책임 | 여기서 풀어야 할 질문 |
|---|---|---|
| [tclmachine/__init__.py](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tclmachine/__init__.py) | 공개 호출 경계 | 사용자가 어떤 작업을 명시적으로 요청했는가 |
| [graph_capture.py · capture_for_export](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/graph_capture.py#L188-L195) | export·정규화 | 원본 의미가 보존되었는가 |
| [edf_export.py · bind_exported](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/edf_export.py) | 입력·가중치·출력 바인딩 | 어떤 값이 어느 ABI(Application Binary Interface) 인수인가 |
| [aten.py · plan_aten_call](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/aten.py#L94-L133) | 제한된 수축 후보 규칙 | 정적 shape·dtype·stride 계약에 맞는가 |
| [pointwise_rules.py](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/pointwise_rules.py) | 원소별 의미·broadcasting·TCL | 같은 의미 계열을 공통으로 해결하는가 |
| [reduction_rules.py](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/reduction_rules.py) | 축약 축·출력 형식 | 어느 축을 줄이고 무엇으로 누산하는가 |
| [compiler_registry.py](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/compiler_registry.py) | 의미 계열→컴파일 함수·소스 의존성 | 어떤 변경이 캐시를 무효화하는가 |
| [execution_plan.py · export_model](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/execution_plan.py#L331-L355) | 영역·컴파일 작업·계획 조립 | 값과 EDF를 어떤 순서로 연결하는가 |
| [tcl_compiler.py · compile_source](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/tcl_compiler.py#L158-L234) | 공식 컴파일·캐시·EDF 계약 | 산출물이 정확한 source/SDK와 일치하는가 |
| [compile_job.py](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/compile_job.py) | subprocess·시간제한·실패 증거 | compiler 거절과 환경 종료를 구분했는가 |
| [aten_runtime.py](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/aten_runtime.py) | 준비된 custom op 등록·복사한 FX graph 수정 | 실행 때 준비되지 않은 계약을 거부하는가 |
| [verified_build.py / verified_worker.py](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/verified_build.py) | 생산자와 독립된 소비자 검증 | 컴파일 없이 다시 로드되어 올바르게 실행되는가 |

## 새 연산을 추가할 때 지켜야 할 경계

모델 전체가 실패했다면 처음 막힌 작은 계산과 입력을 찾는다. ‘모델 A가 안 된다’보다 ‘이 모양의 배열을 전치해 더할 때 입력 검사가 실패한다’가 고칠 곳을 찾기 쉽다.

크기만 맞추면 끝나는 것은 아니다. 서로 다른 배열 이름이 같은 저장 공간을 가리키는 alias, 원래 배열의 값을 직접 바꾸는 mutation도 결과에 영향을 준다. 새 연산을 추가할 때는 값뿐 아니라 이런 동작도 보존해야 한다.

1. `graph_capture`에서 이미 의미가 틀렸는지 확인한다.
2. 바인딩·alias·layout 문제라면 그 경계를 고친다.
3. 기존 의미 계열이 있으면 해당 `*_rules.py`의 허용 조건과 TCL 생성기를 확장한다.
4. 신규 계열일 때만 `compiler_registry.py`에 함수와 source 의존성을 등록한다.
5. 작은 정상·경계·거부 사례에서 CPU 의미, 컴파일 ABI, NPU 수치와 trace를 확인한다.
6. 영향을 받은 모델만 다시 preflight하고, 마지막에 파일의 독립 소비자 replay를 수행한다.

이 순서는 현재 저장소의 수정 가이드와 맞춰 구성했다. 허용오차를 바꾸거나 모델명 예외로 통과시키면 지원 범위가 늘어난 것이 아니라 계약이 달라진 것이다. [수정 가이드](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/docs/REPAIR_GUIDE.md)

## 출처

| 자료 | 확인 범위 |
|---|---|
| [공개 facade](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tclmachine/__init__.py), [측정 선택 실행기](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/research/selected_runtime.py), [영역 분석기](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/region_planner.py) | optimize의 공개 인수, 서로 다른 plan의 생산·소비 계약 |
| [PyTorch Linear 2.10](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Linear.html) | 가중치 shape·전치·행렬곱·bias의 의미 |
| [README](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/README.md) | 목표·분석 코어·미구현 경계 |
| [ATen 후보 규칙](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/aten.py) | 정적 BF16 수축 admission |
| [준비 실행 경로](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/aten_runtime.py) | custom namespace 등록과 forward 경계 |
| [계획 생성기](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/execution_plan.py) | 다중 EDF·계획·생성 상태 |
| [Core 지원 기준선](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/docs/CORE_ATEN_WORKFLOW.md) | 과거 snapshot의 분모와 증거 단계 |
| [독립 소비자 검증](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/verified_worker.py) | 생산자 import·재컴파일 차단·NPU trace |

