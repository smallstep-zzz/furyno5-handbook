# 용어사전: 장치에서 서빙까지

모델의 계산, 장치 실행, 요청 처리에서 사용하는 말들을 모았다. 같은 이름이라도 역할이 다르면 구분해 설명한다. 약어의 영문 풀네임과 함께, 실제로 무엇을 하거나 가리키는지 읽으면 된다.

## 모델을 구성하고 실행하는 기초

입력에 2를 곱하고 1을 더하는 `y=x×2+1`을 작은 예로 사용한다. 예를 들어 x=4이면 y=9다.

| 용어 | 작은 예 | 뜻과 주의할 점 |
|---|---|---|
| 모델, model | 입력에 2를 곱하고 1을 더함 | 입력으로 답을 만드는 계산 방법과 그 계산에 사용하는 숫자들이다. |
| 가중치, weight | 입력에 곱하는 수 2 | 입력의 각 값이 계산에 얼마나 반영될지 정하는 수다. 학습에서 조정할 수 있다. |
| 바이어스, bias | 곱한 뒤 더하는 수 1 | 입력에 곱한 결과를 더하거나 빼서 보정하는 값이다. 입력과 곱하는 가중치와 역할을 구분한다. |
| 매개변수, parameter | 가중치 2와 바이어스 1 | 모델이 보관하고 학습으로 조정할 수 있는 값들이다. 이번 호출에서 새로 넣는 입력 x와 구분한다. |
| 추론, inference | x=4를 넣어 답 9를 구함 | 이미 정한 모델을 새 입력에 사용해 답을 계산하는 일이다. |
| 학습, training | 예제의 정답에 맞도록 곱할 수·더할 수를 찾음 | 모델이 내는 답과 정답을 비교하며 가중치·바이어스 같은 값을 조정하는 과정이다. |
| 커널, kernel | 곱셈과 덧셈을 함께 처리하는 장치 함수 | 장치에 맡겨 실행하는 계산 함수다. 모델의 계산을 한 커널로 묶거나 여러 커널로 나눌 수 있다. |
| ReLU | `max(0, −1)=0`, `max(0, 5)=5` | Rectified Linear Unit. 음수는 0으로, 0 이상은 그대로 두는 함수다. 배열에서는 각 원소에 적용한다. |
| 체크포인트, checkpoint | 저장한 가중치와 학습 상태 등 | 다시 읽어 모델을 구성하거나 학습을 이어갈 때 쓰는 저장물이다. 포함 항목은 형식에 따라 다르며, 가중치 파일을 곧바로 NPU 실행 바이너리로 보지 않는다. |

## 숫자·배열·행렬의 기초

작은 배열의 값이 실제로 바뀌는 과정은 01·02장에서 이어 볼 수 있다.

| 용어 | 작은 예 | 뜻과 주의할 점 | 관련 장 |
|---|---|---|---|
| 스칼라·벡터·행렬 | `3`, `[1,2,3]`, `[[1,2,3],[4,5,6]]` | 값 하나, 한 축을 가진 배열, 두 축을 가진 배열이다. 이 책에서 tensor는 이 배열 개념을 여러 축으로 넓힌 표현이다. | [01 계산 기초](01-hardware.md) |
| 축과 인덱스 | `A[m,k]` | 축은 좌표를 바꾸는 방향이고 인덱스는 그 방향의 위치다. 행렬에서 m은 행, k는 열을 고른다. | [01](01-hardware.md) |
| 내적 | `[1,2,3]·[1,0,1]=4` | 같은 위치끼리 곱한 뒤 모두 더한다. 길이가 같은 두 벡터에서 값 하나가 나온다. | [01](01-hardware.md), [04](04-tcl-edf.md) |
| 행렬곱 | `(2,3) @ (3,2) → (2,2)` | 왼쪽의 각 행과 오른쪽의 각 열을 내적해 출력 한 칸씩 만든다. 가운데 길이 3이 같아야 한다. | [01](01-hardware.md) |
| 원소별 곱 | `[1,2,3]*[1,0,1]=[1,0,3]` | 같은 위치끼리 곱하며 자동으로 합산하지 않는다. 행렬곱과 구별한다. | [01](01-hardware.md) |
| Σ와 합산 | `1+0+3=4` | 여러 항을 더한다는 표기다. 반복문에서 `total += term`을 수행하는 것과 연결해 읽는다. | [01](01-hardware.md) |
| `einsum` | `"mk,kn->mn"` | Einstein summation 표기로 배열 계산을 적는 NumPy 함수다. 이 표기에서는 m·n을 출력에 남기고 k를 곱한 뒤 합산한다. | [01](01-hardware.md) |
| `reshape` | `(6,) → (2,3)` | 총 원소 수를 유지하며 축 길이를 바꾼다. 축 순서를 바꾸는 전치와 다르고, 항상 복사 없이 가능한 것도 아니다. | [02](02-memory-dma.md) |
| 전치, transpose | `A.T` | 축의 역할을 교환한다. 행렬에서 `A[m,k]`가 전치 결과의 `[k,m]`에 해당한다. | [02](02-memory-dma.md) |
| layout, 메모리 배치 | 행부터 저장 / 열부터 저장 | 논리적인 배열의 값이 실제 메모리 위치에 대응하는 방식이다. 같은 shape라도 배치는 다를 수 있다. | [01](01-hardware.md), [02](02-memory-dma.md) |
| view | `x.T`가 x와 저장 공간 공유 | 같은 저장 공간을 다른 모양·stride로 해석하는 배열이다. 한쪽 변경이 다른 쪽에 보일 수 있다. | [02](02-memory-dma.md) |
| contiguous, 연속 배치 | C 순서에서 열 방향이 연속 | 지정한 순서로 원소를 차례로 읽을 수 있는 배치다. 전치된 값들을 C 순서로 만들면 복사가 필요할 수 있다. | [02](02-memory-dma.md) |
| broadcasting | `(2,3)+(3,) → (2,3)` | 작은 배열을 여러 위치에 반복 적용하는 모양 규칙이다. 모든 원소를 물리적으로 복제해야 한다는 뜻은 아니다. | [02](02-memory-dma.md) |
| 타일, tile | 큰 행렬을 작은 블록으로 나눔 | 한 번에 메모리에 두거나 계산할 작업 단위다. 타일 사이에 값을 재사용하거나 다음 타일 복사를 겹칠 수 있다. | [02](02-memory-dma.md), [07](07-optimization.md) |
| 부분합 | K의 앞부분 결과 1 + 뒷부분 결과 3 | 같은 출력 한 칸을 여러 작업이 나눠 계산했을 때 얻는 중간 합이다. 마지막에 합쳐야 전체 답 4가 된다. | [01](01-hardware.md), [15](15-multi-npu.md) |

계산 표기와 배열 동작은 [NumPy einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html), [matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html), [reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html), [배열·stride](https://numpy.org/doc/stable/reference/arrays.ndarray.html), [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)를 근거로 한다.

## 장치와 메모리

| 용어 | 풀네임 또는 공식 표기 | 쉽게 읽는 의미 | 관련 장 |
|---|---|---|---|
| NPU | Neural Processing Unit | 신경망 계산을 실행하는 가속기. 이 책의 대상 장치는 FuriosaAI의 RNGD다. | [1장 하드웨어](01-hardware.md) |
| RNGD | RNGD, 제품명 RENEGADE | FuriosaAI의 2세대 NPU 제품. 카드·칩·PE를 같은 단위로 세지 않는다. | [1장 하드웨어](01-hardware.md) |
| TCP | Tensor Contraction Processor | 많은 배열의 곱셈·합산을 처리하는 Furiosa의 프로세서 구조다. CPU 프로그램 전체가 이 하드웨어 안으로 들어간다는 뜻은 아니다. | [1장 하드웨어](01-hardware.md) |
| PE | Processing Element | RNGD 안에서 계산·메모리 자원을 묶은 처리 단위다. 한 RNGD에는 8개가 있다. 사용하는 PE 수와 카드 수는 구분한다. | [1장 하드웨어](01-hardware.md), [15장 여러 NPU](15-multi-npu.md) |
| Cluster | Cluster | PE들을 묶는 칩 내부의 공간 계층. vISA에서 어떤 축을 어느 자원에 펼칠지 설명할 때 등장한다. | [1장 하드웨어](01-hardware.md), [8장 vISA](08-eisa-visa.md) |
| Slice | Slice | PE 내부의 더 작은 공간 단위. 텐서 배치가 slice에 분산되는 방식은 메모리 접근과 계산 병렬성에 영향을 준다. | [1장 하드웨어](01-hardware.md) |
| Lane | Lane | slice 내부의 세부 데이터·연산 배치 단위. Python 텐서의 축 이름과 자동으로 일치하는 것은 아니다. | [1장 하드웨어](01-hardware.md), [8장 vISA](08-eisa-visa.md) |
| HBM | High Bandwidth Memory | 장치가 사용하는 큰 고대역폭 메모리. 가중치·입력·중간값·KV cache가 용량과 대역폭을 함께 사용한다. | [2장 메모리와 DMA](02-memory-dma.md) |
| SRAM | Static Random Access Memory | RNGD 칩 안의 빠른 메모리를 설명할 때 쓰는 메모리 기술명. HBM보다 용량이 작아 재사용·배치가 중요하다. | [1장 하드웨어](01-hardware.md), [2장 메모리와 DMA](02-memory-dma.md) |
| DM | Data Memory | 장치 안에서 이번에 계산할 값들을 준비하는 데이터 메모리다. 큰 배열을 보관하는 HBM에서 필요한 부분을 옮겨 사용할 수 있다. | [2장 메모리와 DMA](02-memory-dma.md) |
| SPM | Scratchpad Memory | 인덱스 목록 같은 보조 데이터를 명시적으로 놓는 장치 메모리다. CPU의 자동 캐시처럼 알아서 모든 값을 보관하는 공간으로 보지 않는다. | [2장 메모리와 DMA](02-memory-dma.md) |
| TRF | Tensor Register File | 텐서 계산 가까이에 값을 잠시 보관하고 재사용하는 하드웨어 레지스터 저장소. 여기서 file은 디스크 파일이 아니라 레지스터의 묶음이다. 용량·배치 제약이 타일과 스케줄 선택에 연결된다. | [1장 연산 파이프라인](01-hardware.md), [8장 vISA](08-eisa-visa.md) |
| VRF | Vector Register File | 벡터 연산에서 값을 보관하는 하드웨어 레지스터 저장소. TRF와 별도의 역할·접근 경로를 가지며 저장 장치의 파일이 아니다. | [1장 연산 파이프라인](01-hardware.md) |
| DMA | Direct Memory Access | 배열을 한 메모리에서 다른 메모리로 옮기는 일을 전송 엔진에 맡기는 방식이다. 요청한 뒤 언제 복사가 끝나는지는 여전히 확인해야 한다. | [2장 메모리와 DMA](02-memory-dma.md) |
| PDMA | PCIe DMA | host와 장치 HBM 사이를 PCIe로 이동하는 DMA 경로. 전송을 시작했다는 사실과 완료했다는 사실이 다르다. | [2장 메모리와 DMA](02-memory-dma.md) |
| TDMA | Tensor DMA | 장치 내부 텐서 이동에 사용하는 DMA 경로. 복사와 함께 배치·변환을 생각해야 한다. | [2장 메모리와 DMA](02-memory-dma.md) |
| PCIe | Peripheral Component Interconnect Express | host와 가속기 등을 연결하는 인터페이스. HBM 내부 대역폭과 host↔NPU 전송 대역폭은 다른 수치다. | [1장 하드웨어](01-hardware.md), [2장 메모리와 DMA](02-memory-dma.md) |
| NUMA | Non-Uniform Memory Access | host CPU와 메모리의 위치에 따라 접근 비용이 달라지는 구조. 장치에 가까운 CPU·메모리 배치가 전송 성능에 영향을 줄 수 있다. | [2장 메모리와 DMA](02-memory-dma.md), [15장 여러 NPU](15-multi-npu.md) |

RNGD의 구조는 [공식 하드웨어 개요](https://developer.furiosa.ai/v2026.4.0/en/overview/rngd.html), 세부 공간 계층과 메모리 이름은 [vISA 공간·시간 축](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/docs/src/mapping-tensors/spatial-temporal-dimensions.md), [메모리 공간](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/docs/src/moving-tensors/index.md), [DMA 문서](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html)를 기준으로 한다.

## 드라이버·컴파일러·런타임

| 용어 | 풀네임 또는 공식 표기 | 쉽게 읽는 의미 | 관련 장 |
|---|---|---|---|
| SDK | Software Development Kit | 컴파일러·런타임·개발용 패키지 등 장치를 개발에 쓰기 위한 도구 집합. 같은 SDK 세대라도 구성 요소의 정확한 버전을 기록한다. | [3장 소프트웨어 계층](03-driver-runtime.md), [출처와 기준](sources.md) |
| PERT | PE Runtime | PE에서 실행되며 host runtime과 통신하고 NPU task를 실행하도록 자원을 관리하는 계층. PE는 Processing Element다. | [3장 소프트웨어 계층](03-driver-runtime.md) |
| SMI | System Management Interface | 장치 정보·상태를 조회하는 관리 인터페이스. `furiosa-smi`의 상태 조회는 모델 추론 실행과 다르다. | [3장 소프트웨어 계층](03-driver-runtime.md) |
| ABI | Application Binary Interface | 컴파일한 프로그램과 호출자가 맞춰야 할 규칙이다. 예를 들어 입력 두 개의 순서·크기·자료형과 장치 구성이 일치해야 한다. | [4장 TCL과 EDF](04-tcl-edf.md), [11장 컴파일 경로](11-furyno5-compiler.md) |
| IR | Intermediate Representation | 컴파일러가 검사·변환하기 위해 만든 프로그램 표현이다. 예를 들어 ‘입력과 1을 더해 반환한다’는 계산 관계를 담는다. | [4장 TCL과 EDF](04-tcl-edf.md), [8장 vISA](08-eisa-visa.md) |
| 그래프 | Graph | 계산과 값의 연결을 점과 선으로 적은 구조다. 덧셈 결과를 ReLU 입력으로 보내는 관계가 한 예다. | [5장 실행계획](05-graphs-fallback.md) |
| 캡처 | Capture | 모델 코드를 분석해 어떤 계산이 어떤 값을 다음 계산에 넘기는지 기록하는 과정이다. 한 번 나온 답만 저장하는 일과 구분한다. | [11장 컴파일](11-furyno5-compiler.md) |
| 바인딩 | Binding | 프로그램의 인수 자리에 실제 값을 연결하는 일이다. 첫 인수에 이번 입력, 둘째 인수에 가중치를 넣는 경우를 생각하면 된다. | [11장 컴파일](11-furyno5-compiler.md) |
| 특수화 | Specialization | 아직 이름으로 둔 입력 길이 등에 실제 수를 대입해 구체적인 프로그램을 준비하는 일이다. M에 32를 넣는 것이 한 예다. | [4장 TCL 실습](04-tcl-edf.md) |
| 정규화 | Normalization | 컴파일 문맥에서는 같은 의미의 계산을 뒤 단계가 처리하기 쉬운 표현으로 정리하는 일이다. BatchNorm에서 숫자를 보정하는 뜻과 구분한다. | [11장 컴파일](11-furyno5-compiler.md) |
| 버퍼 | Buffer | 데이터를 담기 위해 확보한 메모리 영역이다. 전송이나 계산이 아직 읽는 동안에는 덮어쓰면 안 된다. | [2장 메모리](02-memory-dma.md) |
| 메타데이터 | Metadata | 실제 값이나 코드와 함께 보관하는 설명 정보다. 배열의 크기·자료형, 파일의 구성·버전 등이 해당한다. | [12장 패키지](12-fxbx.md) |
| MMIO | Memory-Mapped Input/Output | 장치의 제어 영역을 주소 공간에 매핑해 접근하는 방식. 일반 tensor 데이터 복사와 같은 것으로 보지 않는다. | [3장 소프트웨어 계층](03-driver-runtime.md) |
| BAR | Base Address Register | PCIe 장치의 메모리·제어 영역을 주소 공간에 연결할 때 사용하는 레지스터와 해당 매핑 영역을 가리키는 용어. | [3장 소프트웨어 계층](03-driver-runtime.md) |
| Compiler | Compiler | 작성한 계산을 장치용 프로그램으로 바꾸는 도구다. 같은 답을 만들면서 작업과 데이터 이동을 어떻게 배치할지 정한다. | [3장 소프트웨어 계층](03-driver-runtime.md), [11장 컴파일 경로](11-furyno5-compiler.md) |
| Runtime | Runtime | 준비한 프로그램을 읽어 실제 입력으로 호출하고 완료를 관리하는 소프트웨어다. 요청을 언제 받을지 정하는 서버의 스케줄러와는 역할을 구분한다. | [3장 소프트웨어 계층](03-driver-runtime.md), [5장 그래프와 폴백](05-graphs-fallback.md) |
| Runnable | `furiosa.torch.Runnable` 등 API 이름 | 준비된 프로그램을 호출할 수 있게 감싸는 객체. SDK `Runnable`과 Rust `furiosa-opt-rt`의 공개 인터페이스를 같은 API라고 가정하지 않는다. | [3장 소프트웨어 계층](03-driver-runtime.md), [4장 TCL과 EDF](04-tcl-edf.md) |
| EDF | Executable Device Format | 컴파일된 장치 실행물. SDK와 opt 문맥의 같은 명칭이 직렬화 포맷·ABI의 동일성이나 교차 적재 가능성을 보장하지 않는다. | [4장 TCL과 EDF](04-tcl-edf.md), [8장 vISA·Image](08-eisa-visa.md) |
| Device Function Image | Device Function Image | opt runtime이 적재하는 장치 함수 이미지. 코드와 인수·topology 등 런타임 계약을 함께 확인해야 한다. | [8장 vISA·Image](08-eisa-visa.md) |
| Relocation | Relocation | 미리 정하지 못한 주소 참조를 실제 배치에 맞게 연결·수정하는 처리. 공개 opt에서 확인한 것은 host 실행 파일의 registry 참조 relocation이며, 매 launch의 buffer 바인딩 및 SDK EDF 내부 코드 재배치와 구분한다. | [8장 relocation 범위](08-eisa-visa.md) |
| NoC | Network-on-Chip | 칩 내부 구성 요소들을 연결하는 통신망. 메모리 병목·전송 충돌의 영향이 timeout 등으로 드러날 수 있다. | [2장 메모리와 DMA](02-memory-dma.md) |

PERT의 공식 확장은 **PE Runtime**이다. [공식 소프트웨어 스택](https://developer.furiosa.ai/v2026.4.0/en/overview/software_stack.html)은 PE를 Processing Element로 풀고 펌웨어·PERT·host runtime의 책임을 구분한다. EDF의 풀네임은 [공식 Kernel Validation](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/docs/src/quick-start/kernel-validation.md)에서 확인했다. 이전 판의 “EDF 풀네임 미확인” 표기를 이 근거에 따라 수정했다.

## 커널·축·자료형

| 용어 | 풀네임 또는 공식 표기 | 쉽게 읽는 의미 | 관련 장 |
|---|---|---|---|
| TCL | Tensor Contraction Language | 배열의 어느 축을 곱하고 더할지 컴파일러에 설명하는 언어다. Python 문법을 사용해도 목적은 계산 값을 즉시 구하는 것이 아니라 계산 방법을 작성하는 것이다. | [4장 TCL과 EDF](04-tcl-edf.md) |
| eDSL | embedded Domain-Specific Language | Python·Rust 같은 host 언어 안에 들어 있는 특정 분야용 언어. 여기서는 컴파일 가능한 tensor 계산 표현을 만든다. | [4장 TCL과 EDF](04-tcl-edf.md), [8장 vISA](08-eisa-visa.md) |
| vISA | Virtual Instruction Set Architecture | 계산과 데이터 이동을 장치에 더 가까운 수준으로 표현하는 중간 표현이다. Rust 도구로 작성한 설명은 다시 컴파일해야 하며 텍스트를 NPU에 바로 보내 실행하지 않는다. | [8장 vISA와 eISA의 경계](08-eisa-visa.md) |
| eISA | 공식 풀네임·독립 공개 API 미확인 | 이 책의 공개 실습은 vISA API를 사용한다. eISA라는 이름을 같은 API의 다른 이름으로 가정하지 않는다. 별도의 공식 풀네임과 공개 실행 API는 확인하지 못했다. | [8장 vISA와 eISA의 경계](08-eisa-visa.md) |
| Axis | Axis | 텐서의 차원에 붙인 의미 있는 축. 행렬곱의 `M,K,N`에서 `K`를 축약하고 `M,N`을 출력에 남길 수 있다. | [4장 TCL과 EDF](04-tcl-edf.md), [7장 최적화](07-optimization.md) |
| Shape | Shape | 배열의 각 방향 길이다. 2행 3열이면 (2,3)이다. 프로그램이 처리하도록 준비한 크기와 실제 입력을 맞춰야 한다. | [6장 Shape와 버킷](06-buckets-shapes.md) |
| Stride | Stride | 배열에서 다음 행·열로 한 칸 갈 때 저장소에서 얼마나 건너뛸지 나타낸다. 연속인 2행 3열 배열은 행 방향으로 원소 3개, 열 방향으로 1개를 건너뛴다. | [2장 메모리와 DMA](02-memory-dma.md), [11장 컴파일 경로](11-furyno5-compiler.md) |
| Mapping | Mapping | 배열의 몇 번째 값을 장치의 어느 위치와 처리 순서에 놓을지 정한 대응이다. 계산식이 같아도 배치에 따라 데이터 이동량이 달라질 수 있다. | [7장 최적화](07-optimization.md), [8장 vISA](08-eisa-visa.md) |
| Fusion | Fusion | 문맥에 따라 여러 연산을 결합하는 최적화 또는 SDK에서 PE 구성을 정하는 값을 뜻한다. `set_fusion(8)`을 “연산 8개 합치기”로 읽지 않는다. | [3장 소프트웨어 계층](03-driver-runtime.md), [7장 최적화](07-optimization.md) |
| Reduction | Reduction | 여러 값을 더하거나 최댓값을 골라 더 적은 수의 값으로 모으는 계산이다. 예를 들어 [1,2,3]을 더해 6을 만드는 경우다. | [4장 TCL과 EDF](04-tcl-edf.md), [9장 ATen](09-torch-aten.md) |
| BF16 | bfloat16 | 숫자 하나를 16비트에 저장하는 부동소수점 형식이다. FP32보다 저장량이 적지만 가까운 숫자를 덜 구분한다. 결과 오차도 확인해야 한다. | [1장 하드웨어](01-hardware.md), [4장 TCL과 EDF](04-tcl-edf.md) |
| FP8 | 8-bit Floating Point | 8비트 부동소수점 형식의 계열. “FP8 지원”이라는 이름만으로 실제 format·scale·양자화 계약까지 같다고 볼 수 없다. | [1장 하드웨어](01-hardware.md), [13장 모델 포팅](13-porting-fha.md) |

[공식 TCL 소개](https://furiosa.ai/blog/furiosa-sdk-2026-3-a-new-kernel-framework-and-the-models-it-unlocks)와 [furiosa-opt 소개](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/docs/src/introduction.md)를 기준으로 언어와 실행 도구를 구분했다. `eISA`의 미확인 상태는 [8장](08-eisa-visa.md)에서 조사 범위와 함께 설명한다.

## Torch의 연산 경계

| 용어 | 풀네임 또는 공식 표기 | 쉽게 읽는 의미 | 관련 장 |
|---|---|---|---|
| ATen | A Tensor Library | PyTorch의 배열 연산 계층이다. 덧셈이나 행렬곱 요청을 나타내며, 그 이름 하나가 장치 명령어 한 개라는 뜻은 아니다. | [9장 Torch와 ATen](09-torch-aten.md) |
| Operator schema | Operator schema | 연산 함수의 인수와 반환값에 대한 설명이다. 무엇을 몇 개 받아 무엇을 돌려주는지와 원본 값을 바꾸는지 등을 나타낸다. | [9장 Torch와 ATen](09-torch-aten.md) |
| Overload | Overload | 같은 기본 이름을 인수 형태별로 구분한 것이다. 예를 들어 배열끼리의 덧셈과 배열에 숫자 하나를 더하는 호출을 구분한다. | [9장 Torch와 ATen](09-torch-aten.md) |
| Dispatcher | Dispatcher | 현재 연산과 tensor의 장치·상태 등을 보고 어느 구현을 호출할지 고르는 PyTorch 계층이다. | [9장 Torch와 ATen](09-torch-aten.md) |
| Dispatch key | Dispatch key | 장치·autograd 등 구현 선택에 관여하는 구분자. `torch.compile`의 backend 등록과는 다른 등록 경계다. | [9장 Torch와 ATen](09-torch-aten.md) |
| FX | `torch.fx`, 공식 API 이름 | PyTorch 프로그램을 그래프로 표현·변환하는 도구. 이 책에서는 확인하지 않은 영문 풀네임을 붙이지 않는다. | [9장 Torch와 ATen](09-torch-aten.md), [11장 컴파일 경로](11-furyno5-compiler.md) |
| Decomposition | Decomposition | 복합 계산을 더 기본적인 계산들의 조합으로 바꾸는 규칙이다. 바꾼 뒤 생긴 연산들도 장치에서 처리할 수 있어야 한다. | [9장 Torch와 ATen](09-torch-aten.md), [11장 컴파일 경로](11-furyno5-compiler.md) |

Torch 용어는 [ATen 문서](https://docs.pytorch.org/cppdocs/api/aten/index.html), [Dispatcher 설명](https://docs.pytorch.org/tutorials/advanced/dispatcher.html), [FX 문서](https://docs.pytorch.org/docs/stable/fx.html)를 참고한다. 이 책의 연산 개수는 버전과 집계 단위를 고정한 결과이며 용어 자체의 영구 속성이 아니다.

## 프로젝트와 실행 산출물

| 용어 | 풀네임 또는 공식 표기 | 쉽게 읽는 의미 | 관련 장 |
|---|---|---|---|
| FXB | Furiosa Executable Bundle | 컴파일한 장치 프로그램과 구성 설명 등을 묶는 공식 형식이다. 모델 가중치·tokenizer·설정 같은 다른 자산도 실행에 필요하다. | [3장 소프트웨어 계층](03-driver-runtime.md), [12장 fxbX](12-fxbx.md) |
| fxbX | fxbX, 프로젝트명 | FuryNo5가 만든 실행 계획과 실행물 등을 포장·검사·적재하는 사용자 프로젝트. 공식 FXB 지원과 동의어가 아니다. | [12장 fxbX](12-fxbx.md) |
| FuryNo5 | FuryNo5, 프로젝트명 | Torch 그래프의 의미를 분석하고 TCL·EDF·실행 계획을 생산하는 프로젝트. 실제 import 이름은 `tclmachine`이며 현재 전체 ATen 구현 완료를 뜻하지 않는다. | [10장 FuryNo5 구조](10-furyno5-architecture.md) |
| FHA | Furiosa-Hug-Anything | 모델 포팅 절차·연결·검증을 자동화하는 프로젝트. FuryNo5의 연산 생산자 및 fxbX의 실행 소비자 역할과 구분한다. | [13장 모델 포팅](13-porting-fha.md) |
| FuryAwesome | FuryAwesome, 프로젝트명 | 포팅 모델의 사용법·배포 링크·검증 정보를 공유하고 요청을 모으는 카탈로그. 컴파일러나 장치 런타임이 아니다. | [13장 역할과 게시](13-porting-fha.md) |
| Manifest | Manifest | 묶음 파일의 목차와 설명에 해당한다. 어떤 프로그램 파일이 들어 있고 어떤 버전·설정을 사용하는지 적는다. | [12장 fxbX](12-fxbx.md) |
| Fingerprint | Fingerprint | 소스·컴파일 옵션·SDK·입출력 계약 등의 동일성을 판단하기 위해 기록하는 식별값. 파일 이름만 같은 캐시의 잘못된 재사용을 막는 데 쓴다. | [11장 컴파일 경로](11-furyno5-compiler.md) |
| Execution plan | Execution plan | 프로그램 사이의 호출 순서와 값 연결을 적은 계획이다. ‘덧셈의 답을 ReLU 입력에 넣는다’는 연결이 한 예다. | [5장 그래프와 폴백](05-graphs-fallback.md), [12장 fxbX](12-fxbx.md) |

[공식 FXB 문서](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/fxb.html)와 사용자 프로젝트의 [FuryNo5 README](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/README.md), [fxbX README](https://github.com/smallstep-zzz/fxbX/blob/03d993b4d1b36cdda209d61bbf447434c174b401/README.md), [FHA 역할 설명](https://github.com/smallstep-zzz/Furiosa-Hug-Anything/blob/8fc675f488763bef7c779ee037e8393ddd9f856a/docs/FHA_ROLE_TRANSITION.md)을 서로 다른 근거로 읽는다.

## LLM 실행과 서빙

| 용어 | 풀네임 또는 공식 표기 | 쉽게 읽는 의미 | 관련 장 |
|---|---|---|---|
| LLM | Large Language Model | 문장을 숫자로 처리해 다음 토큰 등을 예측하는 큰 언어 모델이다. 서비스에서는 여러 요청의 진행 상태도 따로 관리해야 한다. | [3장 소프트웨어 계층](03-driver-runtime.md), [14장 vLLM 서빙](14-serving-vllm.md) |
| KV cache | Key–Value cache | 다음 토큰을 만들 때 다시 참고할 과거 위치의 key·value 배열을 보관한 것이다. 어느 요청에서 만든 정보인지 구분해야 한다. | [6장 Shape와 버킷](06-buckets-shapes.md), [14장 vLLM 서빙](14-serving-vllm.md) |
| Prefill | Prefill | 처음 받은 입력 토큰들을 처리해 이후 생성에 사용할 정보를 준비하는 단계다. | [6장 Shape와 버킷](06-buckets-shapes.md) |
| Decode | Decode | 앞에서 계산한 정보를 사용해 다음 토큰을 만들고, 그 토큰을 다시 넣어 생성을 이어가는 반복 단계다. | [6장 Shape와 버킷](06-buckets-shapes.md), [14장 vLLM 서빙](14-serving-vllm.md) |
| Bucket | Bucket | 길이 8·16처럼 입력 조건별로 미리 준비한 실행 선택지다. 입력 길이 5를 8칸짜리에 넣는 식으로 고를 수 있다. | [6장 Shape와 버킷](06-buckets-shapes.md) |
| TP | Tensor Parallelism | 하나의 계산을 여러 계산 자원에 나누는 병렬화. 공식 Furiosa-LLM의 PE 단위와 opt plugin의 chip 단위 설정을 구분한다. | [15장 여러 NPU](15-multi-npu.md) |
| PP | Pipeline Parallelism | 모델의 단계·층을 여러 자원에 나누고 데이터가 순서대로 지나가게 하는 병렬화. 단계 간 전송과 pipeline 대기 비용이 생긴다. | [15장 여러 NPU](15-multi-npu.md) |
| DP | Data Parallelism | 모델 복제본들이 서로 다른 요청·데이터를 처리하는 병렬화. 각 복제본의 가중치와 cache 용량이 필요하다. | [15장 여러 NPU](15-multi-npu.md) |
| TTFT | Time To First Token | 요청 후 첫 토큰을 받기까지의 시간. 큐 대기·prefill·초기 실행 등 측정에 포함한 범위를 함께 기록한다. | [6장 Shape와 버킷](06-buckets-shapes.md), [14장 vLLM 서빙](14-serving-vllm.md) |
| TPOT | Time Per Output Token | 출력 토큰당 걸리는 시간. 첫 토큰 포함 여부·평균 방식 등 benchmark 정의를 명시한다. | [7장 최적화](07-optimization.md), [14장 vLLM 서빙](14-serving-vllm.md) |
| vLLM | vLLM, 프로젝트명 | LLM 추론·서빙 프레임워크. Furiosa opt plugin은 이를 장치 엔진과 연결하지만 fxbX를 자동으로 읽는 범용 loader는 아니다. | [14장 vLLM 서빙](14-serving-vllm.md) |
| Continuous batching | Continuous batching | 실행이 진행되는 동안 끝난 요청을 빼고 새 요청을 넣어 매 step의 배치를 조정하는 방식. 고정 batch 한 번의 실행과 다르다. | [14장 vLLM 서빙](14-serving-vllm.md) |

병렬화 단위는 [Furiosa-LLM의 모델 병렬화 문서](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/model-parallelism.html), opt plugin의 경계는 [고정 커밋의 vLLM 연결 문서](https://github.com/furiosa-ai/furiosa-opt-models/blob/c982976100d18d7a9641d5a8c22b89cec4544049/docs/serving/vllm.md)와 실제 코드를 함께 확인한다. 같은 TP·DP라는 단어를 사용해도 설정 단위와 지원 조합이 같다는 보장은 없다.

## 생태계 비교와 모델 사전 최적화

| 용어 | 풀네임 또는 공식 표기 | 쉽게 읽는 의미 | 관련 장 |
|---|---|---|---|
| AOT | Ahead-of-Time compilation | 실행 요청 전에 미리 컴파일해 실행물을 준비하는 방식. 준비 비용과 반복 실행 비용을 분리한다. | [18장 다른 NPU](18-npu-ecosystems.md) |
| JIT | Just-in-Time compilation | 실행 시 필요한 프로그램을 컴파일하는 방식. 첫 호출·새 shape·캐시 적중 조건을 확인한다. | [18장 다른 NPU](18-npu-ecosystems.md) |
| ONNX | Open Neural Network Exchange | 모델의 계산 연결과 연산을 도구 사이에서 주고받는 형식이다. 대상 장치의 컴파일러가 그 내용을 지원해야 장치 프로그램으로 바뀐다. | [19장 사전 최적화](19-model-optimization.md) |
| MLIR | Multi-Level Intermediate Representation | 여러 수준의 중간 표현을 함께 다루는 컴파일러 기반. 그것만으로 특정 NPU 지원이 생기는 것은 아니다. | [18장 TT-XLA](18-npu-ecosystems.md) |
| PJRT | Portable JAX Runtime | frontend와 장치 backend를 연결하는 실행·컴파일 인터페이스. TT-XLA가 사용하는 통합 경계다. | [18장 TT-XLA](18-npu-ecosystems.md) |
| StableHLO | Stable High Level Operations | 고수준 텐서 계산을 전달하는 안정적인 연산 표현. TT-XLA의 frontend와 컴파일러 사이에 사용된다. | [18장 TT-XLA](18-npu-ecosystems.md) |
| NEFF | Neuron Executable File Format | AWS Neuron의 실행물 형식. Furiosa EDF와 호환되는 포맷이라는 뜻이 아니다. | [18장 Neuron](18-npu-ecosystems.md) |
| Composite op | Composite operation | 여러 기본 계산에 높은 수준의 의미를 붙여 컴파일러가 하나의 패턴으로 다룰 수 있게 한 연산. | [18장 의미 보존](18-npu-ecosystems.md), [19장 최적화 연구](19-model-optimization.md) |
| PTQ | Post-Training Quantization | 학습을 마친 모델의 숫자를 더 적은 비트 등으로 표현하도록 바꾸는 방법이다. 저장량을 줄일 수 있지만 값의 오차와 모델 품질을 확인해야 한다. | [19장 양자화](19-model-optimization.md) |
| QAT | Quantization-Aware Training | 숫자를 작게 저장할 때 생길 오차를 학습 중에 반영하는 방법이다. 학습 결과의 저장·계산 조건이 실제 장치와 맞아야 한다. | [19장 양자화](19-model-optimization.md) |
| Structured pruning | Structured pruning | 출력 특징이나 채널처럼 연결된 구조를 실제로 없애 배열 크기를 줄이는 방법이다. 뒤 층이 받는 입력 크기도 함께 바뀐다. | [19장 구조적 가지치기](19-model-optimization.md) |
| Distillation | Knowledge distillation | 큰 모델의 답이나 중간 결과를 작은 모델의 학습 목표로 사용하는 방법이다. 새 모델의 실제 과제 품질을 다시 평가한다. | [19장 모델 압축](19-model-optimization.md) |
| NetsPresso | NetsPresso, Nota의 제품명 | 모델 압축·변환·프로파일링을 다루는 도구와 서비스. 공개 Python client와 전체 서비스 구현의 공개 범위를 구분한다. | [19장 NetsPresso](19-model-optimization.md) |

이 표의 업체별 용어는 [18장의 공식 출처](18-npu-ecosystems.md), 모델 변환 도구와 지원 범위는 [19장의 라이브러리·공식 논문](19-model-optimization.md)에 연결했다.

## 출처

| 근거 | 이 페이지에서 사용하는 범위 |
|---|---|
| [FuriosaAI RNGD](https://developer.furiosa.ai/v2026.4.0/en/overview/rngd.html) | 장치 이름·TCP·PE·메모리의 출발점 |
| [Furiosa 소프트웨어 스택](https://developer.furiosa.ai/v2026.4.0/en/overview/software_stack.html) | 드라이버·PERT·컴파일러·런타임 책임 |
| [furiosa-opt 소개](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/docs/src/introduction.md) | 공개 vISA 표기와 개발 경로 |
| [TCL 도입 설명](https://furiosa.ai/blog/furiosa-sdk-2026-3-a-new-kernel-framework-and-the-models-it-unlocks) | TCL의 정식 명칭과 역할 |
| [ATen](https://docs.pytorch.org/cppdocs/api/aten/index.html) · [FX](https://docs.pytorch.org/docs/stable/fx.html) | PyTorch 계층과 명칭 |
| [공식 FXB](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/fxb.html) | 공식 형식의 정식 명칭 |
| [출처와 재현 기준](sources.md) | 버전·공개 범위·검증 수준 |
