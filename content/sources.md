# 출처·버전·재현 기준

**이 핸드북의 조사 기준일은 2026-11-19다.** 공식 문서, 공개 구현, 사용자 프로젝트의 소스, 저장소에 보존된 기존 시험 결과를 함께 읽었다. 이들은 서로 다른 종류의 근거다. 코드가 존재하는 것, 공식 지원을 약속한 것, 특정 장비에서 한 번 측정한 것, 이번 편집 과정에서 새로 실행한 것을 같은 수준의 사실로 쓰지 않는다.

각 장은 주장 가까이에 출처를 붙이고 마지막에 핵심 출처 표를 둔다. 모든 인용 URL은 빌드 과정에서 생성하는 [전체 출처 색인](data/source-index.json)에서 확인할 수 있다. 이 페이지는 그 자료를 어떤 기준으로 읽고 재현해야 하는지 설명한다.

## 버전 기준: URL의 숫자와 설치본은 다를 수 있다

| 대상 | 이 책의 기준 | 독자가 확인할 점 |
|---|---|---|
| 공식 SDK 문서 | 2026.4 계열 중심 | 조회 당시 `/v2026.4.0/` URL의 페이지 제목·헤더가 **2026.4.1**로 표시됐다. URL만으로 정확한 2026.4.0 API를 보장하지 않는다. |
| 2026.5 기능 | 해당 beta release note에 한정 | 예를 들어 `--evaluate-tcl`은 2026.5.0b0 변경으로 구분했다. 안정판 2026.4에 그대로 복사하지 않는다. |
| furiosa-opt | `3d43bc9…` 고정 소스 | 예제 template의 `furiosa-opt-std`는 0.10 계열이다. 0.8.1 포럼 토론과 현재 소스는 시점이 다르다. |
| furiosa-opt-models | `c982976…` 고정 소스 | opt-vLLM 설치는 이 저장소의 compiler preview revision·wheel·Python 조건을 함께 맞춰야 한다. 일반 opt 예제 환경과 무조건 호환된다고 보지 않는다. |
| PyTorch 연산 목록 | v2.10.0 소스 스냅샷 | 연산 이름 수·overload 수·Core 목록 수는 다른 집계다. 미래 버전에서 같은 숫자라고 가정하지 않는다. |
| 사용자 프로젝트 | 아래 표의 고정 커밋 | `main` 최신 상태나 배포된 wheel과 이 책의 API가 같다는 보장은 없다. |

SDK 문서의 버전 표시 차이는 [소프트웨어 스택](https://developer.furiosa.ai/v2026.4.0/en/overview/software_stack.html)과 [RNGD 개요](https://developer.furiosa.ai/v2026.4.0/en/overview/rngd.html)에서 확인했다. 2026.5 비용 추정은 [해당 release note](https://developer.furiosa.ai/v2026.5.0b4/en/whatsnew/release-2026.5.0b0.html), opt 버전은 [template 의존성](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/base-template/Cargo.toml.liquid), opt-vLLM의 별도 조건은 [설치 문서](https://github.com/furiosa-ai/furiosa-opt-models/blob/c982976100d18d7a9641d5a8c22b89cec4544049/docs/getting-started/installation.md)와 [Cargo 의존성](https://github.com/furiosa-ai/furiosa-opt-models/blob/c982976100d18d7a9641d5a8c22b89cec4544049/Cargo.toml)을 근거로 한다.

CLI(Command-Line Interface) 옵션은 설치본의 `--help`와 함께 확인한다. 문서의 명령이 오래된 설치본에서 거절되면 옵션을 임의로 바꿔 성공한 것처럼 해석하지 말고, 지원 버전과 대체 절차를 먼저 찾는다. API(Application Programming Interface) 예제도 함수 이름뿐 아니라 인수·반환형·입출력 ABI(Application Binary Interface)까지 해당 버전과 대조해야 한다.

## 고정한 소스 스냅샷

커밋 링크에는 전체 40자리 SHA가 들어 있다. 표의 짧은 표기는 읽기 편하게 줄인 표시다. 공개·비공개 여부는 조사 당시의 저장소 메타데이터 기준이며 나중에 바뀔 수 있다.

| 저장소 | 고정 커밋 | 자료 성격·접근 범위 | 이 책에서 확인한 역할 |
|---|---|---|---|
| furiosa-ai/furiosa-opt | [3d43bc9d20f3](https://github.com/furiosa-ai/furiosa-opt/tree/3d43bc9d20f324af60e3006c3f1541938d7f5097) | Furiosa 공식 공개 저장소 | vISA, 공간·메모리 mapping, 컴파일 도구, 정적 schedule, Rust runtime |
| furiosa-ai/furiosa-opt-models | [c982976100d1](https://github.com/furiosa-ai/furiosa-opt-models/tree/c982976100d18d7a9641d5a8c22b89cec4544049) | Furiosa 공식 공개 저장소 | 모델 engine, vLLM plugin, 지원 옵션 검사, 설치 의존성 |
| pytorch/pytorch | [449b17684101](https://github.com/pytorch/pytorch/tree/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0) | PyTorch 공식 공개 저장소, v2.10.0 | `native_functions.yaml`을 기준으로 ATen schema와 overload를 재현 가능하게 집계 |
| smallstep-zzz/FuryNo5 | [bba9efea8201](https://github.com/smallstep-zzz/FuryNo5/tree/bba9efea8201aec48150514a35d0b637d29d107e) | 사용자 프로젝트, **접근 권한 필요** | 실제 import 이름, 지원 경계, capture·TCL·EDF·cache·실행 계획 생산 |
| smallstep-zzz/fxbX | [03d993b4d1b3](https://github.com/smallstep-zzz/fxbX/tree/03d993b4d1b36cdda209d61bbf447434c174b401) | 사용자 프로젝트, 공개 저장소 | package 검사·외부 weights·EDF 적재·실행 계획 소비·현재 서빙 경로 |
| smallstep-zzz/Furiosa-Hug-Anything | [8fc675f48876](https://github.com/smallstep-zzz/Furiosa-Hug-Anything/tree/8fc675f488763bef7c779ee037e8393ddd9f856a) | 사용자 프로젝트, **접근 권한 필요** | 포팅 역할 분담, producer bridge, 검증·자동화 절차 |
| smallstep-zzz/furyawesome | [c668ecbceac2](https://github.com/smallstep-zzz/furyawesome/tree/c668ecbceac2bedd04c6e539c1a5509a404a464d) | 사용자 프로젝트, 공개 저장소 | 모델 카탈로그·요청·사용법·HF 배포 참조와 게시 동기화 |
| smallstep-zzz/FuryBoost | [62faf49a29cc](https://github.com/smallstep-zzz/FuryBoost/tree/62faf49a29cc40230ee3df271b8f91ad8488eb5d) | 사용자 프로젝트, **접근 권한 필요** | 트리 학습 구현, CPU 비교 원자료, 최적화 전후 결과 |
| Ascend/pytorch | [2373dee2db39](https://github.com/Ascend/pytorch/tree/2373dee2db393489d91d44f5fbefd2d6cef30439) | 업체 공식 공개 소스 | TorchNPU·PrivateUse1·기본 장치 통합 |
| RBLN-SW/optimum-rbln | [370e2c98e551](https://github.com/RBLN-SW/optimum-rbln/tree/370e2c98e5511373ca3f6454b444c327223822f7) | 업체 공식 공개 어댑터 | Hugging Face 모델과 컴파일·적재 경로 연결. 의존 compiler 공개와 구분 |
| tenstorrent/tt-xla | [fdc3b7773674](https://github.com/tenstorrent/tt-xla/tree/fdc3b7773674081b8f8c0698de3400d8d899dc8f) | 업체 공식 공개 소스 | PJRT·StableHLO·fusion·composite와 TT-Torch 전환 안내 |

fxbX가 공개 저장소라는 사실은 Furiosa의 공식 지원 형식이라는 뜻이 아니다. 반대로 비공개 프로젝트의 소스가 근거로 쓰였다는 사실은 독자에게 자동으로 해당 저장소의 접근 권한이 생긴다는 뜻도 아니다. 권한 없는 독자도 큰 흐름을 읽을 수 있도록 본문에 역할·핵심 코드·확인 범위를 설명했지만, 원본 전체를 독립적으로 재검토하려면 저장소 접근이 필요하다.

## 근거를 다섯 종류로 나눠 읽는다

| 근거 종류 | 설명할 수 있는 것 | 그 자료 하나로 확정할 수 없는 것 |
|---|---|---|
| 공식 문서·release note | 공개 API의 사용법, 명시한 지원 조건, 버전별 변경 | 모든 내부 호출 경로, 임의 버전과 모델의 실측 성공 |
| 공식 또는 사용자 프로젝트 소스 | 해당 커밋의 분기·인수·검사·산출물·오류 처리 | 배포 wheel이 같은 코드인지, 모든 장비에서 통과하는지 |
| 공식 포럼 답변 | 특정 날짜·버전·질문에 대한 개발팀 설명 | 이후 릴리스에도 남아 있는 영구 제약 |
| 보존된 시험 로그·보고서 | 기록된 입력·환경·측정 범위에서 얻은 결과 | 이번 편집 환경의 새 실행, 조건을 바꾼 일반적인 성능 우위 |
| 이 책의 해석·제안 | 기존 근거를 연결한 개발 순서, 검증 방법, 남은 설계 | 제품의 공식 보증이나 구현 완료 상태 |

예를 들어 2026-10-12의 [opt 0.8.1 host CPU 토론](https://forums.furiosa.ai/t/furiosa-opt-v0-8-1-launch-host-cpu-cpu-2/479)은 당시 동작을 설명한다. 이후 고정 소스의 [launch.rs](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/furiosa-opt-std/src/runtime/launch.rs)에는 `Prepared.submit()`과 `Submitted`가 있으므로 과거 답변을 그대로 “현재도 미구현”이라고 옮기지 않았다.

또한 [compile_from_visa 포럼 답변](https://forums.furiosa.ai/t/furiosa-opt-compile-from-visa-furiosa-torch/487)은 답변 시점에 해당 Python 연결이 내부 전용임을 설명한다. 이 자료만으로 이후 모든 버전의 공개 여부를 확정할 수 없으므로, 이 책은 독자가 쓸 수 있다고 확인한 공개 Rust 경로를 설명하고 별도 Python 연결은 미확인 경계로 남겼다.

## 핵심 공식 문헌

| 자료 | 읽는 목적 | 연결되는 장 |
|---|---|---|
| [TCP: A Tensor Contraction Processor for AI Workloads, ISCA 2024](https://furiosa.ai/download/FuriosaAI-tensor-contraction-processor-isca24) | tensor contraction 중심 아키텍처의 설계 배경. 논문 시점 수치와 현재 제품 사양은 구분 | [1장](01-hardware.md), [2장](02-memory-dma.md) |
| [RNGD 하드웨어 개요](https://developer.furiosa.ai/v2026.4.0/en/overview/rngd.html) | 제품의 공개 사양과 단위 | [1장](01-hardware.md) |
| [Furiosa 소프트웨어 스택](https://developer.furiosa.ai/v2026.4.0/en/overview/software_stack.html) | driver·firmware·PERT·compiler·runtime의 책임 | [3장](03-driver-runtime.md) |
| [SDK prerequisites](https://developer.furiosa.ai/v2026.4.0/en/get_started/prerequisites.html) · [SMI CLI](https://developer.furiosa.ai/v2026.4.0/en/device_management/system_management_interface/furiosa_smi_cli.html) | 장치와 호환 소프트웨어 준비·상태 확인 | [3장](03-driver-runtime.md), [4장](04-tcl-edf.md) |
| [TCL을 소개한 SDK 2026.3 글](https://furiosa.ai/blog/furiosa-sdk-2026-3-a-new-kernel-framework-and-the-models-it-unlocks) | Python 커널 언어의 도입 의도 | [4장](04-tcl-edf.md) |
| [furiosa-opt Book](https://developer.furiosa.ai/furiosa-opt/book/quick-start/setup-and-tooling.html) | Rust eDSL 개발 도구와 vISA 커널 실습의 진입점 | [2장](02-memory-dma.md), [8장](08-eisa-visa.md) |
| [공식 FXB](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/fxb.html) | manifest·EDF·호환 cache·공식 bundle의 경계 | [5장](05-graphs-fallback.md), [6장](06-buckets-shapes.md), [12장](12-fxbx.md) |
| [공식 모델 병렬화](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/model-parallelism.html) | PE 단위 TP·PP·DP 구성과 컴파일 시점의 결정 | [15장](15-multi-npu.md) |
| [PyTorch Dispatcher](https://docs.pytorch.org/tutorials/advanced/dispatcher.html) · [ATen](https://docs.pytorch.org/cppdocs/api/aten/index.html) | 연산 계약과 장치 구현 선택의 관계 | [9장](09-torch-aten.md) |
| [vLLM plugin system](https://docs.vllm.ai/en/stable/design/plugin_system/) | 외부 장치 backend와 서빙 프레임워크를 연결하는 경계 | [14장](14-serving-vllm.md) |

공식 홈페이지의 개요는 개념의 출발점으로 쓰고, 구체적인 함수 호출·호환성·오류 처리는 가능한 한 해당 문서나 구현으로 내려가 확인했다. 사용자 프로젝트의 ATen 지원 범위도 README의 목표 문구만으로 판정하지 않고 실제 registry·변환 규칙·실행 경로를 구분했다.

## 기초 계산과 한국어 도식의 출처

행렬곱·`einsum`·`reshape`·전치·stride·broadcasting은 NumPy 공식 문서로 동작을 확인하고, 본문에서는 별도로 만든 작은 배열로 계산을 풀었다. 실제 값은 [tensor_basics.py](examples/tensor_basics.py)와 [array_layout.py](examples/array_layout.py)에서 다시 계산할 수 있다. NumPy의 stride는 바이트 수이고 PyTorch의 stride는 원소 단위이므로 예제를 옮길 때 단위를 유지해야 한다. [NumPy ndarray](https://numpy.org/doc/stable/reference/arrays.ndarray.html) · [PyTorch Tensor.stride](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.stride.html)

01·02장의 메모리와 연산 경계 그림은 공식 furiosa-opt의 [Moving Tensors](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/docs/src/moving-tensors/index.md)와 [Computing Tensors](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/docs/src/computing-tensors/index.md)에 근거해 한국어로 재구성했다. 원본 그림을 그대로 번역한 것으로 표시하지 않고, 학습에 필요한 경로로 나누거나 단순화한 사실을 설명한다. 라이선스는 [Apache-2.0 전문](assets/licenses/furiosa-opt-Apache-2.0.txt), 권리 표시는 [Furiosa Opt NOTICE](assets/licenses/furiosa-opt-NOTICE.txt)에 보존했다.

## 코드 검사와 NPU 실측의 구분

**이번 핸드북 작성 환경에서는 RNGD에서 새 모델을 컴파일·실행하거나 새 성능 수치를 측정하지 않았다.** 새로 추가한 NumPy 기초 예제 두 개는 CPU에서 실제 실행해 배열 값과 shape·stride·메모리 공유를 확인했다. 장치 호출 API·정의 파일·CLI 구성·버전·실행 전제는 소스와 대조하고, 완전한 Python 예제의 문법을 검사했다. 예를 들어 5장의 병렬 컴파일 예제는 소스가 `spawn` 방식의 프로세스를 사용함을 확인하고 `main()` 보호를 추가했다. 이는 재현성을 높이는 코드 검토 결과이지 NPU 통과 결과는 아니다. [프로세스 생성 구현](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/execution_plan.py)

8장의 Schedule Viewer 수치는 **컴파일러가 만든 정적 schedule의 비용**이다. 실제 host 대기·전송·장치 상태·요청 분포를 포함한 benchmark 결과로 바꾸어 읽지 않는다. [Schedule Viewer 설명](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/docs/src/tools/schedule-viewer.md)

7장과 16장의 기존 성능 값은 해당 저장소의 보고서와 원자료를 설명한 것이다. 특히 FuryBoost는 NPU 측정값과 CPU 비교값의 tree 설정·threads·데이터 준비 범위·반복 조건을 함께 읽어야 한다. 하나의 kernel 개선과 전체 학습 시간 개선을 같은 값으로 취급하지 않는다. [FuryBoost 비교 원자료](https://github.com/smallstep-zzz/FuryBoost/blob/62faf49a29cc40230ee3df271b8f91ad8488eb5d/reports/rngd-server3/higgs-dev2/full/comparison.json), [histogram 변경 전후](https://github.com/smallstep-zzz/FuryBoost/blob/62faf49a29cc40230ee3df271b8f91ad8488eb5d/reports/rngd-server3/grouped-histograms/comparison.json)

예제마다 다음 범위를 구별한다.

| 표기·형태 | 독자가 할 수 있는 해석 |
|---|---|
| 완전한 `.py` 예제 | 명시한 패키지·장치·파일을 준비하고 독립 스크립트로 재현할 수 있도록 구성했다. 이 문서에서 NPU로 검증했다는 뜻은 아니다. |
| “핵심 발췌”·함수 내부 fragment | 원본의 책임과 중요한 동작을 설명한다. 주변 정의·인수·호출자가 별도로 필요하다. |
| “의사코드”·설계 표 | 변경 방향이나 계약을 설명한다. 아직 존재하지 않는 API를 실행 명령처럼 제시한 것이 아니다. |
| “기존 실측” | 원자료에 남은 결과다. 조건·단위·측정 범위를 유지해야 비교할 수 있다. |
| “미확인”·“개발 과제” | 공개 근거 또는 구현·검증이 충분하지 않다. 완료된 기능으로 의존해서는 안 된다. |

## 재현 결과에 남길 최소 기록

먼저 **어떤 프로그램을 어떤 환경에서 어떤 기준으로 검증했는지**를 고정한다. 이 순서는 비교를 가능하게 하기 위한 핸드북의 권장 방법이며 모든 서비스에 같은 최적화 우선순위를 강요하는 규칙은 아니다.

1. **환경**: SDK·compiler·driver·firmware·Python·Torch 버전, 프로젝트 commit, 장치 수·PE 구성·mesh, host CPU와 NUMA 배치.
2. **의미**: 모델·가중치 revision, shape·dtype·stride, mask·상태 갱신 의미, reference 구현, 허용오차와 과제 품질 기준.
3. **산출물**: 생성한 TCL·EDF·package·실행 계획의 해시, compiler flags, 입출력 ABI, bucket 집합, 외부 weights 참조.
4. **실행 증거**: 준비·적재·실행의 성공 또는 실패, 실제 NPU Task, CPU fallback 정책, 반복 실행·새 입력·새 프로세스 적재 결과.
5. **성능 범위**: 컴파일·첫 로드·warm 실행·전체 요청을 분리한 시간, 전송 포함 여부, 동시성·요청 분포, 메모리 peak, 지연 분포.

정답이 일치하는 것과 NPU에서 계산한 것은 서로 다른 질문이다. 반대로 NPU Task가 보인다고 모델의 의미가 맞는 것도 아니다. 둘을 함께 확인하는 절차는 [5장](05-graphs-fallback.md), 성능 측정의 축은 [7장](07-optimization.md), 이식 실습의 완료 기준은 [13장](13-porting-fha.md)에 연결되어 있다.

## 수정할 때 지켜야 할 규칙

새 버전의 지원 범위를 추가할 때는 기존 설명을 조용히 덮어쓰기보다 **변경된 버전·코드·검증 범위**를 같이 남긴다. 문서와 코드가 다르면 어느 쪽이 어떤 시점을 나타내는지 적는다. 최신 브랜치 링크만 남기면 같은 주장을 나중에 재검토하기 어려우므로 중요한 구현과 시험 결과는 커밋을 고정한다.

공식 풀네임을 확인하지 못한 약어는 확장하지 않는다. eISA에 대한 이 책의 경계가 여기에 해당한다. **EDF는 추가 조사에서 공식 Kernel Validation의 Executable Device Format 표기를 확인하여 갱신했다.** TCL은 Tensor Contraction Language, PERT는 PE Runtime으로 확인했으며 [용어사전](glossary.md)에 명시했다. [EDF 이름의 고정 원문](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/docs/src/quick-start/kernel-validation.md)

## 추가 조사: 다른 NPU와 모델 사전 최적화

[18장](18-npu-ecosystems.md)은 특정 공식 경로를 골라 비교한다. Neuron의 traced inference는 2.31.1 문서, NEFF 적재 구조의 예는 2.28 문서처럼 범위를 명시했다. 한 업체의 모든 최신 기능을 해당 예제로 대표하거나 NPU 간 성능 순위를 추론하지 않는다.

[19장](19-model-optimization.md)은 NetsPresso 제품·공개 Python client·공개 trainer를 구분하고, 라이브러리별 GitHub와 고정 소스를 연결한다. 공개 라이브러리가 만드는 ONNX·양자화 tensor·다른 backend 실행물을 RNGD가 바로 읽는다는 뜻으로 사용하지 않는다. 원본 모델의 의미 보존, 품질 허용 범위, 목표 compiler의 수용, 실제 NPU 실행을 순서대로 대조해야 한다.

홈의 경로 그림은 vISA를 직접 실행하는 것처럼 보이지 않도록 컴파일 산출물과 Image 적재를 추가했다. 검색 색인의 문구가 고정 소스나 현재 문서 본문과 다를 때는 그 문구만으로 내부 pass 순서·새 CLI 옵션을 확정하지 않았다. 구체적인 확인 범위와 교차 호환성 실험 제안은 [08장](08-eisa-visa.md)에 있다.

## 출처

| 자료 | 이 페이지에서 담당하는 근거 |
|---|---|
| [공식 소프트웨어 스택](https://developer.furiosa.ai/v2026.4.0/en/overview/software_stack.html) | SDK 문서 버전 표시·계층 책임·PERT 이름 |
| [2026.5.0b0 변경](https://developer.furiosa.ai/v2026.5.0b4/en/whatsnew/release-2026.5.0b0.html) | beta 기능을 안정판과 구분 |
| [현재 opt launch.rs](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/furiosa-opt-std/src/runtime/launch.rs) · [과거 포럼](https://forums.furiosa.ai/t/furiosa-opt-v0-8-1-launch-host-cpu-cpu-2/479) | 시점이 다른 근거를 대조하는 예 |
| [Python vISA 연결 포럼](https://forums.furiosa.ai/t/furiosa-opt-compile-from-visa-furiosa-torch/487) | 공개 API 확인 범위 |
| [FuryNo5 exporter](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/execution_plan.py) | 코드 검토로 확인한 병렬 프로세스 실행 전제 |
| [전체 출처 색인](data/source-index.json) | 각 장의 개별 인용 URL 목록 |
