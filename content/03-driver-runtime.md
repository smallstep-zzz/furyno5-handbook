# 드라이버·컴파일러·런타임: 누가 무엇을 책임지는가

NPU(Neural Processing Unit)를 쓰는 프로그램은 장치 파일에 모델을 한 번 쓰고 끝나지 않는다. 컴파일러는 계산을 장치용 프로그램으로 바꾸고, 런타임은 프로그램과 데이터를 준비해 제출하고, 드라이버는 운영체제와 장치 사이의 접근을 관리한다. 장치에서는 펌웨어와 PE Runtime이 실행을 진행한다. **계산을 만드는 단계와 계산을 실제로 제출하는 단계를 분리해서 보아야 한다.** [공식 소프트웨어 스택](https://developer.furiosa.ai/v2026.4.0/en/overview/software_stack.html)

> **버전 표기** · SDK(Software Development Kit) 2026.4 계열 문서와 FuryNo5가 사용하는 2026.4 API를 기준으로 한다. 공식 `/v2026.4.0/` URL도 조사일에는 문서 헤더가 2026.4.1로 표시됐다. 설치한 패키지의 `--help`와 버전을 재현 기록에 함께 남긴다. Rust 직접 런타임은 별도 `furiosa-opt` commit `3d43bc9` 기준이다.

## 1. 층별 책임

| 층 | 입력과 출력 | 대표적인 책임 | 이 층에 맡기면 안 되는 추측 |
|---|---|---|---|
| 모델·서비스 | 사용자 입력 → 모델 호출 | 전처리, 요청 정책, 결과 해석 | 모델 구조를 모르는 드라이버가 대신 추론한다고 생각하기 |
| 컴파일러 | 그래프·TCL → 실행 산출물 | 분해, fusion, 배치, 메모리·명령 스케줄 | 컴파일 성공을 수치 정확성 증명으로 보기 |
| 호스트 런타임 | 실행 산출물·버퍼 → 제출·완료 | 로딩, 자원 배치, 입출력 바인딩, 수명·오류 관리 | 실행 산출물이 없으면 모든 모델을 자동 CPU 실행한다고 보기 |
| 커널 드라이버 | 장치 노드·시스템 호출 | 장치 발견·권한·매핑·전송 인터페이스 | Python 레이어나 토크나이저를 처리한다고 보기 |
| 펌웨어·PERT(PE Runtime) | 장치 명령·메시지 | PE(Processing Element) 실행·자원·호스트 통신 | 호스트의 서비스 스케줄러와 동일하다고 보기 |

공식 스택 문서는 이 책임을 개념적으로 설명한다. `furiosa-llm` 네이티브 구현의 모든 내부 함수가 공개되어 있지는 않으므로, 아래에서 **공개 소스로 따라갈 수 있는 Rust 경로**와 **공식 문서로 확인하는 SDK 경로**를 구분한다. [공식 스택](https://developer.furiosa.ai/v2026.4.0/en/overview/software_stack.html)

```mermaid
flowchart TB
    M["모델·TCL"] --> C["컴파일러"]
    C --> A["실행 산출물"]
    A --> R["호스트 런타임"]
    I["입력·가중치·상태"] --> R
    R --> D["Linux 장치 드라이버"]
    D --> F["펌웨어·PE 실행 계층"]
    F --> N["Tensor Unit·DMA"]
    N --> R
```

## 2. 드라이버가 실제로 노출하는 것

`furiosa-opt-rt/src/driver/kernel/mod.rs`에서 장치 노드 기본 경로 `/dev/rngd`와 ioctl(Input/Output Control) payload를 확인할 수 있다. `PeWindow`, `DmaMap`, `GroupConfig`는 PE 주소 범위, DMA 등록, 여러 장치의 연결 정보를 전달하는 내부 구조체다. 이 파일은 **해당 런타임 구현의 내부 인터페이스**이며 독립적으로 안정성이 보장되는 사용자 SDK라고 선언하지 않는다. [커널 인터페이스 정의](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/furiosa-opt-rt/src/driver/kernel/mod.rs)

아래는 소스의 중요한 선언 두 줄이다. ioctl 번호만 복사해 새 드라이버 호출기를 만드는 예제는 아니다.

```rust
nix::ioctl_readwrite!(set_window, b'N', 0x03, PeWindow);
nix::ioctl_readwrite!(dma_map, b'N', 1, DmaMap);
```

실제 호출에는 장치 파일 descriptor, ABI(Application Binary Interface)와 일치하는 구조체, 소유권·예약 상태, 버전 호환성이 모두 필요하다. `driver/chip.rs`는 장치 노드를 열고 예약·물리 식별·peer window를 준비한다. `driver/kernel/map.rs`는 BAR(Base Address Register) 영역 등의 메모리 매핑을 맡는다. [칩 개방과 예약](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/furiosa-opt-rt/src/driver/chip.rs), [mmap 구현](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/furiosa-opt-rt/src/driver/kernel/map.rs)

## 3. 프로그램에서 장치를 확인하는 공개 방법

관찰 목적이라면 먼저 SMI(System Management Interface)를 사용한다. SDK 2026.4에서는 Python 패키지가 `furiosa-smi-py`에서 `furiosa-smi`로 바뀌었다. 초기화 함수를 가정하지 않고 현재 문서의 `furiosa_smi` API를 사용한다. [SMI Python API](https://developer.furiosa.ai/v2026.4.0/en/device_management/system_management_interface/furiosa_smi_lib.html)

```python
import furiosa_smi

driver = furiosa_smi.driver_info()
print({"driver": str(driver)})
for item in furiosa_smi.list_devices():
    print({
        "device": item.device_name(),
        "firmware": str(item.firmware_version()),
        "power_w": item.power_consumption(),
    })
```

재현 기록에는 Python 패키지와 장치 소프트웨어의 버전을 따로 남긴다.

```python
from importlib.metadata import version, PackageNotFoundError

for name in ("torch", "furiosa-torch", "furiosa-tcc", "furiosa-llm", "furiosa-smi"):
    try:
        print(name, version(name))
    except PackageNotFoundError:
        print(name, "not installed")
```

```bash
furiosa-smi version
furiosa-smi info --format full
furiosa-tcc --version
```

패키지를 설치했다고 장치 노드·드라이버·펌웨어가 자동으로 모두 호환된다는 뜻은 아니다. 설치·업그레이드는 공식 prerequisite와 release note의 조합에 맞춘다. 예를 들어 2026.4 공식 공지는 대응 RNGD 드라이버 업데이트를 요구한다. [2026.4 발표](https://forums.furiosa.ai/t/furiosa-sdk-2026-4-0-release/489), [설치 전제조건](https://developer.furiosa.ai/v2026.4.0/en/get_started/prerequisites.html)

## 4. 공개 Rust 런타임에서 따라갈 수 있는 호출 경로

| 소스 파일 | 핵심 역할 | 읽을 때 확인할 것 |
|---|---|---|
| `furiosa-opt-rt/src/device/mod.rs` | `Device::new`, 할당, 전송, 동기화 | 어떤 장치를 열고 누가 메모리를 소유하는가 |
| `furiosa-opt-rt/src/driver/chip.rs` | 노드 개방, 예약, peer 연결 | 권한·다중 장치·PDMA 지원 확인 |
| `furiosa-opt-rt/src/registry/load/mod.rs` | 컴파일 이미지 로딩 | 이미지의 ABI와 배치 조건 |
| `furiosa-opt-rt/src/function/mod.rs` | 함수 제출과 완료 | 인수·출력 수명, 실패 전파 |
| `furiosa-opt-rt/src/driver/mailbox.rs` | 요청과 응답 연결 | 어느 제출의 응답인지 식별 |
| `furiosa-opt-rt/src/driver/link.rs` | ring과 doorbell | 호스트·장치 메시지 이동 |

여기서 mailbox는 요청을 응답과 매칭하는 계층, ring은 순환 큐, doorbell은 상대에게 새 일이 있음을 알리는 장치 통지 수단이다. `furiosa-opt-rt` README는 이 계층과 `furiosa-opt-ipc`, `furiosa-opt-abi`의 역할을 공개한다. [런타임 계층표](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/furiosa-opt-rt/README.md)

이 경로는 **실행할 이미지가 이미 있는 경우의 경로**다. `registry::load`에 임의 모델 파일을 넣으면 컴파일까지 해 주는 구조가 아니다. 고수준의 `furiosa-opt-std::launch`를 쓰면 아래 런타임을 직접 다룰 필요가 줄지만, 병목을 분석할 때는 이 층들이 중요해진다.

## 5. 컴파일러와 런타임은 언제 비용을 지불하는가

| 시점 | 발생할 수 있는 작업 | 재사용할 수 있는 대상 |
|---|---|---|
| 개발·빌드 | 그래프 캡처, lowering, 커널 컴파일 | 컴파일된 커널·일치하는 메타데이터 |
| 프로세스 시작 | 호환성 확인, 이미지 로딩, 가중치 준비 | 같은 프로세스의 resident 객체 |
| 새 형상 선택 | 버킷 선택, 입력 패딩, 바인딩 | 사전 컴파일된 같은 버킷 |
| 매 요청 | 입력 전송, 제출, 결과·상태 처리 | 고정 가중치·버퍼·실행 준비 |

이 표는 비용을 분석하는 분류다. 특정 SDK가 모든 항목을 반드시 매 요청마다 수행한다는 주장은 아니다. 프로파일에서 어느 구간에 실제 시간이 쓰였는지 기록해야 한다.

`furiosa-opt`는 컴파일·최적화 도구이며 `furiosa-opt-rt`는 실행 런타임이다. SDK의 `furiosa.torch.ir.Runnable`과 이름이 비슷한 개념을 갖더라도, **두 Python/Rust 객체가 같은 함수나 동일 바이너리 ABI를 쓴다고 볼 공개 근거는 없다.** 각각의 loader·image·tensor binding 계약을 따라야 한다. [Kernel Optimizer](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/docs/src/tools/kernel-optimizer.md), [SDK Runnable 실제 사용](https://github.com/smallstep-zzz/FuryNo5/blob/bba9efea8201aec48150514a35d0b637d29d107e/tplmachine/direct_runtime.py)

## 6. Furiosa-LLM은 이 위에서 무엇을 더 하는가

Furiosa-LLM(Large Language Model inference engine)은 요청을 묶고, prefill과 decode를 진행하고, KV(Key–Value) cache를 관리하고, 토큰을 선택하고, API 응답을 만든다. 장치가 계산하는 컴파일 커널과, 다음에 어떤 요청을 실행할지 결정하는 서비스 정책이 함께 필요하다. [Furiosa-LLM 개요](https://developer.furiosa.ai/v2026.4.0/en/overview/software_stack.html)

공식 FXB(Furiosa Executable Bundle)는 compiled kernels를 제공하고, 모델 디렉터리는 설정·토크나이저·가중치를 제공한다. `--fxb`는 모델 전체를 대체하는 인수가 아니다. [공식 FXB 사용](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/fxb.html)

```bash
# 호환되는 공식 FXB가 준비된 SDK 환경에서 실행한다.
furiosa-llm serve Qwen/Qwen3-8B-FP8 --fxb ./qwen3-8b-fp8.fxb
```

```python
from furiosa_llm import LLM

engine = LLM("Qwen/Qwen3-8B-FP8", fxb="./qwen3-8b-fp8.fxb")
```

여기에서 드라이버로 이어지는 관계는 공식 스택의 역할 설명까지 확인된다. **FuryNo5의 임의 EDF와 임의 실행계획을 등록하는 공개 Furiosa-LLM 확장 ABI**는 이 근거만으로 확인되지 않는다. 공식 FXB를 받을 수 있다는 사실과 모든 독자 포맷을 받을 수 있다는 주장은 다르다. 이 제약 때문에 fxbX의 전용 실행 경로와 `furiosa-opt-vllm`의 별도 엔진 경로를 뒤에서 다룬다.

## 7. 오류를 층으로 나눠 읽는다

컴파일 거절은 모델 의미·자료형·배치·lowering 문제일 수 있다. 장치 열기 실패는 드라이버·권한·예약·호환성 문제일 수 있다. 메모리 할당 실패는 현재 실행 범위의 용량 문제일 수 있다. 실행 후 수치 불일치는 입력 계약·정밀도·커널 의미까지 점검해야 한다. 서로 다른 오류를 `NPU 미지원` 하나로 묶으면 수정 위치를 잃는다.

특히 공개 runtime은 장치 기능이 부족한 경우 명시적인 오류를 낸다. README는 firmware DMA에 필요한 PDMA 채널 예약이 지원되지 않는 드라이버에서는 해당 launch가 `Error::Unsupported`로 실패할 수 있다고 설명한다. 이는 CPU 연산 폴백이 아니다. [드라이버와 DMA 조건](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/furiosa-opt-rt/README.md)

## 출처

| 근거 | 확인한 내용 |
|---|---|
| [공식 소프트웨어 스택](https://developer.furiosa.ai/v2026.4.0/en/overview/software_stack.html) | 컴파일러·런타임·드라이버·PERT·LLM 역할 |
| [2026.4 release 발표](https://forums.furiosa.ai/t/furiosa-sdk-2026-4-0-release/489) | 드라이버·FXB·SMI 변경 |
| [SMI Python API](https://developer.furiosa.ai/v2026.4.0/en/device_management/system_management_interface/furiosa_smi_lib.html) | 공개 장치 관찰 API |
| [커널 인터페이스 소스](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/furiosa-opt-rt/src/driver/kernel/mod.rs) | 노드·ioctl 구조 |
| [Runtime README](https://github.com/furiosa-ai/furiosa-opt/blob/3d43bc9d20f324af60e3006c3f1541938d7f5097/furiosa-opt-rt/README.md) | 공개 실행 계층·DMA 조건 |
| [공식 FXB 문서](https://developer.furiosa.ai/v2026.4.0/en/furiosa_llm/fxb.html) | LLM의 공식 번들 전달 |
