# 분산 GPU Greedy Minimum Set Cover 설계 정리

OPC segment dedup 파이프라인을 위한 분산 greedy minimum set cover 알고리즘 설계 문서. MPI 기반 SPMD CPU 구현에서 multi-GPU 확장까지, 그리고 hash universe 크기 증가에 대한 scalability를 다룬다.

---

## 1. 문제 정의

### 1.1 Greedy Minimum Set Cover

- **Element (원소)**: 덮어야 하는 대상. 본 파이프라인에서는 uint64 hash 값.
- **Patch (집합)**: element들의 부분집합. "이 patch가 이러이러한 element를 덮는다."
- **목표**: 최소 개수의 patch로 모든 element를 덮기.

매 iteration의 동작:

1. 아직 덮이지 않은 element를 가장 많이 덮는 patch 선택 (argmax)
2. 선택된 patch를 solution에 추가
3. covered set 업데이트
4. 모든 element가 덮이면 종료

### 1.2 규모 가정

| 항목 | 기호 | 가정값 |
|---|---|---|
| 전체 distinct element (universe) | N | 최대 100억 (10^10), 이후 더 큰 경우도 고려 |
| 전체 patch 수 | M | 10^6 ~ 10^7 |
| Patch당 평균 element 수 | K | 100 ~ 1000 |
| Process(rank) 수 | P | 환경에 따라 (V100/A100/B300 클러스터) |
| 가용 RAM | - | 노드당 2 TB |

---

## 2. 자료구조: Bitset 기반 Sparse/Dense 하이브리드

### 2.1 표현 방식 비교

같은 "patch가 어떤 element를 덮는지" 정보를 세 가지로 표현할 수 있다.

| 표현 | 메모리 | 연산 속도 | 비고 |
|---|---|---|---|
| List (element ID 나열) | 작음 (실제 원소 수 비례) | 느림 (lookup 필요) | 단순 |
| Dense bitset | 큼 (N 비례) | 가장 빠름 (popcount/AND) | N이 작을 때만 |
| Sparse bitset (Roaring 등) | 중간 | 빠름 | 구현 복잡 |

핵심은 두 set의 교집합/합집합을 bitwise 연산으로 한 번에 처리할 수 있다는 점이다. Score 계산의 hot path는 "patch가 아직 안 덮인 element를 몇 개 새로 덮는가"이며, bitset이면 `popcount(patch AND NOT covered)`로 word 단위 처리가 가능하다.

### 2.2 핵심 결론: Patch는 sparse, Covered는 dense

- **Patch는 sparse**: 각 patch는 universe의 극히 일부만 덮으므로 dense bitset(patch당 N/8 bytes)은 메모리 폭발. 정렬된 ID 리스트(sparse)로 표현.
- **Covered는 dense bitset**: iteration이 진행되며 점점 채워지므로 dense가 유리. N=100억이면 1.25 GB로 GPU 메모리에 수용 가능.

### 2.3 Hash → ID 변환의 필요성

원본은 uint64 hash 값(범위 [0, 2^64))이라 그대로는 bitset 인덱스로 사용 불가능 (2^64 bits = 2 EB).

해결: universe(모든 patch의 hash 합집합, N개)에만 [0, N) 정수 ID를 부여한다.

```
정렬된 hash:   [0x4a08..., 0x55de..., 0x7c91..., 0xa3f2..., ...]
부여된 ID:     [    0,        1,        2,         3,       ...]
```

ID로 변환하면 dense bitset 인덱싱이 가능해진다 (`covered[id >> 6] |= 1ULL << (id & 63)`). **모든 알고리즘 hot path는 ID 위에서 동작하며, 원본 hash는 setup 이후 등장하지 않는다.**

ID 타입:
- N < 2^32 (≈ 42.9억): `uint32_t` 가능 (메모리 절반)
- N ≥ 2^32: `uint64_t` 필요. 100억 가정 시 uint64 사용.

---

## 3. CSR (Compressed Sparse Row) 레이아웃

### 3.1 동기

`std::vector<std::vector<uint64_t>>`는 nested heap allocation이라 GPU에 직접 올릴 수 없고 메모리가 흩어진다. CSR은 가변 길이 항목들을 평탄한 배열 두 개로 표현한다.

### 3.2 구조

```
patches[0] = {0, 2}          patch_data    = [0, 2, 1, 2, 3, 0, 4, 3, 4]
patches[1] = {1, 2, 3}                        └p0─┘ └─p1──┘ └p2─┘ └p3─┘
patches[2] = {0, 4}          patch_offsets = [0,    2,       5,    7,    9]
patches[3] = {3, 4}
```

- `patch_data`: 모든 patch 내용을 순서대로 연결 (각 patch 내부는 정렬)
- `patch_offsets[p]`: patch p가 patch_data에서 시작하는 위치
- `patch p의 element들 = patch_data[patch_offsets[p] .. patch_offsets[p+1])`
- `patch p의 크기 = patch_offsets[p+1] - patch_offsets[p]`
- `patch_offsets.size() = M_local + 1`

**중요**: `patch_data`에 저장되는 값은 정렬된 순서의 **ID** (정렬된 hash 배열에서의 index)이며 원본 hash 값이 아니다.

### 3.3 Offset 타입 주의

`patch_offsets`는 patch_data의 인덱스를 가리키므로, patch_data 크기가 2^32를 넘으면 `uint64_t` 필수. 안전을 위해 항상 uint64 사용 권장.

### 3.4 CSR의 GPU 친화성

1. **Contiguous memory**: 단일 `cudaMemcpy`로 전송
2. **Coalesced access**: `patch_data[start + threadIdx.x]` 패턴이 자연스럽게 coalesced
3. **표준 형식**: cuSPARSE 등 GPU sparse 라이브러리 호환

---

## 4. Inverted Index (Hash → Patches 매핑)

### 4.1 동기: 불필요한 score 재계산 회피

매 iteration마다 모든 patch의 score를 재계산하는 것은 낭비. Patch가 선택되면 **새로 covered된 element를 공유하는 patch의 score만 변한다.**

Inverted index는 "patch → elements"를 뒤집은 "element → patches" 매핑이다.

```
element_to_patches[e] = {e를 덮는 patch들}
```

### 4.2 알고리즘

```
1. p* = argmax(score)
2. solution.append(p*)
3. newly_covered = patch[p*] AND NOT covered;  covered |= patch[p*]
4. affected = ∅
   for e in newly_covered:
       affected |= element_to_patches[e]
5. for q in affected:
       delta = popcount(patch[q] AND newly_covered)
       score[q] -= delta
```

Step 5에서 업데이트되는 patch는 전체 M이 아니라 affected만. Sparse instance에서 이득이 극적이다.

### 4.3 Top-k vs Inverted Index

- **Top-k (lazy evaluation)**: 근사. 상위 후보만 재계산.
- **Inverted index**: 정확. 영향받은 patch를 정확히 식별.

정확성이 중요하고 sparse structure가 있는 경우 inverted index가 우월하다.

### 4.4 Inverted Index도 CSR로

GPU에서 `unordered_map`은 부적합. Patch CSR을 transpose하여 inverted CSR로 빌드한다.

```
inv_keys    : 이 rank patch들에 등장하는 unique element ID (정렬)
inv_offsets : 각 element가 시작하는 위치
inv_data    : patch ID들
```

**Erase 처리**: CSR은 read-only라 erase 불가. 이미 covered된 element를 다시 lookup해도 `popcount(patch ∩ newly_covered) = 0`이라 결과에 영향 없음. Erase 없이 진행 가능 (affected set이 약간 커질 뿐).

---

## 5. CPU/GPU 알고리즘 흐름

### 5.1 Setup 단계 (1회)

1. 각 rank가 local patch들의 hash 추출
2. **Distributed sort + unique** (sample sort): hash 값 기준으로 일시적 재분할
3. **Global ID 부여** (`MPI_Exscan` prefix sum): 각 rank가 자기 hash range에 ID 할당
4. **매핑 회수** (`MPI_Alltoallv` 역방향): 각 rank가 자기 patch hash들의 ID 받아옴
5. Patch CSR 빌드 + Inverted index 빌드
6. 초기 상태: covered=0, scores=patch 크기

### 5.2 두 종류의 분할 (중요)

| 분할 | 성격 | 기준 | 용도 |
|---|---|---|---|
| **Patch 분할** | 영구적 | patch ID | 알고리즘의 메인 분산 구조 |
| **Hash 분할** | 일시적 (setup만) | hash 값 | ID 매핑 빌드용 보조 도구 |

Setup 동안 patch 분할 → hash 분할 → patch 분할로 두 번 transition이 일어난다. ID 매핑이 완성되면 hash 분할 관련 자료구조는 모두 폐기.

### 5.3 메인 루프 (iteration당)

| Step | 연산 | 위치 | 통신 |
|---|---|---|---|
| A | Local argmax | local (CPU loop / CUB ArgMax) | 없음 |
| B | Global argmax | MPI MAXLOC AllReduce | 8 bytes |
| C | Winner의 newly_covered 빌드 + broadcast | winner only → all | ~N/8 bytes |
| D | covered 업데이트 + 종료 체크 | local (모든 rank 동일) | 없음 |
| E1 | Inverted index로 affected 마킹 | local | 없음 |
| E2 | Affected patch의 score 업데이트 | local | 없음 |
| F | Winner 비활성화 (score = -1) | winner only | 없음 |

**iteration당 통신은 두 번뿐**: 작은 MAXLOC AllReduce + 큰 newly_covered broadcast.

### 5.4 MPI_MAXLOC 패턴

```cpp
struct { int score; int rank; } local{local_best_score, my_rank}, global;
MPI_Allreduce(&local, &global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
int winner_rank = global.rank;
```

`MPI_MAXLOC`은 (값, 인덱스) 쌍에서 최댓값과 그 위치를 한 번의 collective로 반환한다. 값이 같으면 작은 인덱스를 선택(tie-breaking)하므로 reproducibility가 보장된다. 값 순서는 (value, index)이며 value가 먼저.

---

## 6. GPU 최적화

### 6.1 자료구조별 GPU 적합성

| 자료구조 | 적합도 | 변경 사항 |
|---|---|---|
| `vector<vector<uint64_t>>` patches | 높음 | CSR (data + offsets)로 변환 |
| `covered` dense bitset | 매우 높음 | uint64 array 그대로 |
| `unordered_map` inverted index | 낮음 | CSR transpose 또는 cuCollections |
| Priority queue / bucket queue | 낮음 | 단순 argmax (CUB)로 대체 |
| scores | 매우 높음 | int device array |

### 6.2 핵심 GPU 결정

1. **Priority queue 제거**: PQ는 GPU에서 다루기 어렵다. 매 iteration `cub::DeviceReduce::ArgMax`로 score array 전체 scan이 오히려 빠르다 (memory bandwidth-bound, M_local=10^6이면 수십 µs).

2. **Score kernel (CSR 순회)**:
```cuda
__global__ void update_scores(const uint64_t* patch_data,
    const uint64_t* patch_offsets, const uint64_t* newly_covered,
    int* scores, uint64_t M_local)
{
    uint64_t p = blockIdx.x;
    if (p >= M_local || scores[p] <= 0) return;
    uint64_t start = patch_offsets[p], end = patch_offsets[p+1];
    int delta = 0;
    for (uint64_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        uint64_t id = patch_data[i];
        if ((newly_covered[id >> 6] >> (id & 63)) & 1) delta++;
    }
    delta = block_reduce_sum(delta);
    if (threadIdx.x == 0 && delta > 0) scores[p] -= delta;
}
```

3. **통신**: MAXLOC은 MPI(작은 값), newly_covered broadcast는 NCCL(GPU-resident, NVLink 활용).

### 6.3 전략 A vs B (Score update)

- **전략 A (inverted index)**: affected만 정확히 update. 후반 iteration에 유리.
- **전략 B (전체 patch 순회)**: 단순, GPU sequential scan에 친화적. 초반 iteration에 유리.
- **Hybrid**: newly_covered popcount를 기준으로 매 iteration 동적 선택.

### 6.4 예상 성능

A100 단일 노드, M_local=10^6, K=1000 기준 iteration당 (대략):

| 단계 | 시간 |
|---|---|
| Argmax | ~100 µs |
| newly_covered build | ~0.5 ms |
| NCCL Bcast (1.25 GB) | ~10 ms |
| Covered OR | ~1 ms |
| Score update | ~10 ms |

iteration당 ~25 ms. S=1000이면 ~25초. CPU 대비 약 10배 가속 추정. **NCCL broadcast가 dominant cost.**

---

## 7. Scalability (N이 더 커지는 경우)

### 7.1 N 의존성 점검

N에 직접 비례해 모든 rank가 부담하는 것: `covered` + `newly_covered` bitset (합 N/4 bytes/rank).

| N | bitset/rank | 평가 |
|---|---|---|
| 10^10 | 2.5 GB | 여유 |
| 10^11 | 25 GB | GPU 부담 시작 |
| 10^12 | 250 GB | 단일 노드 한계 |
| 10^13 | 2.5 TB | 불가능 |

또한 매 iteration broadcast가 N/8 bytes로 통신이 먼저 한계에 도달한다.

### 7.2 Step 1: Sparse Newly_Covered Broadcast (즉시 적용)

매 iteration 추가되는 element 수 = winner score. 후반엔 매우 작아진다.

- Dense bitset 대신 newly_covered를 ID 리스트(sparse)로 broadcast
- 크기 = winner_score × 8 bytes
- `score × 8 < N/8` (즉 `score < N/64`)일 때 sparse 유리 → 거의 모든 iteration

Adaptive 선택:
```cpp
if (winner_score * 8 < num_words) bcast_sparse(ids);
else bcast_dense(bitset);
```

**효과**: 통신이 N에 거의 무관해진다 (winner score에만 비례). 코드 변경 최소로 N=10^11~10^12까지 통신 측면 확보 (메모리는 별개).

### 7.3 Step 2: Element-Partitioned Covered (메모리 scaling)

Covered bitset 자체를 분할:

```
Element ID 공간 [0, N) → P개 element shard
Rank r은 ID range [r·N/P, (r+1)·N/P)의 covered만 보유
```

- Rank당 covered 크기: N/(8P) → P를 늘리면 rank당 메모리 일정 유지
- Score 계산이 partial score → AllReduce로 변경

흐름:
```
Step A: 각 rank가 자기 shard 부분의 partial score 계산
Step B: M개 partial score AllReduce (M × 4 bytes) → full scores
Step C: argmax (모든 rank가 full scores 보유)
Step D~E: 각 rank가 자기 shard의 covered_local만 업데이트
Step F: 새로 covered된 수 AllReduce SUM (8 bytes)
```

**핵심 변화**: dense bitset broadcast가 사라지고 통신이 N 의존성 → M 의존성으로 바뀐다. M이 N보다 훨씬 작으므로 큰 이득. Inverted index도 shard별로 분할하여 affected만 sparse하게 처리.

### 7.4 Step 3: 2D 분할 (극단적 경우)

Patch도 분할 (element shard P_e × patch shard P_p). Fully scalable이나 통신 패턴이 복잡해 구현 비용이 크다. 일반적으로 불필요.

### 7.5 한계 분석 (Element-partition + patch replicate)

| 자료구조 | per-rank 메모리 | 한계 |
|---|---|---|
| covered_local | N/(8P) | P 늘리면 무한 |
| all_patches (replicate) | M × K × 8 | M × K가 노드 메모리 안에 |
| inv_local | total_K / P | P 늘리면 작아짐 |

**진짜 한계는 patch replicate**: M × K × 8 bytes가 노드 RAM(2 TB) 한계. M × K ≈ 2.5×10^11 근처에서 한계. 그 이상은 2D 분할 필요. 단, 보통 M ≪ N이라 대부분의 경우 patch replicate로 충분.

---

## 8. 권장 진행 순서

1. **CPU 버전 구현 + 검증**: CSR 레이아웃, inverted index, MAXLOC + dense broadcast. 정확성 확인.
2. **Sparse broadcast 적용** (Step 1): 통신 N 의존성 제거. N=10^11~10^12까지 통신 확보.
3. **GPU 이식**: CSR은 mechanical translation. PQ → CUB argmax, MPI → NCCL.
4. **메모리가 한계일 때 element-partition** (Step 2): covered 분할, partial score AllReduce.
5. **2D 분할은 마지막 수단** (Step 3): M × K가 노드 메모리를 넘을 때만.

---

## 9. 미해결 / 측정 필요 항목

알고리즘을 정밀화하려면 다음 실측값이 필요하다:

- **N의 실제 값**: dedup pipeline의 distinct hash 수 (가정 100억은 상한선이지 측정값 아님). Raw input vs unique 비율이 architecture 선택을 좌우.
- **M (patch 수)와 K (patch당 평균 element)**: patch replicate 가능 여부 결정.
- **M × K 곱**: 2D 분할 필요성 판단.
- **가용 노드/GPU 수 (P)**: element-partition scaling 여력.
- **Setup 시간 허용치**: distributed sort가 가장 무거운 단계.

---

## 부록: 핵심 자료구조 요약 (100억 universe, GPU)

```cpp
struct GpuLocalState {
    // === Setup 후 read-only ===
    uint64_t N_total;                    // 100억
    uint64_t M_local;                    // 이 rank의 patch 수

    uint64_t* d_patch_data;              // sorted uint64 ID concatenated (CSR)
    uint64_t* d_patch_offsets;           // M_local + 1
    uint64_t* d_patch_global_ids;        // M_local (winner 결정용)

    // Inverted index CSR
    uint64_t* d_inv_keys;                // unique element IDs (정렬)
    uint64_t* d_inv_offsets;
    uint64_t* d_inv_data;                // patch IDs

    // === Read-write ===
    uint64_t* d_covered;                 // N_total/64 words = 1.25 GB
    uint64_t* d_newly_covered;           // scratch
    int* d_scores;                       // M_local

    // === Host-side ===
    std::vector<uint64_t> solution;      // 선택된 global patch IDs
    uint64_t covered_count;              // incremental popcount
};
```
