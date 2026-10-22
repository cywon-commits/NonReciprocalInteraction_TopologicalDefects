# 비상반 스핀 결합과 위상 구조 (분기 논문 A)

**Nonreciprocal spin couplings and topological structure — topological-charge selection rules and thermal skyrmion-pair statistics.**
Theory notes, verification code, and results for branch paper A of the nonreciprocal spin-coupling project, plus a compute request (stage B) for a coding agent.

Changyeon Won (Department of Physics, Kyung Hee University) · 2026-10

---

## 한눈에 보기

이 저장소는 세 부분으로 되어 있다.

1. **문서** (`docs/`) — 논문 구조안, 계산 의뢰서, 배경이 되는 기초 논문 구조안.
2. **검증 코드와 결과** (`code/`, `results/`) — 구조안 6절 검증 기록의 모든 수치를 재현하는 스크립트와 출력.
3. **의뢰 작업 공간** (`stageB/`) — 계산 의뢰서를 수행할 코드가 들어갈 자리.

핵심 결과는 다음과 같다. 자세한 내용과 수식은 [구조안](docs/branch_A_structure.md)에 있다.

- 비평형에서는 세차가 시간반전 깨짐을 공급하므로, ⟨Q⟩의 선택 규칙은 "평면 방향을 뒤집는 연산이 모두 깨졌는가" 하나로 정해진다(정리 Q1, Q2).
- 균일 δJ에서는 모든 차수에서 ⟨Q⟩ = 0이다. 순 전하는 소용돌이도에서 나오며 κ(T) = κ₀[1 + (8/9)J/T]이다.
- SOC 경우에 δD_z는 유효 자기장 B_eff = −(11/72)δD_z/λ처럼 작용한다.
- 스커미온 정상분포는 흐름 함수의 볼츠만 꼴 e^{−Gψ_e/T}이다. 쌍 탈출 확률은 ln(R₀/a)/[ln(4ℓ_u/a) − γ_E]이고, 자유 밀도는 전류에 무관하다.
- 쌍 상관은 상대 표류의 반대쪽(상류)으로 치우친다. 흡수 핵이 하류에 그림자를 드리우기 때문이다.
- 소용돌이가 있는 흐름은 순 전하 없이 밀도를 ⟨e^{−Gψ/T}⟩배 높이고, 국소적으로 n₊n₋ = n₀²가 성립한다.

## 폴더 구조

```
docs/
  branch_A_structure.md          분기 논문 A 구조안 (이론 + 검증 기록)
  compute_request_stageB.md      계산 의뢰서 (단계 B: 핵생성 편향)
  background/
    00_overall_structure_...md   전체 구조안 (측도 보존, 유효 에너지, 복소 게이지)
    01_foundation_...pdf         기초 이론 구조안 (평형 흉내, 불가능 정리)
code/                            검증 스크립트 (각 폴더에서 실행)
results/                         위 스크립트의 출력 (JSON / JSON Lines)
stageB/                          의뢰 작업 공간
```

## 코드와 문서의 대응

구조안 절과 검증 기록 번호를 기준으로 정리했다. 실행은 해당 폴더 안에서 한다. 스크립트끼리 같은 폴더의 모듈을 불러오기 때문이다.

| 구조안 위치 | 내용 | 스크립트 | 결과 |
| --- | --- | --- | --- |
| 6절 검증 기록 (2단계), 3.3절 | 고온 확률 LLG: 모서리 키랄성의 소용돌이도 응답 | `code/step2_highT/hiT_check.py`, `run_step2.sh` | `results/step2_highT/` |
| 3.5절 | κ(T)의 정확한 2차 계수 (구면 모멘트 전개) | `code/series/hiT_series.py` | `results/series/hiT_kernel.json` |
| 2.7절 | SOC 경우 ⟨q⟩ ∝ δD_z D², 평형 기준 B_z D² | `code/series/soc_series.py`, `eq_BD2.py` | `results/series/` |
| 6절 (가) | Thiele 정상분포 e^{−Gψ/T}: 세포 흐름, 띠 | `code/step3_thiele/thiele_psi.py`, `runA.sh`, `runB.sh` | `results/step3_thiele/` |
| 6절 (나), 5.3절 | 쌍 탈출 확률 | `code/step3_thiele/pair_escape.py`, `runC.sh` | 구조안 표 (표준 출력만 있음) |
| 6절 (다) | 다체 기체: n_f의 전류 무관성 | `code/step3_gas/pair_gas.py`, `runG1.sh`, `runG2.sh` | `results/step3_gas/nf_runs_*.jsonl` |
| 6절 (라), 5.6절 | 쌍 상관 g₊₋(r, θ)의 방향 비대칭 | `code/step3_gas/pair_corr.py`, `runK1.sh`, `runK2.sh` | `results/step3_gas/corr_*.json` |
| 6절 (마), 4.6절 | 소용돌이도에 의한 전하 분리, 교차 길이 ℓ_ν | `code/step3_gas/pair_gas_psi.py`, `runP1–P3.sh` | `results/step3_gas/segregation_runs.jsonl` |
| 6절 (바) | 전체 LLG 선택 규칙 (균일 δJ, 열린 띠) | `code/stageA_llg/llg_general.py`, `runA1.sh`, `runA2.sh` | `results/stageA_llg/` |
| — | 3단계의 독립 교차 검증 (이전 실행본) | `code/crosscheck_step3/` | `results/crosscheck_step3/` |

## 실행

```bash
pip install -r requirements.txt          # numpy, scipy
cd code/series && bash run_series.sh     # 해석 계수 (수 분)
cd code/step3_gas && bash runK1.sh       # 예: 쌍 상관 (CPU 수십 분)
```

- 모든 스크립트는 인자로 JSON 문자열을 받거나(대부분), 위치 인자를 받는다(`hiT_check.py`). 기본값과 의미는 각 파일 머리의 설명에 있다.
- 결과 파일의 한 줄은 계산 하나의 요약이다. 키 이름은 스크립트의 `return` 사전과 같다.
- 실행 시간은 2코어 CPU 기준으로 계산당 수 분에서 15분이다. 단계 B(의뢰서)만 GPU가 필요하다.

## 규약

- 스핀 동역학은 LL 꼴 Ṡ = −S×(h + h^N) − λ S×(S×h)이다. γ = 1이고, 잡음은 Stratonovich Heun으로 적분한다.
- E = −JΣS_i·S_j이다. δJ_ij는 결합 i → j의 비상반 편향이고 δJ_ji = −δJ_ij이다.
- χ_UR = S_i·(S_{i+x}×S_{i+y})이고, 순환은 반시계 방향이다. 이 규약에서 ⟨Q_χ⟩ = +JδJ/(108πλT²)∮v̂·dl이다.
- Thiele 방정식은 G ẑ×(v − u) + D(αv − βu) = ξ이고, u_eff = (β − α)u/α이다.

## 단계 B 의뢰

[계산 의뢰서](docs/compute_request_stageB.md)는 코딩 에이전트가 이 문서만 읽고 작업할 수 있도록 썼다. 모형, 사전 검증 기준값(V1–V4), 측정점(P1–P9), 대조군, 판정 기준이 모두 들어 있다. 결과물은 `stageB/`에 둔다.

수행 결과: 코드·설정·원자료·그림은 [`stageB/`](stageB/README.md), 과제 1 보고서는 [`stageB/REPORT.md`](stageB/REPORT.md)에 있다.

## 참고

- `docs/` 안의 문서는 Claude Docs에서 내보낸 것이다. 원본은 의뢰자의 작업 공간에 있으며, 이 저장소의 사본이 기준이다.
- 라이선스는 아직 정하지 않았다. 공개 전에 `LICENSE` 파일을 추가할 것.
