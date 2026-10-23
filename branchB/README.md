# 분기 B 코드·결과 묶음 — 파일 지도

먼저 읽을 것: `REQUEST.md`(계산 의뢰서), `doc_구조안_분기B.md`(이론·결과 전체, 절 번호는 아래 표와 대응).

환경: Python 3.12+, numpy, scipy, numba, matplotlib. 각 폴더의 스크립트는 그 폴더 안에서 실행하도록 쓰여 있다(상대 경로로 `skx.py`, `fastllg.py`, 상태 파일 `S_B0.5.npy` 등을 찾음). 필요한 상태 파일은 `states/`에서 해당 폴더로 복사할 것.

## 공통 핵심 모듈

| 파일 | 내용 |
|---|---|
| `code/skx/skx.py` (다른 폴더에 같은 사본) | numpy 격자 모형 `Model(Lx,Ly,J,D,B)`, `rhs(S,w1,w2,a,…,k)`(Bloch 위상 포함), `rk4`, `GD`(자이로·소산 텐서), `find_V`(공동이동 정지해+Thiele 속도), `basis`, `bloch_jac`(Bloch 야코비안) |
| `code/kt/fastllg.py` (= `code/hyst/fastllg.py`) | numba Heun 적분기 `Integrator(S,J,D,B,a,w1,w2,T,dt,seed).run(n)`; 확률 LLG(유한 T). numpy 판과 10⁻¹⁵ 일치 |
| `code/kt/orderpar.py`, `melt.py` | 스커미온 중심(`sk_centers`), Delaunay 기반 ψ6·결함 비율, Bragg 위성 피크 |
| `code/helix/helix.py` | 1D 키랄 사슬(블로흐 DMI), 회전 좌표 원뿔 정지해 `predict`, 야코비안 `jac` |

## 단계별 폴더 (구조안 절 번호)

| 폴더 | 절 | 내용 |
|---|---|---|
| `code/helix` | 4.1 | 1D 나선 원뿔 M = s·u_eff/D, Landau 문턱 u_L, 풀림 문턱 |
| `code/skx` | 4.2, 4.4, 4.5, 4.6 | SkX 정지해·쌍극자, λ 검증(중첩 공식 `lam_overlap.py`, `lam_full.py`, 직접 적분 `labsim.py`), κ 지도(`kgrid.py`), 큰 계 용융(`bigsim.py`, `melt*.py`) |
| `code/stripe` | 4.3 | 줄무늬/사이클로이드 비스듬한 전류, 굽힘 불안정 |
| `code/csl` | 4.5 | 1D 키랄 솔리톤 격자 λ 정확식 검증 |
| `code/teff` | 4.7, 4.8 | 움직이는 깁스 앙상블, T_eff(k) 측정(`langevin*.py`, `fitamp.py`, `fitfree.py`), Bloch 감쇠율 |
| `code/kt` | 4.8 | KTHNY 용융점 이동(`run_kt*.py`), T = 0 구동 용융·간헐성(`run_T0.py`, `run_snap.py`), 추적(`run_teff2.py`) |
| `code/b1` | 4.9 | δD± → SOT 분류(`sot.py`: FL·DL 채널), 펌핑 시험 |
| `code/a1` | 4.9–4.12 | κ 정적 정리(`a1.py`, `grid_static.py`), 변형 채널, 두 장 Ritz(`ritz3.py`), 블록 구조·껍질 수렴(`series.py`, `shells.py`) |
| `code/hyst` | 4.13 | 올림/내림 전류 연쇄 `sweep.py`(사용법은 파일 머리말), 감김별 문턱 `thr_n.py`, 실행 셸 `*.sh`, 그림 `fig_hyst.py` |

## 결과

- `results/<폴더>/`: 각 단계의 로그·표·json. 예: `results/a1/grid_static.json`(κ 정확값), `results/hyst/*.log`(연쇄 요약 한 줄/단계), `results/hyst/rec/rec_<tag>_<u>.npy`(열: t, A, Vx, Vy, Q 또는 감김 수, ⟨S_y⟩).
- `results/*.png`: 본문 그림(κ 지도, 용융, T_eff, KTHNY, 간헐성, 이력·계단).
- `states/`: 기준 상태. `S_B0.5.npy` = SkX 기본 세포 19×33(스커미온 2개, B = 0.5D²), `S_B0.3/0.7/0.85.npy` 같은 형식, `Sgrid/Sgrid_<Lx>_<B>.npy` = 격자상수·자기장 격자(κ 지도용), `Scomov_<u>.npy` = 공동이동 정지해.
- 큰 궤적·스냅샷(수백 MB)은 뺐다. 필요하면 해당 스크립트로 재생성.

## 규약 요약

u_eff = (β/α − 1)u; 실험실 좌표 w1 = (−u,0), w2 = (βu,0); 기본 매개변수 J = 1, D = 0.4, B = 0.5D², α = 0.3, β = 0.6. 자세한 주의는 `REQUEST.md` §2.
