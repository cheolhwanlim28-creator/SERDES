import numpy as np
import matplotlib.pyplot as plt
from config import Config
from system import SerDesSystem
import blocks as blk

def run_sweep():
    # 1. 환경 설정
    cfg = Config()
    serdes = SerDesSystem(cfg)
    
    print("=== Phase 1 & 2: CDR & DFE Training ===")
    # run_link_up은 내부에서 PRBS 및 Clock 패턴을 생성하여 학습함
    Link_Training = serdes.run_link_up()

    # CDR이 제대로 Lock되었는 지 확인용 (phase_hist가 수렴했는 지)    
    phase_std = np.std(Link_Training['phase_hist'][-500:])
    print(f"[*] CDR Phase Stability (std): {phase_std:.4f} samples")
    if phase_std > 2.0:
        print("[!] Warning: CDR phase is not stable!")


    print(f"[*] CDR Phase Locked at: {serdes.locked_phase:.3f} Samples")
    print(f"[*] DFE Taps Trained: h1={serdes.h1_lock*1000:.1f}mV, h2={serdes.h2_lock*1000:.1f}mV")

    # 2. 결과 시각화 (DFE 수렴 과정)
    h1_h, h2_h = Link_Training['h1_hist'], Link_Training['h2_hist']

    # 1차원 배열로 확실히 변환하고 단위를 mV로 통일
    # 만약 데이터가 너무 많으면 마지막 5000개만 슬라이싱해서 봐도 좋습니다.
    h1_mv = np.array(h1_h).flatten() * 1000
    h2_mv = np.array(h2_h).flatten() * 1000
    plt.figure(figsize=(10, 5))
    plt.plot(h1_mv, label=f'h1 (Final: {h1_mv[-1]:.1f} mV)', color='blue', alpha=0.8)
    plt.plot(h2_mv, label=f'h2 (Final: {h2_mv[-1]:.1f} mV)', color='green', alpha=0.8)

    # 가이드라인: 수렴 목표 지점을 점선으로 표시
    plt.axhline(y=h1_mv[-1], color='blue', linestyle='--', alpha=0.3)
    plt.axhline(y=h2_mv[-1], color='green', linestyle='--', alpha=0.3)

    # Y축 범위를 데이터에 맞춰 자동 최적화 (여백 10% 추가)
    all_data = np.concatenate([h1_mv, h2_mv])
    y_min, y_max = np.min(all_data), np.max(all_data)
    plt.ylim(min(0, y_min) - 5, y_max + 10)
    plt.title("DFE Tap Adaptation Convergence (mV Scale)")
    plt.xlabel("Adaptation Steps (Bits)")
    plt.ylabel("Tap Weight (mV)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

    print("\n=== Phase 3: BER Measurement (Mission Mode) ===")
    # 고정된 위상과 탭으로 실제 데이터 전송 테스트
    test_bits_count = 100000
    np.random.seed(42)  # 분석용으로, seed고정
    prbs_gen = blk.PRBS(cfg.global_cfg.prbs_order)
    bits_test = prbs_gen.generate(test_bits_count)

    # measure_ber 호출 (n_bits 인자는 이미 bits_test에 반영됨)
    BER_measurement = serdes.measure_ber(bits_test, adaptation_on=False)
    
    print(f"[*] Measurement Completed.")
    print(f"[*] Total Bits: {test_bits_count}")
    print(f"[*] Result BER: {BER_measurement['ber']:.2e}")

    serdes.report_noise_contribution()  # Noise Budget Analysis

    # --- Real 모드 Jitter Tracking 점검 ---
    if cfg.rx.cdr.mode == "real":
        plt.figure(figsize=(12, 4))
        
        # 1. TX에서 인가한 실제 지터 (데이터 전체가 아닌 앞부분 500비트만)
        # serdes.last_jitter_samples는 measure_ber 실행 시 업데이트된 지터 배열입니다.
        tx_jit = serdes.last_jitter_samples[:500]
        
        # 2. CDR이 실시간으로 추적한 위상 변화 (phase_hist)
        # measure_ber의 리턴값인 BER_measurement에서 꺼내옵니다.
        cdr_phs = BER_measurement['phase_hist'][:500]
        
        # 시각화를 위해 CDR 위상의 평균을 빼서 TX 지터와 레벨을 맞춥니다.
        plt.plot(tx_jit, 'b-', alpha=0.7, label='Actual TX Jitter (Target)')
        plt.plot(cdr_phs - np.mean(cdr_phs), 'r--', linewidth=2, label='CDR Tracking Phase (Real)')
        
        plt.title(f"Real Mode Tracking Performance (Jitter Amp: {cfg.tx.jitter.psij_amp} UI)")
        plt.xlabel("Time (Bits)")
        plt.ylabel("Phase Offset (Samples)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.show()
    # ------------------------------------------


    # Eye Diagram 시각화
    plt.figure(figsize=(15, 5))
    # 1. 채널 통과 직후 (아날로그 Eye)
    plt.subplot(1, 3, 1)
    blk.plot_eye(
        BER_measurement['rx_ctle_out'], 
        cfg.global_cfg.samples_per_ui, 
        title="Analog Eye (After Channel)",
        sampling_points = BER_measurement['CDR_sampling_points'],
        show_clock = False
    )

    # 2. CTLE 통과 후 아날로그 Eye + 샘플링 지점(Clock) 표시
    plt.subplot(1, 3, 2)
    blk.plot_eye(
        # BER_measurement['rx_ctle_out'],
        BER_measurement['rx_afe_out'],
        # BER_measurement['rx_ctle_in'], 
        cfg.global_cfg.samples_per_ui, 
        title="Analog Eye (After CTLE) - CDR Mode: {cfg.cdr.mode}",
        sampling_points = BER_measurement['CDR_sampling_points'], # 샘플링 지점 전달
        show_clock = True
    )

    # 3. Slicer 통과 후 (디지털 Eye, 전압 레벨 흔들림 확인)
    plt.subplot(1, 3, 3)
    # v_corr은 이미 샘플링된 전압들의 배열이므로 spui=1로 간주하여 시각화 가능
    plt.plot(BER_measurement['v_corr'][:1000], 'ro', markersize=2, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title("Sampled Voltages (After DFE)")
    plt.show()


def run_cdr_comparison():
    cfg = Config()
    serdes = SerDesSystem(cfg)
    
    # 1. 공통 학습 (CDR Lock & DFE Training)
    # 이 과정에서 초기 locked_phase와 h1/h2_lock이 설정됩니다.
    serdes.run_link_up()
    
    modes = ["perfect", "static", "real"]
    results = {}

    print("=== CDR Mode Comparison Start ===")
    
    # 비교의 공정성을 위해 동일한 테스트 비트 시퀀스 생성
    # 결과 신뢰도를 위해 비트 수를 조금 늘려도 좋습니다.
    prbs_gen = blk.PRBS(cfg.global_cfg.prbs_order)
    bits_test = prbs_gen.generate(100000)

    for mode in modes:
        # [중요] Config와 CDR 인스턴스의 모드를 동시에 변경
        cfg.rx.cdr.mode = mode
        serdes.cdr.mode = mode
        
        # CDR 상태 초기화 (이전 모드의 잔상 제거)
        serdes.cdr.phase = serdes.locked_phase
        serdes.cdr.int_err = 0.0
        serdes.cdr.freq = 0.0

        # BER 측정 (동일한 조건에서 모드만 변경)
        report = serdes.measure_ber(bits_test, adaptation_on=False)
        results[mode] = report['ber']
        
        print(f"[*] Mode: {mode:7s} | BER: {results[mode]:.2e} | Shift: {report['shift']}")

    # 2. 막대 그래프(Bar Chart)로 시각화
    plt.figure(figsize=(9, 6))

    # BER이 0일 경우 로그 차트 표시를 위해 최소값(1e-7) 설정
    plot_values = [max(v, 1e-7) for v in results.values()]
    mode_names = list(results.keys())
    
    colors = ['skyblue', 'salmon', 'lightgreen']
    bars = plt.bar(mode_names, plot_values, color=colors, edgecolor='black', alpha=0.8)

    # BER은 로그 스케일로 보는 것이 정석
    plt.yscale('log') 
    plt.ylim(1e-7, 1) # Y축 범위를 10^-7부터 1까지로 고정

    plt.title("BER Comparison by CDR Mode", fontsize=14)
    plt.ylabel("Bit Error Rate (Log Scale)", fontsize=12)
    plt.grid(True, which="both", axis='y', ls="--", alpha=0.5)

    # 막대 위에 숫자 표시
    for bar in bars:
        yval = bar.get_height()
        display_val = f"{yval:.1e}" if yval > 1e-7 else "Zero ( <1e-7 )"
        plt.text(bar.get_x() + bar.get_width()/2, yval, display_val, 
                 va='bottom', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()

    plt.show()

def run_jitter_sweep(sweep_type='rj', start=0.01, end=0.15, steps=8):
    """
    Jitter (현재는 RJ(Random Jitter)) 크기를 변화시키면서 
    CDR 모드별(Perfect, Static, Real) BER 내성을 측정하는 스윕 시뮬레이션
    - sweep_param: 'rj', 'dcd', 'psij_amp' 중 하나 선택
    - start/end: UI rms(RJ) 또는 UI p-p(DCD, SJ) 단위
    """
    cfg = Config()
    serdes = SerDesSystem(cfg)
    prbs_gen = blk.PRBS(cfg.global_cfg.prbs_order)
    static_bits = prbs_gen.generate(50000)
    
    sweep_range = np.linspace(start, end, steps)
    modes = ["perfect", "static", "real"]
    results = {mode: [] for mode in modes}

    print(f"=== Sweep Started: Target Parameter [{sweep_type}] ===")
    for val in sweep_range:
        print(f"\n[*] Testing {sweep_type} = {val:.4f}")
        
        # # [핵심] 1. 환경 설정 변경
        # cfg.tx.jitter.rj = rj
        # 1. Config의 해당 지터 성분 업데이트
        if sweep_type == 'rj': cfg.tx.jitter.rj = val
        elif sweep_type == 'dcd': cfg.tx.jitter.dcd = val
        elif sweep_type == 'psij_amp': cfg.tx.jitter.psij_amp = val
        
        # [핵심] 2. 바뀐 지터가 적용된 새로운 파형을 생성해야 함!
        # serdes 클래스 내에 파형을 다시 만드는 함수가 있다면 여기서 호출하세요.
        # 예: serdes.process_front_end() 또는 serdes.run_link_up()
        serdes.run_link_up() 

        for mode in modes:
            # CDR 모드 강제 변경
            serdes.cdr.mode = mode
            
            # 모드 변경 시 CDR의 내부 상태 리셋
            # 이전 모드에서 쌓인 int_err나 phase가 다음 모드에 영향을 주지 않도록!
            serdes.cdr.phase = serdes.locked_phase #+ 5 * 32 # 학습된 위치에서 시작
            serdes.cdr.int_err = 0.0
            serdes.cdr.freq = 0.0

            # # Perfect 모드일 때만 Sweep 중인 성분을 0으로 Override 하여 참조 클럭 생성
            # # 이렇게 해야 CDR이 해당 지터 성분을 '추적하지 못하는' Ideal한 상황이 연출됨
            # if mode == "perfect":
            #     curr_override = {sweep_type: 0}
            # else:
            #     curr_override = None # Real 모드는 모든 지터를 있는 그대로 보고 추적

            # 3. 새로 생성된 파형 위에서 BER 측정
            # 동일한 static_bits를 넘겨줌
            report = serdes.measure_ber(bits_test=static_bits) #, jitter_override_ref=curr_override)
            results[mode].append(max(report['ber'], 1e-6))

            # 여기서 Found Shift가 몇이 나오든 상관없습니다. 
            # compute_ber가 최적의 Shift를 찾아낸 후의 BER이 진짜 성능입니다.
            print(f"Mode: {mode:7s} | Found Shift: {report['shift']} | BER: {report['ber']:.2e}")

    # 시각화
    plt.figure(figsize=(10, 6))
    
    colors = {'perfect': 'blue', 'static': 'red', 'real': 'green'}
    markers = {'perfect': 'o', 'static': 's', 'real': '^'}
    
    for mode in modes:
        plt.plot(sweep_range, results[mode], 
                 label=f"CDR: {mode}", 
                 color=colors[mode], 
                 marker=markers[mode], 
                 linewidth=2)

    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.title("Jitter vs BER (Jitter Tolerance Sweep)")
    plt.xlabel("Random Jitter (UI rms)")
    plt.ylabel("Bit Error Rate (Log Scale)")
    plt.legend()
    
    # Target BER 라인 (가이드)
    plt.axhline(y=1e-3, color='gray', linestyle='--', alpha=0.5)
    plt.text(0.01, 1.2e-3, 'Target BER (10^-3)', color='gray')

    plt.show()

    blk.plot_eye(
        report['rx_ref_clk'],
        cfg.global_cfg.samples_per_ui, 
        title="Reference Clock"
    )
    plt.show()

# 하나 선택해서 실행, 나머지는 주석 처리
if __name__ == "__main__":
    run_sweep()
    # run_cdr_comparison()
    # run_jitter_sweep()