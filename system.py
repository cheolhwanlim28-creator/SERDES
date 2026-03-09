import numpy as np
import blocks as blk
import matplotlib.pyplot as plt

class SerDesSystem:    # 하드웨어 상태값은 인스턴스 변수로, 분석 데이터는 딕셔너리로(return)
    def __init__(self, cfg):
        # 1. Config 저장 (Hierarchy: cfg.global_cfg, cfg.tx, cfg.rx, cfg.channel)
        self.cfg = cfg
        self.g_cfg = cfg.global_cfg
        
        # TX clock path
        self.refclk = blk.ReferenceClock(cfg.txclock.refclk, self.g_cfg)
        self.pll = blk.PLL(cfg.txclock.pll, self.g_cfg)
        self.cd = blk.ClockDistribution(cfg.txclock.cd, self.g_cfg) 
        self.dcc = blk.DCC(cfg.txclock.dcc, self.g_cfg)            

        # 2. TX Blocks 초기화
        self.tx_ffe = blk.FFE(cfg.tx.ffe)
        self.tx_predriver = blk.TXPreDriver(cfg.tx.predriver, self.g_cfg)
        self.tx_driver = blk.TXDriver(cfg.tx.driver, self.g_cfg)
        
        # 3. Channel & RX Blocks 초기화
        self.channel = blk.Channel(cfg.channel, self.g_cfg)
        self.ctle = blk.CTLE(cfg.rx.frontend.ctle)
        self.agc = blk.AGC(cfg.rx.frontend)
        self.cdr = blk.CDR(cfg.rx.cdr, self.g_cfg)
        self.dfe = blk.DFE(cfg.rx.dfe)
        
        # 4. State Variables
        # Training 결과 저장용 변수
        self.locked_phase = 0.5 * self.g_cfg.samples_per_ui # 초기값은 중앙
        self.h1_lock = 0.0
        self.h2_lock = 0.0
        # 분석용 데이터 저장소 (마지막 시뮬레이션 결과 보관용)
        self.last_run_data = {}
        self.last_jitter_samples = None # 배열 형태로 저장하기 위해 초기화
        self.rx_ref_clk = {}

    # ============================================================
    # ======== TX Clock Path ===========
    # ============================================================
    def run_tx_clock(self, n_samples):
        fs = self.g_cfg.fs
        t = np.arange(n_samples) / fs

        # global supply ripple
        vdd = (
            self.g_cfg.psij_amp / 2 *
            np.sin(2 * np.pi * self.g_cfg.psij_freq * t)
        )

        # phase arrays
        phase_ref = np.zeros(n_samples)
        phase_pll = np.zeros(n_samples)
        phase_cd = np.zeros(n_samples)
        phase_dcc = np.zeros(n_samples)

        for i in range(n_samples):

            # RefClk
            phase_ref[i] = self.refclk.step(supply=vdd[i])

            # PLL
            phase_pll[i] = self.pll.step(phase_ref[i], supply=vdd[i])

            # Clock Distribution
            phase_cd[i] = self.cd.step(phase_pll[i], supply=vdd[i])

            # DCC
            phase_dcc[i] = self.dcc.step(phase_cd[i], supply=vdd[i])

        return phase_dcc

    # ============================================================
    # ======== Architecture from TX to RX slicer input ===========
    # ============================================================
    def process_front_end(self, bits, jitter_override=None, AGC_adaptation_on=True):
        """
        TX FFE -> TX Pre Driver -> TX Driver -> Channel -> RX AFE
        TX Clock ⬆️
        """

        # TX Clock Chain (샘플링 주파수 fs에 맞춰 위상 생성)
        n_bits = len(bits)
        n_samples = n_bits * self.g_cfg.samples_per_ui
        
        tx_phase = self.run_tx_clock(n_samples)

        # TX FFE (Digital Domain - 탭 가중치 계산)
        v_ffe = self.tx_ffe.process(bits)

        # 3. TX Pre-Driver (Clock Phase를 데이터 에지로 투영)
        tx_jittered_wave, bit_level_jitter = self.tx_predriver.process(v_ffe, tx_phase)
        self.last_jitter_samples = bit_level_jitter # 나중에 참고용

        # TX Driver (Analog Swing & Rise/Fall 반영)
        tx_out = self.tx_driver.process(tx_jittered_wave)


        """
        Channel Out: 매우 작은 열잡음(Thermal Noise)만 존재.      
        """
        # Channel 통과
        ch_out = self.channel.process(tx_out)

        # [데이터 수집] 순수 신호의 Peak-to-Peak 측정
        sig_p2p = np.ptp(ch_out)

        # Channel Noise (채널 끝단 열잡음)        
        # Crosstalk (Amplitude Noise 형태로 주입)
        n_ch = np.random.normal(0, self.cfg.rx.frontend.channel_out_noise, size=len(ch_out))
        n_xtalk = np.random.normal(0, self.cfg.rx.frontend.xtalk_rms, size=len(ch_out))
        rx_in = ch_out + n_ch + n_xtalk

        """
        CTLE Filtered: 신호는 복구되었으나 증폭은 미미함. 신호와 채널 노이즈를 함께 필터링
        CTLE Out: CTLE 내부 회로 자체에서 발생한 노이즈를 출력단에서 합산
        VGA In (여기서 IRN 주입): 여기서 발생하는 노이즈가 VGA 이득에 의해 증폭되어 Slicer의 결정에 가장 큰 영향을 미침.
        """
        # CTLE filtering (Pure Signal + Ch Noise)
        # CTLE Output Referred Noise (ORN)
        ctle_filtered = self.ctle.process(rx_in)
        n_ctle = np.random.normal(0, self.cfg.rx.frontend.ctle.ctle_orn_rms, size=len(ctle_filtered))
        ctle_out = ctle_filtered + n_ctle

        # AGC (VGA + Feedback)
        n_vga = np.random.normal(0, self.cfg.rx.frontend.vga_irn_rms, size=len(ctle_out))
        vga_in = ctle_out + n_vga
        # rx_afe_out = self.agc.process(vga_in, update_on=adaptation_on)
        rx_afe_out = ctle_out

        # [리포트용 데이터 저장] 
        # AGC Gain을 고려하여 모든 노이즈를 최종 출력(Slicer 입력) 관점으로 환산
        # final_gain = 10 ** (self.agc.current_gain_db / 20)
        final_gain = 1  # 임시로 AGC bypass
        self.last_noise_analysis = {
            "Channel Noise": np.std(n_ch) * final_gain,
            "Crosstalk": np.std(n_xtalk) * final_gain,
            "CTLE ORN": np.std(n_ctle) * final_gain,
            "VGA IRN": np.std(n_vga) * final_gain,
            "Signal P2P": sig_p2p * final_gain
        }

        return rx_in, ctle_out, rx_afe_out

    # ============================================================
    # ========== Link Training Sequence for ACI PHY ==============
    # ============================================================
    def run_link_up(self):
        """학습 단계를 수행하고 상태를 self에 저장"""
        """Step 1 & 2: CDR Phase Lock & DFE Tap Adaptation"""
        # 1. CDR Training (Clock Pattern)
        bits_clk = np.tile([1, 0], self.cfg.rx.cdr.tps1_len // 2)   # TPS1
        _, _, rx_clk = self.process_front_end(bits_clk)
        # indices_clk, phase_hist = blk.run_cdr_pi(rx_clk, self.cfg)
        # CDR reset
        self.cdr.phase = 0.5 * self.g_cfg.samples_per_ui
        self.cdr.freq = 0.0
        self.cdr.int_err = 0.0
        original_mode = self.cdr.mode
        self.cdr.mode = "real"  # 트레이닝은 항상 Real로

        indices_clk, phase_hist = self.cdr.run(rx_clk, true_jitter=self.last_jitter_samples)
        
        """
        VCO Noise (RX Clock RJ)
        """
        sigma_vco = self.cfg.rx.cdr.rx_clock_rj * self.g_cfg.samples_per_ui
        indices_clk = np.clip(indices_clk + np.random.normal(0, sigma_vco, size=len(indices_clk)), 0, len(rx_clk)-1).astype(int)

		# 1. 마지막 1000개의 데이터를 '평균' 내서 노이즈(Jitter)를 필터링한 하나의 숫자를 만듭니다. 
        avg_phase = np.mean(phase_hist[-1000:])
        
		# 2. 이 숫자가 0 ~ samples_per_ui(32) 범위를 유지하도록 나머지 연산을 합니다.
        self.locked_phase = float(avg_phase) % self.g_cfg.samples_per_ui
        
        print(f"[*] CDR Phase Locked: Phase Avg = {self.locked_phase:.2f} samples")
        
        # 2. DFE Training (PRBS Pattern)
        bits_train = blk.PRBS(self.g_cfg.prbs_order).generate(self.cfg.rx.dfe.tps2_len) # TPS2
        _, _, rx_train = self.process_front_end(bits_train)

		# 고정된 Locked Phase를 기준으로 모든 비트의 샘플링 인덱스 생성 (Static하게 DFE만 학습)
        # n번째 비트의 샘플링 위치 = (n * UI_samples) + 최적_위상_오프셋
        indices_train = (np.arange(len(bits_train)) * self.g_cfg.samples_per_ui + self.locked_phase).astype(int)
        # 파형 길이를 넘지 않도록 클리핑
        indices_train = np.clip(indices_train, 0, len(rx_train)-1)
        
        # Bit-by-bit DFE Loop (blocks.py의 DFE.process 활용)
        v_raw = rx_train[indices_train]
        for i in range(len(v_raw)):
            self.dfe.process(v_raw[i], adaptation_on=True)
            
        self.h1_lock, self.h2_lock = self.dfe.h1, self.dfe.h2
        self.cdr.mode = original_mode   # 원래 모드로 복구

        print(f"[*] DFE Taps Locked: h1={self.h1_lock:.3f}, h2={self.h2_lock:.3f}")

        return {
            "h1_hist": self.dfe.h1_hist,
            "h2_hist": self.dfe.h2_hist,
            "phase_hist": phase_hist
        }

    # ============================================================
    # ===================== BER Measurement ======================
    # ============================================================
    def measure_ber(self, bits_test, n_bits=50000, adaptation_on=False, jitter_override_ref=None):
        """
        Step 3: BER 측정 (CDR 모드 선택 가능)
        - adaptation_on: True로 설정하면 데이터 측정 중에도 DFE 탭을 계속 업데이트함
        """
        # bits_test = blk.generate_prbs(7, n_bits)
        # n_bits = len(bits_test) # 인자로 받은 비트 길이에 맞춤

        # 1. CDR 상태 초기화 (트레이닝 결과 반영 및 잔여 에러 제거)
        self.cdr.phase = self.locked_phase
        self.cdr.freq = 0.0
        self.cdr.int_err = 0.0

        # 2. CDR 모드 별 인덱스 추출
        if self.cdr.mode == "perfect":
            # 시드 고정을 통해 두 파형의 지터(RJ, SJ)를 완벽히 일치시킴
            current_seed = np.random.randint(0, 100000)
            
            # 1. 참조 클럭 생성 (1010 패턴)
            bits_ref = np.tile([1, 0], len(bits_test) // 2)
            np.random.seed(current_seed)
            _, _, rx_ref = self.process_front_end(bits_ref)
            
            # 2. 실제 데이터 생성 (PRBS 패턴)
            np.random.seed(current_seed)    # 동일한 시드 재고정
            rx_in, ctle_out, afe_out = self.process_front_end(bits_test)
            np.random.seed(None) # 시드 해제. 시뮬레이션의 다른 부분에서 랜덤성 필요할 수 있으므로
            
            # 3. CDR은 rx_ref를 보고 인덱스를 뽑음
            indices_test, phase_hist = self.cdr.run(rx_ref)
        else:
            # Real/Static 모드는 원래대로 데이터 파형 사용
            rx_in, ctle_out, afe_out = self.process_front_end(bits_test)
            indices_test, phase_hist = self.cdr.run(afe_out)

        """
        VCO Noise (RX Clock RJ)
        """
        # sigma_vco = self.cfg.rx.cdr.rx_clock_rj * self.g_cfg.samples_per_ui
        # vco_noise = np.random.normal(0, sigma_vco, size=len(indices_test))

        # 파형 길이를 벗어나지 않도록 클리핑
        # jittered_indices = np.clip(indices_test + vco_noise, 0, len(afe_out)-1).astype(int)
        jittered_indices = np.clip(indices_test, 0, len(afe_out)-1).astype(int)
        
        # # 3. RX Clock Jitter 주입
        # sigma_samples = self.cfg.rx.frontend.clock_jitter * self.g_cfg.samples_per_ui
        # rx_clock_noise = np.random.normal(0, sigma_samples, size=len(indices_test))
        # jittered_indices = np.clip(indices_test + rx_clock_noise, 0, len(afe_out)-1).astype(int)

        # 4. DFE & Slicing
        v_raw = afe_out[jittered_indices]
        n_bits = len(v_raw)
        final_bits = np.zeros(n_bits, dtype=int)
        v_corr_list = np.zeros(n_bits)
        
        # 탭 초기값 설정 (Training 결과를 이어받음)
        self.dfe.h1, self.dfe.h2 = self.h1_lock, self.h2_lock
        # self.dfe.h1_hist, self.dfe.h2_hist = [], []   # 기존 히스토리 초기화
        # 기존 히스토리를 유지하고 싶다면 초기화하지 않음. 
        # 만약 '이번 measure_ber 세션'의 것만 따로 보고 싶다면 
        # 아래처럼 현재 시점의 인덱스를 기억해두었다가 나중에 슬라이싱.
        start_idx = len(self.dfe.h1_hist)
        
        for i in range(n_bits):
            # adaptation_on 인자를 그대로 전달하여 LMS 업데이트 여부 결정
            bit, v_corr, _, _ = self.dfe.process(v_raw[i], adaptation_on=adaptation_on)
            final_bits[i] = bit
            v_corr_list[i] = v_corr
            
        # 만약 측정 중에도 학습을 했다면, 최종 상태를 다시 업데이트
        if adaptation_on:
            self.h1_lock, self.h2_lock = self.dfe.h1, self.dfe.h2

        ber, best_shift = blk.compute_ber(bits_test, final_bits)
        # print(f"[*] Calculated BER: {ber:.2e} (Found Shift: {best_shift})")

        return {
            "ber": ber,
            "v_corr": v_corr_list,
            "rx_samples": v_raw,
            "CDR_sampling_points": jittered_indices,
            "rx_ctle_in": rx_in,
            "rx_ctle_out": ctle_out,
            "rx_afe_out": afe_out,
            "shift": best_shift,
            "phase_hist": phase_hist,
            "rx_ref_clk": self.rx_ref_clk
            # # 이번 측정 구간의 히스토리만 잘라서 반환
            # "h1_hist": self.dfe.h1_hist[start_idx:], 
            # "h2_hist": self.dfe.h2_hist[start_idx:]
        }

    # ============================================================
    # ============== Noise/Jitter Budget Analysis ================
    # 원칙적으로는 섞여 있지만, 설계를 위해 인위적으로 나눔
    #                   Noise Budget	        Jitter Budget
    # 관심 영역	    전압 (Amplitude, Voltage)	시간 (Timing, Phase)
    # 단위	       mV, V	                  UI (Unit Interval), ps (picoseconds)
    # 영향	       Eye Diagram의 높이를 닫음	 Eye Diagram의 너비를 닫음
    # 주요 대책	    VGA 증폭, FFE/DFE (전압 보정) CDR 루프 대역폭 최적화, 깨끗한 전원
    # 최악의 상황	 신호가 너무 작아 0/1 판별 불가	  타이밍이 어긋나 엉뚱한 비트를 샘플링
    # ============================================================    
    def report_noise_contribution(self):
        """최종 Eye Closure에 기여한 노이즈 비율 리포트"""
        if not hasattr(self, 'last_noise_analysis'):
            print("[!] No noise data available. Run process_front_end first.")
            return

        data = self.last_noise_analysis
        # RSS(Root Sum Square) 방식으로 전체 노이즈 합산
        # 기여도(%): 각 노이즈의 분산(Variance, rms^2) 비율로 계산하는 것이 통계적으로 정확
        # 6-Sigma: 보통 Serial Link 설계에서 BER 10^-12를 달성하기 위해서는 노이즈 RMS의 약 14배(7sigma)를 보지만, 시뮬레이션 환경에서는 가시적인 분석을 위해 6σ를 기준으로 Eye Closure를 예측합니다.
        total_noise_rms = np.sqrt(
            data["Channel Noise"]**2 + 
            data["Crosstalk"]**2 + 
            data["CTLE ORN"]**2 + 
            data["VGA IRN"]**2
        )
        
        # Eye Closure 추정 (6-sigma 기준)
        eye_closure_total = 6 * total_noise_rms
        remaining_eye = max(0, data["Signal P2P"] - eye_closure_total)
        closure_pct = (eye_closure_total / data["Signal P2P"]) * 100

        print("\n" + "="*50)
        print(f"{'SERDES NOISE BUDGET ANALYSIS':^50}")
        print("="*50)
        print(f"Final VGA Gain:      {self.agc.current_gain_db:6.2f} dB")
        print(f"Signal Amplitude:    {data['Signal P2P']*1000:6.1f} mVpp (at Slicer)")
        print("-"*50)
        
        for source in ["Channel Noise", "Crosstalk", "CTLE ORN", "VGA IRN"]:
            rms = data[source]
            # 분산(Variance) 비율로 기여도 계산
            contribution = (rms**2 / total_noise_rms**2) * 100
            print(f"{source:<15}: {rms*1000:6.2f} mVrms ({contribution:5.1f} %)")
            
        print("-"*50)
        print(f"Total Noise RMS:     {total_noise_rms*1000:6.2f} mVrms")
        print(f"Est. Eye Closure:    {eye_closure_total*1000:6.1f} mV (6-sigma)")
        print(f"Est. Eye Opening:    {remaining_eye*1000:6.1f} mV ({100-closure_pct:5.1f} %)")

        print("-" * 50)
        print(f"{'JITTER BUDGET ANALYSIS (6-sigma)':^50}")
        print("-" * 50)
        tx_rj_ui = self.cfg.tx.predriver.rj * 6
        vco_rj_ui = self.cfg.rx.cdr.rx_clock_rj * 6
        total_jitter = np.sqrt(tx_rj_ui**2 + vco_rj_ui**2)
        
        print(f"TX Random Jitter:    {tx_rj_ui:6.3f} UI p-p")
        print(f"RX VCO Phase Noise:  {vco_rj_ui:6.3f} UI p-p")
        print(f"Total RJ (RSS):      {total_jitter:6.3f} UI p-p")
        print("="*50 + "\n")

        # 만약 나중에 더 정밀한 분석을 원하신다면, 리포트에 **"Vertical Eye Closure (mV)"**와 **"Horizontal Eye Closure (UI)"**라는 항목을 추가해서, 노이즈가 시간축에 얼마나 나쁜 짓(?)을 했는지 수치화해 볼 수도 있습니다.