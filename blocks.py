##### Note #####
# 26.3.7:
#   - PLL lock time이 필요할까?
#   - KVDD 주파수 특성 반영 해야, 지금은 상수로 되어있음
# 

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# def generate_bits(n_bits):
# 	return np.random.choice([0, 1], size=n_bits)

# ============================================================
# ========================= Clock ============================
# ============================================================

class ReferenceClock:
    """
    Reference clock generator

    Noise sources
    -------------
    1. KVDD_REF_CLK  : supply → jitter (ps/mV)
    2. RJ_REF_CLK    : random jitter spec (mUI)

    Output
    ------
    phase [rad]

    Note
    ----
    RJ spec is defined in UI_rms.
    Internally converted to phase jitter.

    RJ_time is stored for jitter budgeting report.
    """
    def __init__(self, cfg, g_cfg):
        self.freq = cfg.freq
        self.KVDD = cfg.KVDD_REF_CLK

        # simulation step
        self.Ts = 1 / g_cfg.fs

        # phase state
        self.phase = 0.0

        # RJ spec
        self.RJ_mUI = cfg.RJ_REF_CLK
        Tclk = 1 / self.freq
        rj_time = (self.RJ_mUI * 1e-3) * Tclk               # mUI → time jitter
        self.rj_sigma = 2 * np.pi * self.freq * rj_time     # time → phase

    def step(self, supply_noise_mv):
        # KVDD (ps/mV) → time jitter
        jitter_ps = self.KVDD * supply_noise_mv
        jitter_time = jitter_ps * 1e-12
        
        # time jitter → phase shift
        phase_shift = 2 * np.pi * self.freq * jitter_time

        # random jitter
        rj = np.random.normal(0, self.rj_sigma)

        # ideal phase increment
        dphi = 2 * np.pi * self.freq * self.Ts
        self.phase += dphi + phase_shift + rj

        # phase overflow protection
        self.phase = np.mod(self.phase, 2*np.pi)

        return self.phase

class PLL:
    """
    Phase-domain PLL model (Type2)

    Budgeting version

    PLL loop is used only for phase tracking.
    Random jitter is added at PLL output
    so that RJ budgeting matches system spec.
    """

    def __init__(self, cfg, g_cfg):

        # divider
        self.N = cfg.N

        # charge pump
        self.Icp = cfg.Icp

        # loop filter
        self.KLF = cfg.KLF  # loop filter transimpedance gain
        self.z = 2 * np.pi * cfg.zero
        self.p = 2 * np.pi * cfg.pole

        # VCO
        self.KVCO = cfg.KVCO
        self.f_center = cfg.f_center

        # supply sensitivity
        self.KVDD_LOOP = cfg.KVDD_LOOP
        self.KVDD_VCO = cfg.KVDD_VCO

        # simulation
        self.fs = g_cfg.fs
        self.Ts = 1 / self.fs

        # states
        self.phase_vco = 0.0

        # filter memory
        self.x_prev = 0.0
        self.y_prev = 0.0

        # --------------------------------
        # Budgeting Random Jitter
        # --------------------------------
        self.RJ_UI = getattr(cfg, "RJ_PLL", 0)

        Tclk = 1 / self.f_center
        self.rj_time = self.RJ_UI * Tclk
        self.rj_sigma = 2 * np.pi * self.f_center * self.rj_time

        # filter coeff
        self._compute_filter_coeff()

    # ------------------------------------------------
    # Divider
    # ------------------------------------------------
    def divider(self):
        return self.phase_vco / self.N

    # ------------------------------------------------
    # PFD + Charge Pump
    # ------------------------------------------------
    def pfd_cp(self, phase_ref, phase_div):

        phase_err = np.angle(np.exp(1j*(phase_ref - phase_div)))
        i_cp = self.Icp * phase_err

        return i_cp

    # ------------------------------------------------
    # Loop Filter Coefficient Calculation
    #
    # Continuous transfer function
    #   H(s) = (1 + s/z) / (1 + s/p)
    #
    # Bilinear transform
    #   s = (2/T)*(1 - z^-1)/(1 + z^-1)
    #
    # Resulting discrete IIR filter
    #
    #   y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
    # ------------------------------------------------
    def _compute_filter_coeff(self):

        T = self.Ts
        k = 2 / T

        # numerator
        b0 = (1 + k/self.z)
        b1 = (1 - k/self.z)

        # denominator
        a0 = (1 + k/self.p)
        a1 = (1 - k/self.p)

        # normalize
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.a1 = a1 / a0

    # ------------------------------------------------
    # Loop Filter Step
    #
    # x = charge pump current
    # y = control voltage
    #
    # Discrete IIR filter
    #
    #   y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]
    #
    # supply noise is converted to jitter through KVDD
    # ------------------------------------------------
    def loop_filter(self, i_cp, supply_noise_mv):

        # input
        x = i_cp

        # IIR filter
        y = (
            self.b0 * x +
            self.b1 * self.x_prev -
            self.a1 * self.y_prev
        )

        # update memory
        self.x_prev = x
        self.y_prev = y

        # supply noise → phase jitter
        jitter_ps = self.KVDD_LOOP * supply_noise_mv
        jitter = jitter_ps * 1e-12

        v_ctrl = self.KLF * y + jitter

        return v_ctrl

    # ------------------------------------------------
    # VCO (no intrinsic RJ here)
    # ------------------------------------------------
    def vco(self, v_ctrl, supply_noise_mv):

        # VCO frequency
        freq = self.f_center + self.KVCO * v_ctrl

        # supply coupling
        jitter_ps = self.KVDD_VCO * supply_noise_mv
        jitter_time = jitter_ps * 1e-12

        # phase integration
        dphi = 2 * np.pi * freq * self.Ts

        self.phase_vco += dphi
        self.phase_vco += 2 * np.pi * freq * jitter_time

        # phase overflow protection
        self.phase_vco = np.mod(self.phase_vco, 2*np.pi)

        return self.phase_vco

    # ------------------------------------------------
    # PLL Output Jitter Injection
    # ------------------------------------------------
    def add_output_jitter(self, phase):

        rj = np.random.normal(0, self.rj_sigma)

        return phase + rj

    # ------------------------------------------------
    # Top step
    # ------------------------------------------------
    def step(self, phase_ref, supply_noise_mv):

        phase_div = self.divider()

        i_cp = self.pfd_cp(phase_ref, phase_div)

        v_ctrl = self.loop_filter(i_cp, supply_noise_mv)

        phase_vco = self.vco(v_ctrl, supply_noise_mv)

        # --------------------------------
        # Budgeting jitter added here
        # --------------------------------
        phase_out = self.add_output_jitter(phase_vco)

        return phase_out

class ClockDistribution:
    """
    Clock buffer model (LPF)

    Bandwidth determined by rise/fall time.

    BW ≈ 0.35 / tr

    Noise
    -----
    KVDD_CD : supply → jitter
    RJ_CD   : random jitter (mUI)
    """

    def __init__(self, cfg, g_cfg):
        self.freq = g_cfg.bit_rate / 2
        self.KVDD = cfg.KVDD_CD
        self.Ts = 1 / g_cfg.fs

        # bandwidth from rise/fall time
        UI = 1 / g_cfg.bit_rate
        tr = cfg.cd_trf_ui * UI
        BW = 0.35 / tr
        self.wc = 2 * np.pi * BW
        self.phase_state = 0.0
        self.jitter_state = 0.0

        # RJ spec
        self.RJ_mUI = cfg.RJ_CD
        Tclk = 1 / self.freq
        rj_time = (self.RJ_mUI * 1e-3) * Tclk
        self.rj_sigma = 2 * np.pi * self.freq * rj_time

    def step(self, phase_in, supply_noise_mv):

        # phase → time jitter
        t_jitter = phase_in / (2 * np.pi * self.freq)

        # jitter LPF
        BW = self.wc / (2*np.pi)
        tau = 1 / (2*np.pi*BW)
        alpha = self.Ts / (self.Ts + tau)

        self.jitter_state += alpha * (t_jitter - self.jitter_state)

        # filtered jitter → phase
        phase = 2 * np.pi * self.freq * self.jitter_state

        # supply jitter
        jitter_ps = self.KVDD * supply_noise_mv
        jitter_time = jitter_ps * 1e-12
        phase += 2 * np.pi * self.freq * jitter_time

        # random jitter
        rj = np.random.normal(0, self.rj_sigma)
        phase += rj

        # phase overflow protection
        phase = np.mod(phase, 2*np.pi)

        return phase

class DCC:
    """
    Duty Cycle Corrector

    AC coupled inverter → bandpass behaviour

    H(s) = (s/wL)/(1+s/wL) * 1/(1+s/wH)

    Noise Sources
    -------------
    KVDD_DCC : supply → jitter
    RJ_DCC   : random jitter (mUI)

    DCD model
    ---------
    Duty cycle distortion shifts rising/falling edges
    in opposite directions.
    """

    def __init__(self, cfg, g_cfg):

        self.freq = cfg.freq
        self.KVDD = cfg.KVDD_DCC
        self.Ts = 1 / g_cfg.fs

        # bandpass frequencies
        self.wL = 2 * np.pi * cfg.dcc_hp
        self.wH = 2 * np.pi * cfg.dcc_lp

        self.hp_state = 0.0
        self.lp_state = 0.0

        # RJ spec
        self.RJ_mUI = cfg.RJ_DCC
        Tclk = 1 / self.freq
        rj_time = (self.RJ_mUI * 1e-3) * Tclk
        self.rj_sigma = 2 * np.pi * self.freq * rj_time

        # Duty Cycle Distortion
        self.DCD = cfg.DCD
        self.phase_dcd = 2 * np.pi * self.DCD

        # edge detection memory
        self.prev_phase_mod = 0.0
        self.edge_flag = 0
        self.prev_phase = 0.0

    def step(self, phase_in, supply_noise_mv):

        # high pass (AC coupling)
        dhp = self.wL * (phase_in - self.hp_state)
        self.hp_state += dhp * self.Ts
        hp = phase_in - self.hp_state

        # low pass
        dlp = self.wH * (hp - self.lp_state)
        self.lp_state += dlp * self.Ts
        phase = self.lp_state

        # supply jitter
        jitter_ps = self.KVDD * supply_noise_mv
        jitter_time = jitter_ps * 1e-12

        phase += 2 * np.pi * self.freq * jitter_time

        # random jitter
        rj = np.random.normal(0, self.rj_sigma)

        phase += rj

        # Duty Cycle Distortion: High 구간과 Low 구간을 서로 반대로 밀어줌
        # rising edge → +DCD/2
        # falling edge → -DCD/2
        # 1 UI를 2pi로 볼 때, Rising 에지는 0, Falling 에지는 pi 위치임
        phase_mod = np.mod(phase, 2*np.pi)

        edge = False
        if phase_mod < self.prev_phase_mod:
            edge = True

        if edge:
            if phase_mod < np.pi:
                phase += self.phase_dcd / 2
            else:
                phase -= self.phase_dcd / 2

        self.prev_phase_mod = phase_mod

        # phase overflow protection
        phase = np.mod(phase, 2*np.pi)

        return phase


# ============================================================
# ========================= TX SIDE ==========================
# ============================================================

class PRBS:
    def __init__(self, order):
        """
        order: 7 (PRBS7) 또는 23 (PRBS23)
        """
        self.order = order

    def generate(self, length):
        """
        length: 생성할 비트의 총 길이
        """
        # PRBS 차수에 따른 다항식 설정 (ITU-T O.150 표준)
        if self.order == 7:
            # G(x) = x^7 + x^6 + 1
            taps = (7, 6)
            state = 0x7F  # 초기 상태 (모두 1)
        elif self.order == 23:
            # G(x) = x^23 + x^18 + 1
            taps = (23, 18)
            state = 0x7FFFFF
        else:
            raise ValueError("PRBS7 또는 PRBS23만 지원합니다.")

        bits = np.zeros(length, dtype=int)

        # LFSR 동작
        for i in range(length):
            # 출력 비트 (최상위 비트 추출)
            bits[i] = (state >> (self.order - 1)) & 1

            # 피드백 비트 계산 (XOR)
            new_bit = ((state >> (taps[0] - 1)) ^ (state >> (taps[1] - 1))) & 1

            # 상태 업데이트 (Shift & Insert feedback)
            state = ((state << 1) | new_bit) & ((1 << self.order) - 1)

            # LFSR이 0으로 빠지는 것 방지
            if state == 0:
                state = 1

        return bits

class FFE:
    """
    [Digital Domain] 3-tap Voltage Mode Driver FFE
    bits: 원본 데이터 비트 (0 or 1)
    """
    def __init__(self, cfg):      
        self.cfg = cfg  # FFEConfig

    def process(self, bits):
        slice_per_level = 1
        n_pre = self.cfg.pre_level * slice_per_level
        n_post = self.cfg.post_level * slice_per_level

        n_total = 18    # TX Main drv slice 총 18개
        n_main = n_total - n_pre - n_post   # 남는 슬라이스를 Main에 할당

        # 탭별 데이터 준비 (시프트)
        x = np.where(bits > 0, 1, -1)   # 내부적으로 -1, 1로 변환하여 계산
        x_padded = np.pad(x, (1, 1), mode='edge')
        x_pre = x_padded[2:]    # 미래 (Pre-cursor)
        x_main = x_padded[1:-1] # 현재 (Main-tap)
        x_post = x_padded[0:-2] # 과거 (Post-cursor)

        # VMD 슬라이스 합산 (De-emphasis 적용)
        # 결과값은 -1.0 ~ 1.0 사이의 불연속적인 전압 레벨
        v_ffe_bits = (n_main * x_main - n_pre * x_pre - n_post * x_post) / n_total

        return v_ffe_bits

class TXPreDriver:
    """
    2-to-1 MUX + Pre-driver

    Inputs
    ------
    bits : PRBS / FFE output bits (0/1)
    phase_samples : clock phase from
                    ReferenceClock → PLL → CD → DCC
                    (sample-domain phase [rad])

    Outputs
    -------
    jittered_wave
    bit_level_jitter
    """

    def __init__(self, cfg, global_cfg):

        self.cfg = cfg
        self.g_cfg = global_cfg
        self.fs = global_cfg.fs
        self.spui = global_cfg.samples_per_ui
        self.bit_rate = global_cfg.bit_rate

        self.freq = global_cfg.bit_rate / 2

        self.rj = cfg.rj    # RJ budget (UI rms)
        self.ddj = cfg.ddj

        self.psij_amp = global_cfg.psij_amp
        self.psij_freq = global_cfg.psij_freq
        self.KVDD_PREDRV = cfg.KVDD_PREDRV

    def process(self, bits, phase_samples):

        # 이상적인 시간축 생성
        n_bits = len(bits)
        spui = self.spui
        total_samples = n_bits * spui
        t_ideal = np.arange(total_samples)

        # bit boundary indices
        # 비트 경계 지점 정의 (0, 1*spui, 2*spui, ..., n_bits*spui)
        bit_boundary = np.arange(n_bits + 1) * spui

        # 1. Clock phase → bit boundary phase 추출
        # clock chain은 sample마다 phase 생성
        # TX jitter는 bit boundary 기준이므로
        # boundary 위치의 phase만 추출
        # phase_bits = phase_samples[bit_boundary]  # 이건 대부분 맞지만 clock/data alignment 문제
        phase_bits = phase_samples[np.clip(bit_boundary,0,len(phase_samples)-1)]

        # 2. Phase → sample jitter 변환
        # phase [rad]
        # t_jitter [seconds]= phase / (2π) * Tclk
        # sample_jitter [samples] = t_jitter * fs
        t_jitter = phase_bits / (2 * np.pi) * (1 / self.freq)
        clk_jitter = t_jitter * self.fs

        # 3. Random Jitter (RJ)
        noise = np.random.normal(0, self.rj * spui, size=len(bits))
        # rj = np.cumsum(noise)
        rj = noise

        # 4. Data Dependent Jitter (DDJ)
        #   - transition-based 모델
        # data transition
        # 0→1 / 1→0  → edge 존재 → jitter 영향
        # 0→0 / 1→1  → edge 없음 → 영향 작음
        transitions = np.diff(np.concatenate([[bits[0]], bits]))
        # ddj = transitions * self.ddj * spui
        ddj = (np.abs(transitions) > 0) * self.ddj * spui   # transition → 항상 같은 jitter
        ddj = np.concatenate([ddj, [0]])

        # 5. Sinusoidal supply ripple (global PSIJ)
        bit_times = bit_boundary / (spui * self.bit_rate)

        # global supply ripple (mV)
        vdd = (
            self.psij_amp / 2 *
            np.sin(2 * np.pi * self.psij_freq * bit_times)
        )

        tj_ps = self.KVDD_PREDRV * vdd  # convert supply ripple → time jitter (ps)
        psij = tj_ps * 1e-12 * self.fs  # ps → samples

        # 6. Total jitter
        total_bit_jitter = clk_jitter + rj + ddj + psij

        # # 7. Bit jitter → sample jitter interpolation
        # # 비트별 지터를 샘플 단위로 확장 (Interpolation)
        # # 비트 경계에서의 지터 값을 샘플 단위로 부드럽게 연결
        # sample_jitter = np.interp(
        #     t_ideal,
        #     bit_boundary,
        #     total_bit_jitter
        # )

        # # 8. NRZ waveform 생성
        # # 시간축 변환 및 재샘플링
        # ideal_nrz = np.repeat(bits, spui)

        # # 9. Time warping
        # # t_jittered = t_ideal + sample_jitters (샘플들의 실제 위치가 흔들림)
        # # np.interp(원하는 위치, 실제 위치, 실제 값)
        # jittered_wave = np.interp(
        #     t_ideal,
        #     t_ideal + sample_jitter,
        #     ideal_nrz
        # )

        # 7. Edge reconstruction (resample 방식)
        edges = bit_boundary + total_bit_jitter
        jittered_wave = np.zeros(total_samples)

        for i in range(n_bits):
            # start = int(np.clip(edges[i], 0, total_samples-1))
            # end   = int(np.clip(edges[i+1], 0, total_samples))
            start = int(np.floor(edges[i]))
            end   = int(np.floor(edges[i+1]))

            if end > start:
                jittered_wave[start:end] = bits[i]

        # 10. Bit-level jitter (CDR 전달용)
        # CDR 전달용 비트별 지터 계산
        # 중앙이 아니라 시작점(Boundary) 기준의 지터를 넘겨줌
        # bit_start = np.arange(n_bits) * spui

        # bit_level_jitter = np.interp(
        #     bit_start,
        #     t_ideal,
        #     sample_jitter
        # )
        bit_level_jitter = total_bit_jitter[:-1]

        return jittered_wave, bit_level_jitter


class TXDriver:
    """
    swing: Peak-to-Peak 전압 (V)
    trf_ui: Rise/Fall Time (UI 단위)
    """
    def __init__(self, cfg, global_cfg):
        self.cfg = cfg            # TXDriverConfig (trf_ui, swing 등 포함)
        self.global_cfg = global_cfg  # GlobalConfig (samples_per_ui 포함)

    def process(self, waveform):
        # 1. Swing 레벨 적용 (-1/1 -> -V/2, +V/2)
        # 입력이 0/1 근처라면 (jittered_wave - 0.5) * 2 등을 통해 -1/1로 정규화 후 swing 곱셈
        driver_output = waveform * (self.cfg.swing / 2)
    
        # 2. Rise/Fall Time 적용 (LDO/Driver Bandwidth 제한)
        window_size = int(self.cfg.txdrv_trf_ui * self.global_cfg.samples_per_ui)

        if window_size > 1:
            # 가우시안 커널을 쓰면 이동평균보다 더 실제 Rise/Fall 곡선과 유사
            std = window_size / 4
            kernel = np.exp(
                -(np.arange(window_size) - window_size/2)**2 /
                (2 * std**2)
            )
            kernel /= kernel.sum()
            driver_output = np.convolve(driver_output, kernel, mode='same')

        return driver_output


# ============================================================
# ========================= CHANNEL ==========================
# ============================================================

class Channel:
    """
    1) RC 채널 → 이 링크가 성립하는지?
    2) FIR 채널 → equalization으로 복구 가능한지?
    3) Butterworth → BW spec이 합리적인지?
    """
    def __init__(self, cfg, global_cfg):
        self.cfg = cfg  # ChannelConfig
        self.global_cfg = global_cfg  # GlobalConfig (samples_per_ui 포함)

    def process(self, waveform):
        if self.cfg.type == "rc":
            h = self._rc_channel()
            return np.convolve(waveform, h, mode="same")

        elif self.cfg.type == "fir":
            h = self._fir_channel()
            return np.convolve(waveform, h, mode="same")

        elif self.cfg.type == "butter":
            return self._butter_channel(waveform)

        else:
            raise ValueError("Unknown channel type")

    def _rc_channel(self):
        ui = 1 / self.cfg.bit_rate
        dt = ui / self.global_cfg.samples_per_ui
        t = np.arange(0, self.cfg.length_ui * ui, dt)
        tau = 1 / (2 * np.pi * self.cfg.fc)
        h = (1 / tau) * np.exp(-t / tau)
        h /= np.sum(h)  # DC normalization
        return h

    def _fir_channel(self):
        h = np.array([1.0, 0.3, 0.15, 0.05])
        h /= np.sum(h)
        return h

    def _butter_channel(self, waveform):
        b, a = signal.butter(
            1,
            self.cfg.bw / (self.cfg.fs / 2)
        )
        return signal.lfilter(b, a, waveform)
    

# ============================================================
# ========================= RX SIDE ==========================
# ============================================================

class CTLE:
    """
    2-pole, 1-zero CTLE Model
    - peak_freq: 증폭 목표 주파수
    - p1, p2: Pole 주파수 (Hz)
    - boost_db: 저주파 대비 고주파 증폭량
    """
    def __init__(self, cfg):
        self.cfg = cfg

    def process(self, waveform):
        # 1. Zero 위치 계산 (Boost 양에 따라 결정)
        # Gain(peak) / Gain(DC) 비율이 boost_db가 되도록 설정
        # 단순화된 모델: H(s) = (s/wz + 1) / [(s/wp1 + 1)(s/wp2 + 1)]
        w_p1 = 2 * np.pi * self.cfg.p1
        w_p2 = 2 * np.pi * self.cfg.p2

        # Boost 양에 맞춰 Zero 위치(w_z) 역산
        # 보통 wz는 고주파를 살리기 위해 낮은 주파수에 위치함
        w_z = w_p1 / (10**(self.cfg.selected_boost / 20))

        # 2. S-domain 계수 정의
        # 분자 (Numerator): s/wz + 1
        num_s = [1/w_z, 1]
        # 분모 (Denominator): (s/wp1 + 1)(s/wp2 + 1) = s^2/(wp1*wp2) + s(1/wp1 + 1/wp2) + 1
        den_s = [1/(w_p1*w_p2), (1/w_p1 + 1/w_p2), 1]

        # 3. Bilinear Transformation (Analog -> Digital)
        b, a = signal.bilinear(num_s, den_s, self.cfg.fs)

        # 4. 필터 적용
        return signal.lfilter(b, a, waveform)

class AGC:
    def __init__(self, cfg):
        self.cfg = cfg      # cfg는 FrontendConfig 객체
        self.current_gain_db = cfg.vga_gain_db  # 초기 Gain 설정
        self.vga_irn_rms = cfg.vga_irn_rms      # VGA Input Referred Noise (V rms)
        self.target_v = cfg.agc_v_target        # 목표 진폭 (e.g., 0.3V)
        self.mu = cfg.agc_mu                    # 이득 업데이트 속도
        self.gain_hist = []

    def process(self, waveform, update_on=True):
        # 1. 현재 Gain 적용
        gain_linear = 10 ** (self.current_gain_db / 20)
        output = waveform * gain_linear
        
        # 2. Adaptation (신호의 절대값 평균을 목표값과 비교)
        if update_on:
            # 신호의 Envelope 또는 RMS를 측정하여 에러 계산
            avg_amp = np.mean(np.abs(output))
            # 에러 계산: 목표치보다 작으면 Gain 증가, 크면 감소
            error = self.target_v - (avg_amp * 1.5) # 1.5는 Peak-to-Avg 보정 계수
            self.current_gain_db += self.mu * error
            
            # 하드웨어 한계 (VGA Max Gain) 설정
            self.current_gain_db = np.clip(self.current_gain_db, 0, 30) 
            
        return output

# class AnalogFrontEnd:
#     '''
#     Channel Out: 매우 작은 열잡음(Thermal Noise)만 존재.
#     CTLE Out: 신호는 복구되었으나 증폭은 미미함.
#     VGA In (여기서 IRN 주입): 여기서 발생하는 노이즈가 VGA 이득에 의해 증폭되어 Slicer의 결정에 가장 큰 영향을 미침.
#     '''
#     def __init__(self, cfg):
#         self.cfg = cfg              # FrontendConfig
#         self.ctle = CTLE(cfg.ctle)  # CTLEConfig를 전달
#         self.agc = AGC(cfg)         # VGA/AGC 설정을 포함한 FrontendConfig 전달

#     def process(self, waveform, adaptation_on=True):
#         # 신호 흐름: Input -> CTLE -> AGC -> Output
#         # 1. Channel Noise (채널 끝단 열잡음)
#         ch_noisy = waveform + np.random.normal(0, self.cfg.channel_out_noise, size=len(waveform))

#         # 2. CTLE filtering (Pure Signal + Ch Noise)
#         # CTLE는 신호와 채널 노이즈를 함께 필터링
#         ctle_filtered = self.ctle.process(ch_noisy)
        
#         # 3. CTLE Output Referred Noise (ORN)
#         # CTLE 내부 회로 자체에서 발생한 노이즈를 출력단에서 합산합니다.
#         ctle_out = ctle_filtered + np.random.normal(0, self.cfg.ctle.ctle_orn_rms, size=len(ctle_filtered))

#         # 4. AGC (VGA + Feedback)
#         # IRN 노이즈는 VGA 입력단에서 더해짐
#         # vga_in = ctle_out + np.random.normal(0, self.cfg.vga_irn_rms, size=len(ctle_out))
#         # rx_afe_out = self.agc.process(vga_in, update_on=adaptation_on)
#         rx_afe_out = ctle_out

#         return ctle_out, rx_afe_out   # CTLE out, AGC out

class CDR:
    def __init__(self, cfg, global_cfg):
        """
        mode: "perfect" | "static" | "real"
        
        1. static (기준점)
        특징: 학습 때 찾은 위치에 고정.
        한계: SJ(저주파 지터)가 있으면 눈(Eye)이 좌우로 흔들리는데, 클럭은 가만히 있어 BER이 폭발함.

        2. real (실전용)
        특징: Alexander PD와 PI Loop를 통해 지터 궤적을 실시간 추적.
        한계: 루프 필터 때문에 반응이 미세하게 늦고(Lag), RJ(고주파 지터)에 의해 클럭 자체가 떨림.

        3. perfect (이론적 한계)
        특징: 참조 클럭에서 에지 궤적을 뽑아 RJ만 필터링(Moving Average).
        강점: SJ는 100% 따라가고 RJ는 무시함. **"이 시스템에서 나올 수 있는 최상의 성능"**을 보여주는 벤치마크.
        """
        self.cfg = cfg  # CDRConfig
        self.global_cfg = global_cfg

        self.spui = global_cfg.samples_per_ui
        self.br = global_cfg.bit_rate

        self.mode = cfg.mode
        self.damping = cfg.damping
        self.f_3db = cfg.f_3db
        self.CDR_offset = cfg.CDR_offset

        # ---- Loop filter gains ----
        self._compute_loop_gains()

        # ---- State ----
        self.phase = 0.5 * self.spui     # sampling phase (samples)
        self.freq  = 0.0                 # frequency correction
        self.int_err = 0.0               # integrator state

        self.phase_hist = []

    # -------------------------------------------------
    def _compute_loop_gains(self):
        wn = (2 * np.pi * self.f_3db) / (
            np.sqrt(1 + 2*self.damping**2 +
            np.sqrt((1 + 2*self.damping**2)**2 + 1))
        )   # f_3dB를 바꿀 때마다 fn이 자동으로 계산되도록

        self.kp = 2 * self.damping * wn / self.br
        self.ki = (wn**2) / (self.br**2)

    # -------------------------------------------------
    def _sample(self, waveform, idx):
        idx = int(np.clip(idx, 0, len(waveform) - 1))
        return waveform[idx]

    # -------------------------------------------------
    def _alexander_pd(self, early, data, late):
        """
        Bang-Bang Alexander Phase Detector
        Returns: +1 (early), -1 (late), 0 (no update)
        """
        d = 1 if data >= 0 else -1
        e = 1 if early >= 0 else -1
        l = 1 if late >= 0 else -1

        if d != e and d == l:
            return -1   # late (에지가 데이터보다 늦게 옴 -> 위상을 줄여야 함)
        elif d == e and d != l:
            return 1    # early (에지가 데이터보다 빨리 옴 -> 위상을 늘려야 함)
        else:
            return 0    # No transition

    # -------------------------------------------------
    def run(self, waveform, true_jitter=None):
        """
        waveform: CTLE output waveform
        true_jitter: (optional) TX jitter samples (for perfect mode)
        """
        n_bits = len(waveform) // self.spui
        indices = np.zeros(n_bits, dtype=int)
        self.phase_hist = []

        # 하드웨어 제약 적용 함수 내부 정의
        def apply_hw_constraints(raw_indices):
            # 1. PI Resolution (6-bit 등) 적용
            # 1 UI를 2^N 단계로 나눔
            pi_step = self.spui / (2 ** self.cfg.pi_resolution_bits)
            quantized = np.round(raw_indices / pi_step) * pi_step
            
            # 2. RX Intrinsic Jitter (VCO Phase Noise)
            # CDR 루프와 상관없이 발생하는 RX 내부 클럭의 흔들림
            rx_internal_rj = np.random.normal(0, self.cfg.rx_clock_rj * self.spui, size=len(raw_indices))
            
            # 정수형 인덱스로 변환하여 반환
            return np.clip(quantized + rx_internal_rj, 0, len(waveform)-1).astype(int)
            # return np.clip(quantized, 0, len(waveform)-1).astype(int)

        # 1. Perfect 모드: 이상적 추적 (Ideal Tracking) ---
        if self.mode == "perfect":
            signs = np.sign(waveform)
            zero_crossings = np.where(signs[:-1] != signs[1:])[0]
            
            if len(zero_crossings) < 10:
                return (np.arange(n_bits) * self.spui + self.phase + self.spui//2).astype(int), np.zeros(n_bits)

            # 1. 각 UI 경계에 가장 적합한 에지 하나씩 매칭
            raw_edges = []
            for i in range(n_bits + 1):
                target = i * self.spui + self.phase # Training된 에지 위치 기준
                # target 근처 +/- 0.5 UI 범위 내에서만 에지 탐색
                in_window = zero_crossings[(zero_crossings > target - self.spui//2) & 
                                           (zero_crossings < target + self.spui//2)]
                
                if len(in_window) > 0:
                    # 윈도우 내 에지가 있다면 가장 가까운 것 선택
                    raw_edges.append(in_window[np.argmin(np.abs(in_window - target))])
                else:
                    # 에지가 없다면 이상적인 위치(target)를 강제로 넣음 (Drop 방지)
                    raw_edges.append(target)
            
            raw_edges = np.array(raw_edges, dtype=float)
            
            # 2. 이동 평균 필터 (RJ 제거)
            window_size = 31
            # 패딩을 'edge'로 하여 시작과 끝단에서의 위상 왜곡 방지
            padded = np.pad(raw_edges, window_size//2, mode='edge')
            smooth_edges = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
            
            # 3. 샘플링 인덱스 생성
            # smooth_edges[i]는 i번째 비트의 시작 에지, smooth_edges[i+1]은 끝 에지
            # 그 정중앙이 최적의 샘플링 포인트
            indices = []
            for i in range(n_bits):
                sampling_point = (smooth_edges[i] + smooth_edges[i+1]) / 2
                sampling_point += self.CDR_offset   # 약 1~2샘플 정도 오른쪽으로 강제 시프트하여 중앙을 맞춤 (실무적 튜닝)
                indices.append(int(round(sampling_point)))
            
            indices = np.array(indices)

            # return np.clip(indices, 0, len(waveform)-1), np.zeros(n_bits)
            return apply_hw_constraints(indices), np.zeros(n_bits)  # perfect는 phase변화가 없으므로, zeros처리

        # 2. STATIC 모드: 학습된 위상에 고정 (추적 X)
        elif self.mode == "static":
            # for i in range(n_bits):
            #     # 학습된 phase(Edge)에서 0.5 UI 옆인 중앙을 샘플링
            #     indices[i] = int(i * self.spui + self.phase + self.spui / 2)
            #     self.phase_hist.append(self.phase)
            # return np.clip(indices, 0, len(waveform)-1), np.array(self.phase_hist)

            # 학습된 self.phase(Edge)를 기준으로 0.5UI 옆인 정중앙 샘플링 위치 계산
            # n_bits 만큼의 베이스 위치 생성
            raw_static_indices = (np.arange(n_bits) * self.spui + self.phase + self.spui / 2)
        
            # 여기서 PI 양자화와 내부 지터를 입힘
            final_indices = apply_hw_constraints(raw_static_indices)
        
            # 히스토리는 양자화된 고정 위상값으로 저장
            self.phase_hist = [self.phase] * n_bits 
            return final_indices, np.array(self.phase_hist)

        # 3. REAL 모드: PI 루프를 돌며 실시간 추적
        else:
            for i in range(n_bits):
                ui_base = i * self.spui

                # 1. 샘플링 (현재 phase를 기준으로 중앙을 찍음)
                center_idx = ui_base + self.phase + self.spui / 2
                current_sample_idx = int(np.clip(center_idx, 0, len(waveform)-1))
                
                # 결과를 indices에 먼저 저장
                indices[i] = current_sample_idx
            
                # 2. PD 에러 추출 (D_prev, Edge, D_curr)
                # 루프 첫 비트는 d_prev가 없으므로 예외 처리 필요
                if i > 0:
                    # d_prev는 바로 이전 비트의 샘플링 위치(indices[i-1])를 그대로 사용
                    d_prev_val = waveform[indices[i-1]] 
                    # edge는 현재 비트의 시작점(phase)
                    edge_idx = int(np.clip(ui_base + self.phase, 0, len(waveform)-1))
                    # edge_val   = self._sample(waveform, ui_base + self.phase)
                    edge_val = waveform[edge_idx]
                    # d_curr_val은 방금 indices[i]에 넣은 그 위치의 값
                    d_curr_val = waveform[current_sample_idx]

                    pd_err = self._alexander_pd(d_prev_val, edge_val, d_curr_val)
                
                    # 3. 위상 업데이트 (단위 주의: kp, ki가 이미 spui를 포함하는지 확인)
                    self.int_err += pd_err * self.ki * self.spui
                    self.phase += (pd_err * self.kp * self.spui) + self.int_err
                    self.phase %= self.spui
            
                self.phase_hist.append(self.phase)
            
            # return indices, np.array(self.phase_hist)
            return apply_hw_constraints(indices), np.array(self.phase_hist)


class DFE:
    def __init__(self, cfg, h1_init=0.0, h2_init=0.0):
        self.cfg = cfg  # DFEConfig

        # Continuous internal tap accumulator
        self.h1 = h1_init
        self.h2 = h2_init

        # Previous decisions (+1 / -1)
        self.prev_d1 = -1
        self.prev_d2 = -1

        # History
        self.h1_hist = []
        self.h2_hist = []

    # ----------------------------------------
    def _quantize(self, value):
        return round(value / self.cfg.lsb) * self.cfg.lsb

    # ----------------------------------------
    def process(self, v_raw, adaptation_on=True):
        """
        Sign-Sign LMS DFE
        mu: Step size (SS-LMS에서는 보통 작은 값을 써서 천천히 수렴시킴)
        v_target: 에러 판정의 기준이 되는 목표 전압 (Adaptation Reference)
        h1: 1st tap coefficient (V)
        h2: 2nd tap coefficient (V)

        v_raw: CTLE + CDR sampling output
        returns:
            bit (0/1),
            v_corrected,
            h1_quantized,
            h2_quantized
        """

        # Quantized tap (실제 HW가 쓰는 값)
        h1_q = self._quantize(self.h1)
        h2_q = self._quantize(self.h2)

        # DFE correction
        v_corr = v_raw - (self.prev_d1 * h1_q + self.prev_d2 * h2_q)

        # # Slicer: Decision (0인지 1인지) --> Threshold / noise 문제 = slicer 문제
        # bit = 1 if v_corr >= self.cfg.threshold else 0
        # d_curr = 1 if bit == 1 else -1

        # --- Slicer Non-ideality 반영 ---
        # 1. DC Offset: 공정 편차로 인해 판정 기준선이 0V가 아닌 상황
        v_eff = v_corr - self.cfg.slicer_offset 
        
        # 2. Sensitivity (Dead-band): 전압이 너무 낮으면 결과가 랜덤하게 튀는 현상
        if abs(v_eff) < self.cfg.slicer_sensitivity:
            bit = np.random.choice([0, 1]) # Meta-stability 영역
        else:
            bit = 1 if v_eff >= self.cfg.threshold else 0

        d_curr = 1 if bit == 1 else -1

        # Adaptation (SS-LMS)
        if adaptation_on:
            # 1. Error의 부호(Sign)를 추출. 실제 전압과 목표 전압의 차이값 자체가 아니라, 그 방향(+, -)만 봄
            error_sign = np.sign(
                (self.cfg.v_target * d_curr) - v_corr
            )

            # 2. Tap 업데이트 시 'Error의 부호'와 '이전 결정값의 부호'만 사용
            # clipping을 통해 탭 값이 음수가 되거나 무한히 커지는 것을 방지
            self.h1 = np.clip(
                self.h1 + self.cfg.mu * error_sign * self.prev_d1,
                0, self.cfg.tap_max[0]
            )

            self.h2 = np.clip(
                self.h2 + self.cfg.mu * error_sign * self.prev_d2,
                0, self.cfg.tap_max[1]
            )

        # History save
        self.h1_hist.append(h1_q)
        self.h2_hist.append(h2_q)

        # Shift register
        self.prev_d2 = self.prev_d1
        self.prev_d1 = d_curr

        return bit, v_corr, h1_q, h2_q


# ============================================================
# ======================== UTILITIES =========================
# ============================================================

def db_to_linear(db):
    return 10**(db/10)

def linear_to_db(x):
    return 10*np.log10(x + 1e-30)

def compute_ber(bits_true, bits_rx, max_shift=20):
    """
    bits_true: 송신 PRBS 비트
    bits_rx: CDR/DFE를 거쳐 판정된 수신 비트
    max_shift: 앞뒤로 몇 비트까지 정렬(Sync)을 시도할지 범위
    - 만약 채널 지연이 긴 환경(latency가 큰 채널)이라면 
    - 즉, best_shift가 계속 20 혹은 -20 (범위의 끝)으로 나온다면
    - max_shift 값을 늘려서 사용
    """
    best_ber = 1.0
    best_shift = 0
    
    # 지터 영향으로 앞뒤 비트가 깨질 수 있으므로 안정적인 중간 구간만 비교
    margin = 50
    test_rx = bits_rx[margin:-margin]
    
    for s in range(-max_shift, max_shift + 1):
        # s가 5일 때: rx[5 + margin : ...] 과 true[margin : ...] 비교
        start_idx = margin + s
        if start_idx < 0 or (start_idx + len(test_rx)) > len(bits_true):  # 유효한 범위 내에서만 비교하도록
            continue
            
        compare_rx = test_rx
        compare_true = bits_true[start_idx : start_idx + len(test_rx)]

        errors = np.sum(compare_true != compare_rx)
        ber = errors / len(compare_true)

        if ber < best_ber:
            best_ber = ber
            best_shift = s

        # Inversion 대응
        if best_ber > 0.5:
            best_ber = 1.0 - best_ber

    return best_ber, best_shift


def plot_eye(waveform, samples_per_ui, num_ui=2, num_traces=300,
            title="", 
    		sampling_points=None, # RX 샘플링 지점 시각화용
            show_clock=False,	# clock edge 수직선 표시
            lock_skip=3000      # lock될 때까지 버릴 bit수
			):

    span = num_ui * samples_per_ui
    t_eye = np.linspace(-num_ui/2, num_ui/2, span)
    
    # 전체 파형 기준이 아니라, 실제 사용 가능한 샘플링 포인트 개수를 기준으로 계산
    if sampling_points is not None:
        # CDR 결과 배열의 길이를 기준으로 가용 비트 수 설정
        n_bits_total = len(sampling_points)
    else:
        n_bits_total = len(waveform) // samples_per_ui

    # 샘플링할 비트 인덱스 범위를 [lock_skip, 끝]으로 제한
    if n_bits_total <= (lock_skip + num_ui + 1):
        print(f"Warning: Not enough bits ({n_bits_total}) for lock_skip ({lock_skip}).")
        return

    # lock_skip 이후의 비트 중에서 랜덤하게 traces를 선택
    bit_indices = np.random.randint(lock_skip, n_bits_total - num_ui, size=num_traces)

    for k in bit_indices:
		# Sampling point 기준으로 정렬
        if sampling_points is not None:
            # 실제 샘플링이 일어난 지점(Jittered Point)을 기준(0)으로 삼습니다.
            center = int(sampling_points[k])    # Clock jitter도 data에 투영
        else:
            # 샘플링 포인트 데이터가 없으면 중앙 계산
            center = k * samples_per_ui + samples_per_ui // 2

        seg_start = (center - span//2)
        seg_end = (center + span//2)
        # 파형 경계를 넘지 않도록 클리핑
        seg_start = max(0, seg_start)
        seg_end = min(len(waveform), seg_end)

        seg = waveform[seg_start : seg_end]

        if len(seg) == span: # span 길이만큼 정확히 잘렸을 때만 plotting
            plt.plot(t_eye, seg, color='b', alpha=0.1)
               
  	  		# 1) Sampling Point (빨간 점) - jittered_points가 있을 때만 표시
            if sampling_points is not None and show_clock:
				# 0 위치에서의 전압값 (정렬했으므로 x는 항상 0)
                sample_y = waveform[center]

                # 1) Sampling Point (빨간 점) - 항상 x=0 위치에 찍힘
                plt.plot(0, sample_y, 'ro', markersize=3, alpha=0.6)

                # 2) Clock Edge (수직선) - 옵션이 True일 때만 표시
                # 각 비트의 엣지를 얇은 선으로 표시 (겹쳐지면 진해짐)
                plt.axvline(x=0, color='r', linestyle='--', linewidth=0.5, alpha=0.1)
    
    # 전체 샘플링 위치를 대표하는 진한 수직선 하나 추가
    if show_clock:
        plt.axvline(x=0, color='r', linestyle='-', linewidth=1, label='Sampling Clock')

    plt.xlabel("UI")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
	# plt.show()