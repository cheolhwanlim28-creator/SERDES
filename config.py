class Config:
    def __init__(self):

        # ==================================================
        # GLOBAL
        # ==================================================
        self.global_cfg = GlobalConfig()

        # ==================================================
        # TX Clock
        # ==================================================
        self.txclock = ClockConfig(self.global_cfg)

        # ==================================================
        # TX
        # ==================================================
        self.tx = TXConfig(self.global_cfg)

        # ==================================================
        # CHANNEL
        # ==================================================
        self.channel = ChannelConfig(self.global_cfg)

        # ==================================================
        # RX
        # ==================================================
        self.rx = RXConfig(self.global_cfg)


# ==================================================
# GLOBAL
# ==================================================
class GlobalConfig:
    def __init__(self):
        self.bit_rate = 20e9  # 20Gbps
        self.samples_per_ui = 64  # Resolution
        self.ui = 1 / self.bit_rate
        self.fs = self.samples_per_ui * self.bit_rate  # Sampling Frequency
        self.prbs_order = 7
        self.refclk_freq = 24e6 # Reference Clock

        # global supply jitter
        self.psij_amp = 10      # mV
        self.psij_freq = 20e6

# ==================================================
# TX CLOCK
#
# clock jitter source 대부분 = VCO phase noise
# Reference clock jitter  << 매우 작음
# PLL VCO phase noise     << 대부분
# Clock buffer noise      << 약간
# DCC noise               << 거의 없음

# ReferenceClock : 거의 ideal
# PLL            : phase noise source
# CD             : supply coupling
# DCC            : deterministic distortion
# ==================================================

class ClockConfig:
    def __init__(self, global_cfg):
        self.refclk = RefClockConfig(global_cfg)
        self.pll = PLLConfig(global_cfg)
        self.cd = CDConfig(global_cfg)
        self.dcc = DCCConfig(global_cfg)

class RefClockConfig:
    def __init__(self, global_cfg):
        self.freq = global_cfg.refclk_freq
        self.RJ_REF_CLK = 0     # Reference Clock의 자체 RJ (ps rms) -> budgeting용으로 0, PLL에 RJ합산
        self.KVDD_REF_CLK = 0   # Supply sensitivity, unit: ps/mV

class PLLConfig:
    def __init__(self, global_cfg):
        self.N = global_cfg.bit_rate / (2 * global_cfg.refclk_freq)

        # PFD + CP
        self.Icp = 1        # Charge Pump 전류 -> budgeting용으로 phase detector gain = 1

        # Loop filter design
        self.KLF = 1        # loop filter transimpedance gain -> budgeting용으로 1
        self.zero = 5e6
        self.pole = 50e6

        # VCO
        self.f_center = global_cfg.bit_rate / 2
        self.KVCO = 1       # VCO Gain (Hz/V) -> budgeting용으로 1

        # supply sensitivity
        # unit: ps/mV
        self.KVDD_LOOP = 0.3    # LF/CP 쪽 전원 민감도 (ps/mV)
        self.KVDD_VCO = 0.6     # VCO 전원 민감도 (ps/mV)

        # # Phase noise spec (dBc/Hz)
        # self.pn_offset = 1e6
        # self.pn_dbc = -100

        # Random jitter (RMS)
        # unit: UI
        # self.RJ_VCO = 0 
        self.RJ_PLL = 0 

class CDConfig:
    def __init__(self, global_cfg):

        # Random jitter (RMS)
        # unit: UI
        self.RJ_CD = 0 

        # Supply sensitivity
        # unit: ps/mV
        self.KVDD_CD = 0    # CD LDO 사용

        self.cd_trf_ui = 0.2   # Rise/Fall Time (20% to 80%, UI 단위)

class DCCConfig:
    def __init__(self, global_cfg):

        # static duty distortion
        # unit: UI
        self.dcd = 0.002

        # random duty noise
        # unit: UI rms
        self.RJ_DCC = 0

        # Supply sensitivity
        # unit: ps/mV
        self.KVDD_DCC = 0.2


# ==================================================
# TX
# ==================================================

class TXConfig:
    def __init__(self, global_cfg):
        self.ffe = FFEConfig()
        self.predriver = TXPreDriverConfig(global_cfg)
        self.driver = TXDriverConfig()

class FFEConfig:
    def __init__(self):
        self.pre_level = 0      # 0, 1, 2, 3
        self.post_level = 0     # 0, 1, 2, 3

class TXPreDriverConfig:
    def __init__(self, global_cfg):

        # transition based DDJ
        # unit: UI
        self.ddj = 0.002

        # Random jitter
        # (optional – 보통 0으로 둠)
        self.rj = 0.0

        # Supply sensitivity
        # unit: ps/mV
        self.KVDD_PREDRV = 0.2

        # Resampling
        self.spui = global_cfg.samples_per_ui
        self.fs = global_cfg.fs

class TXDriverConfig:
    def __init__(self):
        self.swing = 0.45   # Single-ended Swing (V)
        self.txdrv_trf_ui = 0.2   # Rise/Fall Time (20% to 80%, UI 단위)


# ==================================================
# CHANNEL
# ==================================================
class ChannelConfig:
    def __init__(self, global_cfg):
        self.type = "rc"   # "rc" | "fir" | "butter"
        self.length_ui = 10  # ISI를 충분히 보기 위해 10UI로 확장
        self.fc = 4e9       # 20G 시스템에서 4G fc는 꽤 도전적인 채널 (DFE가 활약하기 좋음)
        self.bw = 5e9

        # GLOBAL 참조 필요 값은 계산 기반으로 연결
        self.spui = global_cfg.samples_per_ui
        self.bit_rate = global_cfg.bit_rate
        self.fs = global_cfg.fs


# ==================================================
# RX
# ==================================================
class RXConfig:
    def __init__(self, global_cfg):
        # AFE (Analog Front-End): CTLE, VGA, AGC를 포함하는 아날로그 덩어리
        self.frontend = FrontendConfig(global_cfg) 
        self.cdr = CDRConfig()

        # frontend에 정의된 slicer 사양을 DFE에 그대로 전달
        self.dfe = DFEConfig(
            slicer_offset=self.frontend.slicer_offset,
            slicer_sensitivity=self.frontend.slicer_sensitivity
        )

# -------- Frontend --------
class FrontendConfig:
    def __init__(self, global_cfg):
        self.channel_out_noise = 0.001   # 채널 끝단 열잡음 (V rms)
        self.xtalk_rms = 0.005           # Crosstalk (V rms)

        # CTLE 설정 (객체로 포함 - 주파수 파라미터가 많으므로)
        self.ctle = CTLEConfig(global_cfg)

        # VGA & AGC
        self.vga_gain_db = 10        # 초기/고정 Gain (dB)
        self.vga_irn_rms = 0.002     # VGA Input Referred Noise (V rms)
        self.agc_v_target = 0.250    # AGC 목표 Peak 진폭 (V) - Slicer 입력 최적화용
        self.agc_mu = 0.005          # AGC Gain 업데이트 속도 (Step size)  

        # Slicer (Hard-decision Block) 비이상성 (나중에 DFE가 가져다 쓸 값)
        self.slicer_offset = 0.005    # Slicer DC Offset (V)
        self.slicer_sensitivity = 0.003 # Slicer Sensitivity (Dead-band) (V)

# -------- CTLE --------
class CTLEConfig:
    def __init__(self, global_cfg):
        self.peak_freq = 10e9			# Target Peak Frequency (10GHz)
        self.p1 = 7.5e9					# 첫 번째 Pole (Boost 시작점 근처)
        self.p2 = 14.35e9				# 두 번째 Pole (고주파 노이즈 차단용)
        self.boost_list = [0, 3, 6, 9, 12]	# Gain Settings (dB)
        self.selected_boost = 6				# 현재 선택된 Boost 값
        self.fs = global_cfg.fs
        self.ctle_orn_rms = 0.001        # CTLE 자체 발생 잡음 (V rms)

# -------- CDR --------
# "perfect": 지터를 100% 추적 (Infinite BW CDR)
# "static":  고정된 평균 위상에서 샘플링 (No Tracking)
# "real":    설계한 CDR 루프(PI)가 직접 추적 (Finite BW CDR)
class CDRConfig:
    def __init__(self):
        self.mode = "perfect"  # perfect | static | real
        self.damping = 0.94
        self.f_3db = 20e6
        # self.fn = 8.37e6	# Defined in the Spec., but must be calculated using other parameters.
        self.pi_resolution_bits = 6   # PI 해상도 (64 steps per UI)
        self.rx_clock_rj = 0.01   # RX VCO Phase Noise (UI rms)
        self.tps1_len = 20000
        self.CDR_offset = 1.5

# -------- DFE --------
class DFEConfig:
    def __init__(self, slicer_offset=0.005, slicer_sensitivity=0.003):
        self.enable = True
        self.adaptation_on = True
        self.n_taps = 2		# 현재는 2-tap 구조
        self.threshold = 0.0

        # 회로 비이상성 파라미터 상속 (FrontendConfig에서 가져와서 사용)
        self.slicer_offset = slicer_offset 
        self.slicer_sensitivity = slicer_sensitivity

        self.mu = 0.0001	# Step size, 하드웨어에서 매우 큰 누적기(카운터)를 써서 탭이 아주 천천히, 신중하게 움직이도록 만든 것과 수학적으로 동일한 효과
        self.lsb = 0.0011	# 실제 하드웨어의 DAC 해상도 설정 (예: 1.1mV LSB)
        self.v_target = 0.050   # DFE Adaptation 목표 높이

        self.tap_max = [0.070, 0.040]
        self.tps2_len = 50000