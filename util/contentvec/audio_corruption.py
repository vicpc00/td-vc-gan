import numpy as np
import scipy.signal as sps
import util.contentvec.audio_utils as audio_utils

Qmin, Qmax = 2, 5
Fc = np.exp(np.linspace(np.log(60), np.log(7600), 10))
def random_eq(wav, sr):
    z = np.random.uniform(0, 1, size=(10,))
    Q = Qmin * (Qmax / Qmin)**z
    G = np.random.uniform(-12, 12, size=(10,))
    sos = audio_utils.params2sos(G, Fc, Q, sr)
    wav = sps.sosfilt(sos, wav)
    return wav

def random_formant_f0(wav, sr, f0_l0 = 60, f0_hi = 600):

    
    ratio_fs = np.random.uniform(1, 1.4)
    coin = (np.random.random() > 0.5)
    ratio_fs = coin*ratio_fs + (1-coin)*(1/ratio_fs)
    
    ratio_ps = np.random.uniform(1, 2)
    coin = (np.random.random() > 0.5)
    ratio_ps = coin*ratio_ps + (1-coin)*(1/ratio_ps)
    
    ratio_pr = np.random.uniform(1, 1.5)
    coin = (np.random.random() > 0.5)
    ratio_pr = coin*ratio_pr + (1-coin)*(1/ratio_pr)
    
    ss = audio_utils.change_gender(wav, sr, f0_l0, f0_hi, ratio_fs, ratio_ps, ratio_pr)
    
    return ss

if __name__ == '__main__':
    import sys
    import soundfile as sf

    in_fn = sys.argv[1]
    out_fn = sys.argv[2]
    
    s, sr = sf.read(in_fn)
    
    s = random_formant_f0(s, sr)
    s = random_eq(s, sr)
    
    sf.write(out_fn, s, sr)
    
    