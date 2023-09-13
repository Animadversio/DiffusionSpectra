import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch


# https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
def get_spectrum(image):
    fourier_image = np.fft.fftn(image)
    npix = image.shape[0]
    fourier_amplitudes = np.abs(fourier_image)**2
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins
    
def plot_spectrum(kvals, Abins):
    plt.loglog(kvals, Abins)
    plt.xlabel("$k$")
    plt.ylabel("$P(k)$")
    plt.tight_layout()


# whitening like: https://arxiv.org/pdf/1806.08887.pdf
def get_r(u, v):
    return np.sqrt(u**2 + v**2)

def w1(u, v, coeff=0.75):
    """For 1/f images, coeff should be 1, adjust to obtain flat spectra.
    coeff=0.75 works well on Butterflies dataset."""
    return get_r(u, v) ** coeff

def w2(u, v, r0=48): # r0 ~ SNR
    """Adjust r0 according to the SNR of the data, noisier data needs lower r0.
    r0=48 works well on Butterflies dataset."""
    return np.exp(- (get_r(u, v) / r0) ** 4)


# some parameters:
npix = 128  # image size
kfreq = np.fft.fftfreq(npix) * npix
kfreq2D = np.meshgrid(kfreq, kfreq)
u = kfreq2D[0].copy()
v = kfreq2D[1].copy()
mask = w1(u, v) * w2(u, v)
mask[0, 0] = 1 # leave the DC component untouched

def whiten_image(image):
    img_white = image.copy()
    for c in range(3):
        # whiten channels separately
        fourier_image = np.fft.fftn(image[c])
        img_white[c] = np.fft.ifftn(fourier_image * mask).real.astype(np.float32)
    # rescale to same range as original
    img_white -= np.min(img_white)
    img_white /= np.max(img_white)
    img_white *= np.max(image) - np.min(image)
    img_white += np.min(image)
    return img_white

def transform_whiten(img):
    """Just add to datapipeline after transforms.ToTensor()"""
    img = img.detach().cpu().numpy()
    return torch.tensor(whiten_image(img))

def inverse_whitening(img_white):
    image = img_white.copy()
    for c in range(3):
        # whiten channels separately
        fourier_img_white = np.fft.fftn(img_white[c])
        image[c] = np.fft.ifftn(fourier_img_white / mask).real.astype(np.float32)
    # rescale to [-1, 1]
    image -= np.min(image)
    image /= np.max(image)
    image *= 2
    image -= 1
    return image


