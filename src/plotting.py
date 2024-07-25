import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from rdkit.Chem import Draw 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def gauss(f, x, x0, w):
    return f * np.exp(-((x - x0) ** 2) / (2 * (w / 2.355) ** 2))

def plot_spectrum(spectrum_df, filename='orca_tddft', figsize=(12, 8), x_axis='wavelength', plot_indiv=False, plot_overall=True, plot_stems=False, show_mol=None):
    states = spectrum_df['State'].tolist()
    energies = spectrum_df['Energy'].tolist()
    wavelengths = spectrum_df['Wavelength'].tolist()
    fosc = spectrum_df['fosc'].tolist()

    if x_axis == 'wavelength':
        x_data = wavelengths
        x_label = 'Wavelength $\lambda$ (nm)'
        w = 10.0
    elif x_axis == 'wavenumber':
        x_data = energies
        x_label = 'Energy (cm$^{-1}$)'
        w = 1000.0

    fig, ax = plt.subplots(figsize=figsize)
    x = np.linspace(200, max(x_data) + (w * 3), 5000)
    gauss_sum = []

    # Plot the individual Gaussian curves for each state in the data
    # showing the contribution of each state to the overall spectrum
    for index, X in enumerate(x_data):
        G = gauss(fosc[index], x, X, w)
        if plot_indiv:
            ax.plot(x, G, color="cornflowerblue", alpha=0.2, zorder=10)
        gauss_sum.append(G)

    # Sum the Gaussian curves to get the overall absorption spectrum
    y_gauss_sum = np.sum(gauss_sum, axis=0)
    peaks, _ = find_peaks(y_gauss_sum, height=0)
    # Plot the overall absorption spectrum
    if plot_overall:
        ax.plot(x, y_gauss_sum, color='darkblue', linewidth=2, zorder=10)
    
    # A stem at each state
    if plot_stems:
        ax.stem(x_data, fosc, linefmt='dimgrey', markerfmt=" ", basefmt=" ", zorder=10)

    for index, txt in enumerate(peaks):
        ax.annotate(
            int(x[peaks[index]]),
            xy=(x[peaks[index]], y_gauss_sum[peaks[index]]),
            ha="center",
            rotation=0,
            size=12,
            xytext=(0, 5),
            textcoords='offset points',
            zorder=10
        )
        
    ax.set_xlabel(x_label)
    ax.set_ylabel('Intensity')
    ax.set_title('Absorption Spectrum', fontsize=16, fontweight='bold')
    ax.get_yaxis().set_ticks([])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    if show_mol:
        img = Draw.MolToImage(show_mol, size=(600, 600))
        ax_inset = inset_axes(ax, width="65%", height="65%", loc='upper left')
        ax_inset.imshow(img, zorder=1)
        ax_inset.axis('off')
    
    plt.savefig(f"{filename}-abs.png", dpi=300)
    plt.show()
    
    
    return fig