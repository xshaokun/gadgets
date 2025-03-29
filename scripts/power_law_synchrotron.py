import matplotlib.pyplot as plt
import numpy as np
import unyt as u
from xsk_pytools.radiation import PowerLawCR, Synchrotron

# fiducial CRes power-law distribution
tote = u.unyt_quantity(3.5e-10, "erg/cm**3")
cr_jet = PowerLawCR(2.4, 2000, 2e6, total_energy=tote)
sync_jet = Synchrotron(30 * u.uG, cr_jet)
# with weaker magnetic field
sync_jet_weak_b = Synchrotron(10 * u.uG, cr_jet)

# with lower total energy
cr_jet_low_e = PowerLawCR(2.4, 2000, 2e6, total_energy=0.01 * tote)
sync_jet_low_e = Synchrotron(30 * u.uG, cr_jet_low_e)

# with weaker lower and upper energy of CRes
cr_thermal = PowerLawCR(2.4, 1, 1e4, total_energy=tote)
sync_thermal = Synchrotron(30 * u.uG, cr_thermal)

# generate emissivity array for given energy band
energy = np.logspace(-10, 3, 1000) * u.eV
spec_jet = sync_jet.spectrum(energy.to_equivalent("Hz", "spectral"))
spec_jet_weak_b = sync_jet_weak_b.spectrum(energy.to_equivalent("Hz", "spectral"))
spec_jet_low_e = sync_jet_low_e.spectrum(energy.to_equivalent("Hz", "spectral"))
spec_thermal = sync_thermal.spectrum(energy.to_equivalent("Hz", "spectral"))

plt.rcParams["text.usetex"] = True

fig = plt.figure()
ax = fig.add_subplot(111)

# fiducial spectrum of power-law distributed CRes
(jet,) = ax.plot(
    energy,
    spec_jet,
    label=r"$2000<\gamma<2\times10^6$,$B=30\;\mathrm{\mu G}$",
    color="b",
)
# label the critical frequency from the lower and upper limit of CRes
ax.vlines(
    sync_jet.freq_c(cr_jet.g_min).to_equivalent("eV", "spectral"),
    0,
    1,
    linestyles="-.",
    colors="b",
)
ax.vlines(
    sync_jet.freq_c(cr_jet.g_max).to_equivalent("eV", "spectral"),
    0,
    1,
    linestyles="-.",
    colors="b",
)

# spectrum with weaker magnetic field
(jet_weak_b,) = ax.plot(
    energy, spec_jet_weak_b, label=r"$B=10\;\mathrm{\mu G}$", color="r"
)
ax.vlines(
    sync_jet_weak_b.freq_c(cr_jet.g_min).to_equivalent("eV", "spectral"),
    0,
    1,
    linestyles="-.",
    colors="r",
)
ax.vlines(
    sync_jet_weak_b.freq_c(cr_jet.g_max).to_equivalent("eV", "spectral"),
    0,
    1,
    linestyles="-.",
    colors="r",
)
ax.text(10, 1e-38, r"$F_\mathrm\nu \propto B^{(p+1)/2}$", color="r")
ax.text(10, 3e-39, r"$\nu_\mathrm c \propto B$", color="r")

# spectrum with lower CRe energy density
(jet_low_e,) = ax.plot(energy, spec_jet_low_e, label=r"1\% $e_\mathrm{CR}$", color="c")
ax.text(1e-9, 1e-37, r"$F_\mathrm\nu \propto e_\mathrm{CR}$", color="c")

# spectrum with less powerful distributed CRes
(thermal,) = ax.plot(energy, spec_thermal, label=r"$1<\gamma<10^4$", color="g")
ax.text(5e-6, 5e-39, r"$\nu_\mathrm c \propto \gamma^2$", color="g")

ax.set_xscale("log")
ax.set_yscale("log")
smax = (spec_jet).max()
ax.set_ylim(1e-10 * smax, 10 * smax)
ax.set_xlabel(r"$E$ (eV)")
ax.set_ylabel(r"$F_\nu$ ($\mathrm{erg/s/Hz/cm^3}$)")


# setting upper x-axis showing frequency
def forward_transform(x):
    return (x * u.eV).to_equivalent("Hz", "spectral")


def inverse_transform(x):
    return (x * u.Hz).to_equivalent("eV", "spectral")


top_ax = ax.secondary_xaxis("top", functions=(forward_transform, inverse_transform))
top_ax.set_xlabel(r"$\nu$ (Hz)")


# setting second upper x-axis showing wavelength
def forward_transform(x):
    return (x * u.eV).to_equivalent("cm", "spectral")


def inverse_transform(x):
    return (x * u.cm).to_equivalent("eV", "spectral")


top_ax = ax.secondary_xaxis("top", functions=(forward_transform, inverse_transform))
top_ax.spines["top"].set_position(("outward", 35))
top_ax.set_xlabel(r"$\lambda$ (cm)")


# setting right y-axis showing with units of mJansky/pc
def forward_transform(x):
    return (u.unyt_array(x, "erg/s/Hz/cm**3")).to("mJy/pc")


def inverse_transform(x):
    return (u.unyt_array(x, "mJy/pc")).to("erg/s/Hz/cm**3")


right_ax = ax.secondary_yaxis("right", functions=(forward_transform, inverse_transform))
right_ax.set_ylabel(r"$F_\nu$ ($\mathrm{mJy/pc}$)")

# legend showing different lines
legend1 = fig.add_axes([0.23, 0.16, 0.2, 0.1])
legend1.axis("off")
legend1.legend(handles=[jet, jet_weak_b, jet_low_e, thermal], loc="center")

# showing the slope of lower energy part of spectrum
ref_x1 = np.logspace(-10, -8, 2)
ref_y1 = 7e-35 * (ref_x1 / 1e-9) ** (1 / 3)
ax.plot(ref_x1, ref_y1, color="black")
ax.text(1e-10, 5e-35, r"$\propto \nu^{1/3}$", rotation=25)

# showing the slope of medium energy part of spectrum
index = 2.4
ref_x2 = np.logspace(-4, -2, 2)
ref_y2 = 7e-35 * (ref_x2 / 1e-4) ** (-(index - 1) / 2)
ax.plot(ref_x2, ref_y2, color="black")
ax.text(1e-4, 4e-36, r"$\propto \nu^{-(p -1)/2}$", rotation=-42)

# showing the slope of higher energy part of spectrum
xc_max = sync_jet.freq_c(cr_jet.g_max).to_equivalent("eV", "spectral")

ref_x3 = np.logspace(np.log(xc_max) + 0.3, 1.8 + 0.3, 10)
me0 = (u.physical_constants.me_cgs * u.physical_constants.c_cgs**2).to_value("eV")
ref_x3x = (ref_x3 - 2e1) / xc_max
ref_y3 = 1e-40 * np.exp(-ref_x3x) * (ref_x3x) ** 0.5
ax.plot(ref_x3, ref_y3, color="black")
ax.text(4e1, 1e-42, r"$\propto (\nu/\nu_c)^{1/2} e^{-\nu/\nu_c}$")

# label the bands
span1 = ax.axvspan(0, 1e-3, color="grey", alpha=0.3, label="Radio")
span2 = ax.axvspan(1e-2, 1.7, color="red", alpha=0.3, label="IR")
span3 = ax.axvspan(3.1, 124, color="purple", alpha=0.3, label="UV")
span4 = ax.axvspan(124, 124 * 1000, color="blue", alpha=0.3, label="Xray")

# legend showing different bands
legend2 = fig.add_axes([0.65, 0.7, 0.3, 0.1])
legend2.axis("off")
legend2.legend(handles=[span1, span2, span3, span4], loc="center")

ax.set_title(
    r"CRes $\mathrm{d}N/\mathrm{d}\gamma\propto \gamma^{-p} \propto \gamma^{-2.4} (e_\mathrm{CR}=3.5\times 10^{-10}$ erg/cm$^3$)"
)
fig.savefig("power_law_synchrotron_spectra.png", bbox_inches="tight", dpi=300)
