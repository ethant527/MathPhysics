
"""
Quantum Vacuum Fluctuations — 3D Isosurface Visualiser

Physics
  τ  = ℏ / (2E)                 pair lifetime  (Heisenberg ΔE·Δt ≥ ℏ/2)
  P(E) ∝ E·√(E²−m²)             relativistic density-of-states spectrum
  amp = sin(π·t/τ)·(0.6+E)      envelope → zero at birth and annihilation
  weight = exp(−π·E/E_MAX)      Schwinger-inspired creation suppression

Field
  Spike grid accumulated per particle/antiparticle position, then
  smoothed with a 3-D Gaussian filter (scipy) — two blobs per pair
  giving the dipole structure of vacuum polarisation.
  Isosurface extracted with marching-cubes; faces coloured by local
  field value (teal body → yellow-green hotspot).

Fermionic exclusion
  A coarse occupancy grid prevents two pairs nucleating in the same
  voxel simultaneously, approximating Pauli exclusion.

"""

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


# PARAMETERS  (all distances in "box units" where BOX = 36)
HBAR          = 1.0
BOX           = 36.0        # side length; axes tick at 0, 12, 24, 36
CREATION_RATE = 50          # mean new pairs per simulated second
E_MAX         = 20          # Schwinger suppression scale / UV reference - i have changed it from 3 originally
MAX_PAIRS     = 99999       # hard cap on simultaneous pairs
DT            = 0.05        # simulated seconds per animation frame - i have changed it from 0.06 originally
WARMUP_STEPS  = 220         # pre-run steps so vacuum is full at t=0
GRID_N        = 40          # voxels per axis  (40³ is fast yet smooth)
SIGMA         = 3.0         # Gaussian blob radius in BOX units
ISO_FRAC      = 0.1         # isosurface at this fraction of field max
FACE_ALPHA    = 0.90        # blob opacity
ROTATE        = True
ELEV          = 26
AZIM_START    = 210
AZIM_SPEED    = 0.28        # degrees per frame


# electron rest-mass energy and UV scale for energy sampling (natural units)
ELECTRON_MASS = 0.511       # MeV (in natural units where c=1)
KT_UV         = 3.0         # exponential scale for proposal distribution


# colourmap
BLOB_CMAP = LinearSegmentedColormap.from_list(
    'vacuum',
    [
        (0.00, '#004f5a'),
        (0.22, '#00b5aa'),
        (0.50, '#00dfa0'),
        (0.75, '#8fff00'),
        (1.00, '#d0ff00'),
    ]
)


# pre-compute the sigma in voxel units for the scipy filter (used every frame)
_SIGMA_VOXELS = SIGMA * GRID_N / BOX



# one virtual particle–antiparticle pair, with energy E, lifetime τ, and dipole structure
class VirtualPair:

    def __init__(self):

        self.E   = VirtualPair.sample_energy()
        self.tau = HBAR / (2.0 * self.E)
        self.t   = 0.0

        # Schwinger-inspired suppression: heavy/high-E pairs are rare
        # weight ∈ (0, 1]; used in VacuumField.step as accept probability
        self.weight = np.exp(-np.pi * self.E/E_MAX)

        # dipole structure: particle and antiparticle positions
        centre = np.random.uniform(0.08 * BOX, 0.92 * BOX, 3)
        self.centre = centre # self.centre is stored so VacuumField can find the pair's midpoint for occupancy checks and flash positions without re-computing it
               

        # Compton-like half-separation ∝ ℏ/(mc) scaled to box coords;
        # high-energy pairs are more tightly localised
        compton  = HBAR / (self.E * BOX)
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        half_sep  = compton * direction

        self.pos_particle     = centre + half_sep
        self.pos_antiparticle = centre - half_sep

    # energy sampling
    @staticmethod
    def sample_energy(m=ELECTRON_MASS, kT_uv=KT_UV):

        """""
        Rejection-sample P(E) ∝ E·√(E²−m²)  for E > m.

        This is the relativistic density of states: d³k ∝ k² dk and
        E² = (pc)² + (mc²)², so dN/dE ∝ E·√(E²−m²).
        """
        
        E_cap = m + 6.0 * kT_uv          # practical maximum (~99.75 % coverage) ####????
        upper = E_cap ** 2                # true upper bound since √(E²−m²) < E
        while True:
            E = m + np.random.exponential(kT_uv)
            if E > E_cap:
                continue                  # discard the rare extreme tail
            p_val = E * np.sqrt(max(E ** 2 - m ** 2, 0.0))
            if np.random.uniform(0.0, upper) < p_val:
                return E

    # time evolution
    @property
    def progress(self):
        return min(self.t / self.tau, 1.0)

    @property
    def amplitude(self):
        #sin-envelope × energy — zero at birth and at annihilation 
        return np.sin(np.pi * self.progress) * (0.6 + self.E)

    def step(self, dt):
        self.t += dt
        return self.t < self.tau # false → annihilated



class VacuumField:

    def __init__(self):

        self.pairs         = []
        self.flashes       = [] # list of (centre_3d, time_to_live, energy)
        self.n_created     = 0
        self.n_annihilated = 0

        # Occupancy grid for fermionic exclusion (BUG FIX 5)
        self.occupancy = np.zeros((GRID_N, GRID_N, GRID_N), dtype=np.int8)

    # fermionic exclusion
    def _voxel(self, pos):

        """Convert a BOX-space position to a voxel index tuple."""
        return tuple(
            np.clip((pos / BOX * GRID_N).astype(int), 0, GRID_N - 1)
        )

    def can_create(self, pos):

        # True if the voxel at pos is not already occupied
        return self.occupancy[self._voxel(pos)] < 1


    # rebuild the fermionic occupancy grid from current live pairs
    def _rebuild_occupancy(self):

        self.occupancy[:] = 0
        for p in self.pairs:
            self.occupancy[self._voxel(p.centre)] = 1

    # time step
    def step(self, dt):

        self._rebuild_occupancy()

        n_attempts = np.random.poisson(CREATION_RATE * dt)

        for _ in range(n_attempts):

            if len(self.pairs) >= MAX_PAIRS:
                break
            candidate = VirtualPair()

            # Schwinger acceptance: high-E pairs suppressed
            if np.random.random() > candidate.weight:
                continue

            # fermionic exclusion
            if not self.can_create(candidate.centre):
                continue

            self.pairs.append(candidate)
            self.occupancy[self._voxel(candidate.centre)] = 1
            self.n_created += 1

        alive, dead = [], []

        for p in self.pairs:
            (alive if p.step(dt) else dead).append(p)

        for p in dead:

            # uses p.centre for pair midpoint
            self.flashes.append((p.centre.copy(), 0.28, p.E))
            self.n_annihilated += 1

        self.pairs  = alive
        self.flashes = [(o, t - dt, e) for o, t, e in self.flashes if t > dt]

    def warmup(self):

        for _ in range(WARMUP_STEPS):
            self.step(DT)

    # field
    def build_field(self):

        """
        Build the 3-D energy-density field via spike accumulation +
        Gaussian smoothing (scipy.ndimage.gaussian_filter).
        """

        spike_field = np.zeros((GRID_N, GRID_N, GRID_N), dtype=np.float32)

        # live pairs: two spikes per pair (dipole)
        for p in self.pairs:

            amp = p.amplitude # zero at birth and annihilation

            if amp < 0.03:
                continue

            for pos in (p.pos_particle, p.pos_antiparticle):

                ix, iy, iz = self._voxel(pos)
                spike_field[ix, iy, iz] += amp


        # annihilation flashes: single bright spike 
        for pos, ttl, e in self.flashes:
            amp = 9.0 * (ttl / 0.28) * (1.0 + e)
            ix, iy, iz = self._voxel(pos)
            spike_field[ix, iy, iz] += amp

        # gaussian smoothing: converts point spikes to smooth energy blobs
        # Flash blobs are wider — achieved by a second, broader filter pass. what does this mean ?????????
        field_pairs  = scipy.ndimage.gaussian_filter(spike_field, sigma=_SIGMA_VOXELS)

        # build flash-only spike grid for the wider smoothing
        flash_spikes = np.zeros_like(spike_field)

        for pos, ttl, e in self.flashes:

            amp = 9.0 * (ttl / 0.28) * (1.0 + e)
            ix, iy, iz = self._voxel(pos)
            flash_spikes[ix, iy, iz] += amp

        field_flashes = scipy.ndimage.gaussian_filter(flash_spikes, sigma=_SIGMA_VOXELS * np.sqrt(3.0))

        return field_pairs + field_flashes



class Figure:

    @staticmethod
    def build_figure():
        fig = plt.figure(figsize=(10, 9), facecolor='black')
        ax = fig.add_axes([0.01, 0.01, 0.98, 0.98], projection='3d')
        ax.set_facecolor('black') # looks best with black or white

        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):

            pane.fill = False
            pane.set_edgecolor('black')
            pane.set_alpha(1.0)

        ax.grid(True, color='white', linewidth=0.30, alpha=0.50)

        ticks = [0, 12, 24, 36]
        ax.set_xlim(0, BOX); ax.set_ylim(0, BOX); ax.set_zlim(0, BOX)
        ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_zticks(ticks)
        ax.tick_params(colors='grey', labelsize=7, pad=1)
        ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
        ax.set_box_aspect([1, 1, 1])

        return fig, ax


class Animate:

    @staticmethod
    def run():
        print(f'Warming up vacuum ({WARMUP_STEPS} steps)...')
        vac = VacuumField()
        vac.warmup()
        print(f'Ready — {len(vac.pairs)} active pairs | {vac.n_created} created')

        fig, ax = Figure.build_figure()

        def update(frame):
            vac.step(DT)

            for coll in ax.collections[:]:
                coll.remove()

            field = vac.build_field()
            fmax  = field.max()

            if frame % 20 == 0:
                print(f'frame {frame:4d} | pairs {len(vac.pairs):3d} '
                      f'| field max {fmax:.3f} '
                      f'| annihilations {vac.n_annihilated}')

            if fmax > 1e-3: # what does this do ?????
                level = ISO_FRAC * fmax
                try:
                    verts, faces, normals, _ = marching_cubes(
                        field,
                        level=level,
                        spacing=(BOX / GRID_N,) * 3,
                    )


                    centroids = verts[faces].mean(axis=1)
                    ci        = np.clip(
                        (centroids / (BOX / GRID_N)).astype(int), 0, GRID_N - 1
                    )
                    fvals = field[ci[:, 0], ci[:, 1], ci[:, 2]]
                    lo, hi = np.percentile(fvals, 1), np.percentile(fvals, 99)
                    fnorm  = np.clip((fvals - lo) / (hi - lo + 1e-9), 0.0, 1.0)

                    face_rgba       = BLOB_CMAP(fnorm).copy()
                    face_rgba[:, 3] = FACE_ALPHA


                    light_dir = np.array([0.6, 0.4, 0.7])
                    light_dir /= np.linalg.norm(light_dir)
                    face_normals = normals[faces].mean(axis=1)
                    nlen = np.linalg.norm(face_normals, axis=1, keepdims=True)
                    face_normals /= np.where(nlen > 0, nlen, 1.0)
                    diffuse = np.clip(face_normals @ light_dir, 0.0, 1.0)
                    shade   = 0.35 + 0.65 * diffuse
                    face_rgba[:, :3] *= shade[:, np.newaxis]

                    mesh = Poly3DCollection(
                        verts[faces],
                        facecolors=face_rgba,
                        edgecolors='none',
                        shade=False,
                    )

                    ax.add_collection3d(mesh)

                except Exception as exc: # what does this do ??????
                    print(f'marching_cubes skipped: {exc}')

            ax.set_xlim(0, BOX); ax.set_ylim(0, BOX); ax.set_zlim(0, BOX)

            if ROTATE:
                ax.view_init(elev=ELEV, azim=(AZIM_START + frame * AZIM_SPEED) % 360,)

        ani = animation.FuncAnimation(  # noqa: F841 (keeps reference alive)
            fig, update,
            frames=1200,
            interval=int(DT * 1000),
            blit=False,
        )

        plt.show()


if __name__ == '__main__':
    Animate.run()

