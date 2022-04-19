#!/usr/bin/env python3

"""Kolmogorov Flow with a custom operator.
Output files are stored in output/."""

import cubismup2d as cup2d
import numpy as np
import argparse


class CustomOperator(cup2d.Operator):
    def __init__(self, sim):
        super().__init__(sim)
        # Get number of wavenumbers
        self.Nfreq = self.sim.cells[0]//2
        self.energySpectrum = np.zeros((self.Nfreq,self.Nfreq))
        self.done = False

    def __call__(self, dt: float):
        data: cup2d.SimulationData = self.sim.data

        timeStart = 10.0
        timeEnd = 50.0
        
        # Skip transient region
        if (data.time > timeStart) and (not self.done):
            # Get the whole field as a large uniform matrix
            # Note that the order of axes is [y, x], not [x, y]!
            vel = data.vel.to_uniform()
            N = vel.shape[0]

            # print("Field:", vel, vel.shape)

            # Separate Field into x- and y-velocity and change order of axis
            u = vel[:,:,0].transpose()
            v = vel[:,:,1].transpose()

            # print("Velocities:", u, v, u.shape, v.shape)

            # Perform Fourier Transform on Fields
            Fu = np.fft.fft2(u)
            Fv = np.fft.fft2(v)

            # print("Transformed Velocities:", Fu, Fv, Fu.shape, Fv.shape )

            # Compute Energy; Note that for real numbers the fourier transform is symmetric, so only half of the spectrum needed
            factor = 1 / ( 2 * N * N )
            energy = factor * np.real( np.conj(Fu)*Fu + np.conj(Fv)*Fv )
            energy = energy[:N//2, :N//2]

            # print("Computed Energies:", energy, energy.shape )

            # Compute temporal average of energy; Careful dt is not constant, so not simple average!
            self.energySpectrum += energy*dt

            # Finalize Spectrum at end of averaging window
            if (data.time >= timeEnd):
                self.done = True

                # Divide by Integration-Horizont and flatten for further processing
                energy = self.energySpectrum/(timeEnd-timeStart)
                energy = energy.flatten()

                # print("Average Energies:", energy, energy.shape )

                # Get Wavenumbers; Note that for real numbers the fourier transform is symmetric, so only half of the spectrum needed
                h = 2*np.pi / N
                freq = np.fft.fftfreq(N,h)[:N//2]

                # Create Flattened Vector with absolute values for Wavenumbers
                kx, ky = np.meshgrid(freq, freq)
                k = np.sqrt(kx**2 + ky**2)
                k = k.flatten()

                # Detect and remove dublicates from k and average energies
                folded, indices, counts = np.unique(k, return_inverse=True, return_counts=True)
                output = np.zeros((folded.shape[0], ))
                np.add.at(output, indices, energy)
                output /= counts

                # Perform (k+dk)-wise integration )
                dk = freq[1]
                wavenumbers = np.arange(0, folded[-1]+2*dk, dk )
                averagedEnergySpectrum = np.zeros_like(wavenumbers)
                currWavenumberIndex = 0
                numPerWavenumber = 1
                for i, _k in enumerate(wavenumbers[:-1]):
                    next_k = wavenumbers[ i + 1 ]
                    mid_k = (next_k-k)/2
                    indices = (_k <= k) & (k < next_k)
                    averagedEnergySpectrum[i] = np.mean(energy[indices]/k[indices])

                #### Save Energy Spectrum
                np.savetxt("EnergySpectrum_N={}_Cs={}.out".format(N,data.Cs), (wavenumbers,averagedEnergySpectrum))

parser = argparse.ArgumentParser()
parser.add_argument('--N', help='Number of Gridpoints per Dimension.', required=True, type=int)
parser.add_argument('--Cs', help='Smagorinsky Model constant Cs', required=True, type=float)
args = parser.parse_args()

sim = cup2d.Simulation(cells=(args.N, args.N), nlevels=1, start_level=0,
                       extent=2.0*np.pi, tdump=0.0, ic="random",
                       bForcing=1, output_dir="./", cuda=True, Cs=args.Cs)
sim.init()
if args.Cs == 0:
    sim.insert_operator(CustomOperator(sim), after='advDiff')
else:
    sim.insert_operator(CustomOperator(sim), after='advDiffSGS')
sim.simulate(tend=50.1)
