import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

class Blisk:
    """
    Class to represent a Blisk (Blade Integrated Disk) for aeroelastic analysis.
    """
    def __init__(self, num_blades=20, blade_length=0.1, disk_radius=0.5):
        self.num_blades = num_blades
        self.blade_length = blade_length
        self.disk_radius = disk_radius
        self.blade_positions = self.compute_blade_positions()

    def compute_blade_positions(self):
        """Compute the angular positions of the blades on the disk."""
        angles = np.linspace(0, 2 * np.pi, self.num_blades, endpoint=False)
        return angles
    
    def plot(self, color = 'black'):
        """Plot the Blisk structure with blades."""
        fig, ax = plt.subplots(figsize=(6, 6))
        circle = plt.Circle((0, 0), self.disk_radius, color=color, fill=True, alpha=0.7)
        ax.add_patch(circle)
        for angle in self.blade_positions:
            x_start, y_start = self.disk_radius * np.cos(angle), self.disk_radius * np.sin(angle)
            x_end, y_end = (self.disk_radius + self.blade_length) * np.cos(angle), (self.disk_radius + self.blade_length) * np.sin(angle)
            ax.plot([x_start, x_end], [y_start, y_end], 'b', lw=2)
        ax.set_xlim(-self.disk_radius - self.blade_length, self.disk_radius + self.blade_length)
        ax.set_ylim(-self.disk_radius - self.blade_length, self.disk_radius + self.blade_length)
        ax.set_aspect('equal')
        ax.set_title("Blisk Structure")
        plt.grid(True)
        return fig

class BliskAnalysis:
    """
    Class for analyzing the flutter characteristics of a blisk.
    """
    def __init__(self, mass_ratio, frequency_ratio, reduced_velocity):
        self.mass_ratio = mass_ratio
        self.frequency_ratio = frequency_ratio
        self.reduced_velocity = reduced_velocity
        self.eigenvalues = None
        self.eigenvectors = None

    def compute_flutter_response(self):
        """Compute eigenvalues and eigenvectors for flutter analysis."""
        M = np.array([[1, 0.2], [0.2, 1]])
        K = np.array([[self.frequency_ratio**2 / self.reduced_velocity**2, 2/self.mass_ratio],
                      [0, (1 / self.reduced_velocity**2) - ((2 / self.mass_ratio) * 0.5)]])
        A = np.block([
            [np.zeros_like(M), -np.eye(2)],
            [la.inv(M) @ K, np.zeros_like(M)]
        ])
        self.eigenvalues, self.eigenvectors = la.eig(A)

    def get_results(self):
        """Return computed eigenvalues and damping ratios."""
        if self.eigenvalues is None:
            self.compute_flutter_response()
        damping_ratios = -self.eigenvalues.real / np.abs(self.eigenvalues.imag)
        return self.eigenvalues, damping_ratios