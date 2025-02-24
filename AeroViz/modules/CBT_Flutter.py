# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider, IntSlider, Dropdown, GridspecLayout, Checkbox
from IPython.display import HTML, display, clear_output

from matplotlib.animation import FuncAnimation, PillowWriter
import scipy.linalg as la
from matplotlib.patches import Polygon
import matplotlib.transforms as transforms
import matplotlib.patches as patches

# Refactoring code to use oop principles
# Define a class for the airfoil
class Airfoil:
    """
    Class to represent a NACA 4-digit airfoil with variable parameters.
    """  
    def __init__(self, max_camber, camber_position, thickness, num_points=100, length=1, centrepos=0.5):
        self.max_camber = max_camber
        self.camber_position = camber_position
        self.thickness = thickness
        self.num_points = num_points
        self.length = length
        self.centrepos = centrepos
        self.coords = None
        self.code = None
    
    def generate_naca_airfoil4(self):
        """
        Generates a 4-digit NACA airfoil based on the given parameters, with variable length and centering position.

        Using Wikipedia's parameterization for a 4-digit NACA airfoil for symmetric airfoils, and NACA's original equations for cambered airfoils.

        Parameters:
        max_camber (float): Maximum camber as a percentage of the chord (0 to 9.9).
        camber_position (float): Position of maximum camber as a fraction of the chord (0 to 0.9).
        thickness (float): Maximum thickness as a percentage of the chord (0 to 40).
        num_points (int): Number of points to generate for the airfoil (default: 100).
        length (float): Length of the airfoil (chord length). Defaults to 1.
        centrepos (float): Position along the chord at which the airfoil will be centered (0 to 1).

        Returns:
        numpy.ndarray: Array of airfoil coordinates with columns for x and y coordinates.
        """
        # Convert max_camber and thickness to decimals
        m = self.max_camber / 100
        p = self.camber_position / 10
        t = self.thickness / 100
        
        # Generate x-coordinates with cosine spacing
        beta = np.linspace(0, np.pi, self.num_points)
        x = 0.5 * (1 - np.cos(beta))
        
        # Initialize y-coordinates for camber line (yc) and thickness distribution (yt)
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        
        # Compute yc and dyc_dx based on the position of maximum camber (p)
        for i in range(self.num_points):
            if x[i] < p:
                yc[i] = (m / p**2) * (2 * p * x[i] - x[i]**2)
                dyc_dx[i] = (2 * m / p**2) * (p - x[i])
            else:
                yc[i] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[i] - x[i]**2)
                dyc_dx[i] = (2 * m / (1 - p)**2) * (p - x[i])
        
        # Calculate the angle (theta) for each point
        theta = np.arctan(dyc_dx)

        # Calculate upper and lower surface coordinates
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # Combine upper and lower surface points; start at trailing edge and move along upper surface
        # back to the leading edge, then along the lower surface to form a closed loop
        x_coords = np.concatenate([xu[::-1], xl[1:]])
        y_coords = np.concatenate([yu[::-1], yl[1:]])
                                  
        # Scale by the length parameter, then center along the chord
        x_coords *= self.length
        y_coords *= self.length
        x_coords -= self.centrepos * self.length

        # Stor to the relevant class attribute
        self.coords = np.column_stack((x_coords, y_coords))
        self.code = f"{int(self.max_camber)}{int(self.camber_position)}{int(self.thickness):02d}"

    def update(self, max_camber, camber_pos, thickness, num_points=100, chord_length=1, centrepos=0.5):
        self.max_camber = max_camber
        self.camber_position = camber_pos
        self.thickness = thickness
        self.num_points = num_points
        self.length = chord_length
        self.centrepos = centrepos
        # Regenerate airfoil coordinates with updated parameters
        self.generate_naca_airfoil4()

        # Update the airfoil code
        self.code = f"{int(max_camber * 100)}{int(camber_pos * 10)}{int(thickness * 100):02d}"
    
    def plot(self, show_chord=False):
        fig, ax = plt.subplots()
        ax.plot(self.coords[:, 0], self.coords[:, 1], 'k-', lw=2)
        if show_chord:
            ax.axhline(0, color='r', linestyle='--', lw=1)
            ax.text(0.5, -0.05, 'Chord Line', color='r', ha='center')
        ax.set_aspect('equal', 'box')
        ax.set_title(f"Airfoil {self.code} Plot")
        ax.axis('off')
        plt.show()
        return fig

# Define a class for the flutter analysis
class FlutterAnalysis:
    """
    Class to represent flutter analysis for a coupled system.
    """

    def __init__(self, mu, sigma, V, a, b, e, r, mode='Steady - State Space', w_theta=100):
        self.mu = mu
        self.sigma = sigma
        self.V = V
        self.a = a
        self.b = b
        self.e = e
        self.r = r
        self.mode = mode
        self.w_theta = w_theta
        self.vals = None
        self.vecs = None
        self.omega = None
        self.zeta = None

        self.x_theta = None
    
    def compute_response(self):
        """
        Compute the eigenvalues and eigenvectors for analysis of the flutter response of a coupled system with the given parameters.

        Returns:
        Eigen Value Problem Solution: 
        Eigenvalues and Eigenvectors to problem below:

        p**2*I - M*^-1(K* - F) = 0
        """

        # Torsional Axis Offset
        self.x_theta = (self.e - self.a)# * self.b

        # Mass and Stiffness Matrices
        M = np.array([
                    [1, self.x_theta],
                    [self.x_theta, self.r**2]
                    ])
        

        if self.mode == 'Steady - Basic':
            # **Placeholder implementation to avoid execution errors**
            self.vals = np.array([0 + 0j, 0 + 0j])  # Assign default zero eigenvalues
            self.vecs = np.array([[1, 0], [0, 1]])  # Assign default identity eigenvectors
            self.omega = np.array([0, 0])
            self.zeta = np.array([0, 0])

            print("Steady - Basic mode selected, but implementation is not included yet.")


        elif self.mode == 'Steady - State Space':
            K = np.array([
                [self.sigma**2 / self.V**2, 2/self.mu],
                [0, (self.r**2 / self.V**2) - ((2 / self.mu) * (self.a + 0.5))]
                ])

            # Create state space representation then solve the eigenvalue problem, then set the eigenvalues and eigenvectors attributes
            A = np.block([
                [np.zeros_like(M), -np.eye(2)],
                [la.inv(M) @ (K), np.zeros_like(M)]
            ])

            p, self.vecs = la.eig(A)

            # Dimensionalise eigenvalues - in essence returns lambda = p*u/b
            self.vals = p*self.V*self.w_theta
            #self.vecs[0] = self.vecs[0] * self.b # uncomment if needed to scale the eigenvectors
            
            # Split lambda into components
            self.omega = np.abs(self.vals.imag)  # Frequency component
            self.zeta = -self.vals.real / np.abs(self.vals.imag)  # Damping ratio
        
        elif self.mode == 'Quasi Steady - State Space':
            # Placeholder for Quasi-Steady implementation
            self.vals = np.array([0 + 0j, 0 + 0j])
            self.vecs = np.array([[1, 0], [0, 1]])
            self.omega = np.array([0, 0])
            self.zeta = np.array([0, 0])

            print("Quasi-Steady mode selected, but implementation is not included yet.")


        else:
            raise ValueError("Invalid mode selected; choose 'Steady - Basic' or 'Steady - State Space' or 'Quasi Steady - State Space'.")


    # Static Plots Section
    
    # def plot_freq_damp(self):
    #     """
    #     Plot frequency ratio and damping ratio vs reduced velocity.
    #     """
    #     fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    #     # Frequency Ratio Plot
    #     axes[0].plot(self.V, self.omega / self.w_theta, 'bx', label="Computed Frequency")
    #     axes[0].set_title("Modal Frequency Ratio vs. Reduced Velocity")
    #     axes[0].set_xlabel("Reduced Velocity (V)")
    #     axes[0].set_ylabel("Modal Frequency Ratio ($\\omega / \\omega_\\theta$)")
    #     axes[0].legend()
    #     axes[0].grid()

    #     # Damping Ratio Plot
    #     axes[1].plot(self.V, self.zeta, 'bx', label="Computed Damping")
    #     axes[1].set_title("Modal Damping Ratio vs. Reduced Velocity")
    #     axes[1].set_xlabel("Reduced Velocity (V)")
    #     axes[1].set_ylabel("Damping Ratio")
    #     axes[1].legend()
    #     axes[1].grid()

    #     plt.tight_layout()
    #     return fig

    # def plot_amp_phase(self):
    #     """
    #     Plot modal amplitudes and phase differences.
    #     """
    #     fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    #     # Mode Shape Amplitudes
    #     amplitudes = np.abs(self.vecs)
    #     axes[0].plot(self.V, amplitudes[:, 0], 'bo-', label="Mode 1")
    #     axes[0].plot(self.V, amplitudes[:, 1], 'ro-', label="Mode 2")
    #     axes[0].set_title("Mode Shape Amplitudes")
    #     axes[0].set_xlabel("Reduced Velocity (V)")
    #     axes[0].set_ylabel("Amplitude")
    #     axes[0].legend()
    #     axes[0].grid()

    #     # Phase Differences
    #     phases = np.angle(self.vecs[:, 1] / self.vecs[:, 0])  # Phase difference between modes
    #     axes[1].plot(self.V, phases, 'go-', label="Phase Difference")
    #     axes[1].set_title("Mode Phase Differences")
    #     axes[1].set_xlabel("Reduced Velocity (V)")
    #     axes[1].set_ylabel("Phase Difference (radians)")
    #     axes[1].legend()
    #     axes[1].grid()

    #     plt.tight_layout()
    #     return fig

    def plot_displacement(self):
        """
        Plot the displacement time history for flutter analysis.
        """
        t = np.linspace(0, 10, 500)  # Time vector

        # Extract modal frequencies and damping
        omega = np.abs(self.vals.imag)
        gamma = self.vals.real
        zeta = -gamma / omega

        # Compute natural frequency
        w = omega / np.sqrt(1 - zeta**2)

        # Compute plunge (h) and pitch (θ) displacements
        h_t = np.exp(gamma[0] * t) * np.cos(omega[0] * t)
        theta_t = np.exp(gamma[1] * t) * np.cos(omega[1] * t)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        # Plot plunge displacement
        axes[0].plot(t, h_t, 'b-', label="Plunge Displacement (h)")
        axes[0].set_title("Time History of Plunge Displacement")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Displacement (h)")
        axes[0].legend()
        axes[0].grid()

        # Plot pitch displacement
        axes[1].plot(t, theta_t, 'r-', label="Pitch Displacement (θ)")
        axes[1].set_title("Time History of Pitch Displacement")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Displacement (θ)")
        axes[1].legend()
        axes[1].grid()

        plt.tight_layout()
        return fig
    
    # Animation Section

    #def generate_torsional_spring(num_points, c, spiral_length, trim_factor=0):
    
    # def animate_airfoil(self, airfoil_coords, eigmode=1, scale=0.1, duration=10, fps=30, properties=None):
    #         """
    #         Animate the flutter response of the airfoil based on computed eigenvalues.

    #         Parameters:
    #         - airfoil_coords: (numpy.ndarray) Precomputed airfoil coordinates (x, y).
    #         - eigmode: (int) Eigenmode to animate (1 or 2).
    #         - scale: (float) Scaling factor for visualization.
    #         - duration: (float) Duration of animation in seconds.
    #         - fps: (int) Frames per second.

    #         Returns:
    #         - HTML animation output.
    #         """
    #         if properties is None:
    #             properties = {
    #                 'airfoil_color': 'lightblue',
    #                 'transparency': 50,
    #                 'angled_lines': True,
    #                 'angled_text': True,
    #                 'annotated_text_color': 'black',
    #                 'show_chord': True
    #             }

    #         # Ensure eigenvalues are computed
    #         if self.vals is None or self.vecs is None:
    #             raise ValueError("Eigenvalues and eigenvectors are not computed. Run compute_response() first.")

    #         eigmode -= 1  # Convert to zero-based index

    #         # Extract first eigenvalue and first eigenvector pair
    #         lambda_val = self.vals[eigmode]  # First eigenvalue
    #         h_tidal = self.vecs[0, eigmode]  # First plunge mode
    #         theta_tidal = self.vecs[1, eigmode]  # First pitch mode

    #         # Time discretization
    #         t = np.linspace(0, duration, duration * fps)

    #         # Compute plunge and pitch displacement over time
    #         h_t = np.real(h_tidal * np.exp(lambda_val * t)) * scale
    #         theta_t = np.real(theta_tidal * np.exp(lambda_val * t))  # θ in radians

    #         # Setup figure
    #         fig, ax = plt.subplots()
    #         airfoil_length = np.max(airfoil_coords[:, 0]) - np.min(airfoil_coords[:, 0])
    #         alpha = properties['transparency'] / 100

    #         # Plot airfoil
    #         airfoil_patch = Polygon(airfoil_coords, closed=True, edgecolor='k',
    #                                 facecolor=properties['airfoil_color'], alpha=alpha)
    #         ax.add_patch(airfoil_patch)

    #         # Axis settings
    #         ax.set_xlim(-2 * airfoil_length, 2 * airfoil_length)
    #         ax.set_ylim(-2 * airfoil_length, 2 * airfoil_length)
    #         ax.set_aspect('equal')
    #         ax.set_title(f"Coupled Flutter Response ({self.mode})")
    #         ax.set_xlabel("Angular Displacement")
    #         ax.set_ylabel("Vertical Displacement")

    #         # Reference chord line
    #         if properties['show_chord']:
    #             chord_line, = ax.plot([-airfoil_length, airfoil_length], [0, 0], 'k--', alpha=0.5)

    #         # Annotation Text
    #         angle_text = ax.text(0.98, 0.95, '', transform=ax.transAxes, fontsize=10,
    #                             color=properties['annotated_text_color'], ha='right')
    #         vertical_text = ax.text(0.98, 0.90, '', transform=ax.transAxes, fontsize=10,
    #                                 color=properties['annotated_text_color'], ha='right')

    #         # **Update function for animation**
    #         def update(frame):
    #             # Convert θ to degrees
    #             theta_deg = np.degrees(theta_t[frame])

    #             # Apply transformation: **rotation & translation**
    #             trans = transforms.Affine2D().rotate_deg_around(self.a, 0, theta_deg).translate(0, h_t[frame]) + ax.transData
    #             airfoil_patch.set_transform(trans)

    #             # Rotate chord line
    #             if properties['show_chord']:
    #                 chord_x = np.array([-airfoil_length, airfoil_length])
    #                 chord_y = np.array([0, 0])
    #                 chord_x_rot = chord_x * np.cos(theta_t[frame]) - chord_y * np.sin(theta_t[frame])
    #                 chord_y_rot = chord_x * np.sin(theta_t[frame]) + chord_y * np.cos(theta_t[frame])
    #                 chord_line.set_xdata(chord_x_rot)
    #                 chord_line.set_ydata(chord_y_rot)

    #             # Update text annotations
    #             angle_text.set_text(f'Pitch Angle: {theta_deg:.2f}°')
    #             vertical_text.set_text(f'Plunge Displacement: {h_t[frame]:.2f}')

    #             return airfoil_patch, chord_line, angle_text, vertical_text

    #         # Create animation
    #         ani = FuncAnimation(fig, update, frames=len(t), blit=True, interval=1000 / fps)

    #         plt.show()
    #         return HTML(ani.to_jshtml())

    def animate_flutter(self, airfoil_coords, duration=10, fps=30, properties={'airfoil_color': 'lightblue', 
                    'transparency': 50, 'angled_lines': True, 'angled_text': True, 
                    'annotated_text_color': 'black', 'show_chord': True}):
        """
        Animate the flutter response of a coupled system with the given parameters.
        """

        # Compute eigenvalues and eigenvectors
        # Compute eigenvalues and eigenvectors
        self.compute_response()

        # Extract the 4 eigenvalue pairs (lambda values)
        lambda_vals = self.vals[:4]  # Select first 4 eigenvalues
        real_parts = np.real(lambda_vals)  # Gamma (damping)
        imag_parts = np.imag(lambda_vals)  # Omega (frequency)

        # Extract the 4 eigenvector pairs (q_tidal)
        h_tidals = np.real(self.vecs[0, :4]) * self.b  # Extract real plunge displacements
        theta_tidals = np.real(self.vecs[1, :4])  # Extract real torsional displacements

        # Compute phase differences between plunge and pitch modes
        phase_diffs = np.angle(self.vecs[1, :4] / self.vecs[0, :4])

        # Define time range for animation
        t = np.linspace(0, duration, duration * fps)

        # Set up figure with 2x2 grid for 4 modes
        fig, axes = plt.subplots(4, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [1.5, 1], 'height_ratios': [1, 1, 1, 1]})
        fig.suptitle("Coupled Flutter Modes - Time Response", fontsize=14)

        # Precompute displacement histories for 4 eigenvector pairs
        h_t = np.array([
            h_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t) for i in range(4)
        ])
        theta_t = np.array([
            theta_tidals[i] * np.exp(real_parts[i] * t) * np.cos(imag_parts[i] * t + phase_diffs[i]) for i in range(4)
        ])

        # Create 4 airfoil patches for animation (1 per row)
        airfoil_patches = []
        for i in range(4):
            airfoil_patch = Polygon(airfoil_coords, closed=True, edgecolor='k', facecolor=properties['airfoil_color'], alpha=0.5)
            axes[i, 0].add_patch(airfoil_patch)
            axes[i, 0].set_xlim(-self.b, self.b)
            axes[i, 0].set_ylim(-2 * self.b, 2 * self.b)
            axes[i, 0].set_aspect('equal')
            airfoil_patches.append(airfoil_patch)

        # Titles for each mode (left column)
        mode_titles = [f"Mode {i+1} Animation" for i in range(4)]
        for ax, title in zip(axes[:, 0], mode_titles):
            ax.set_title(title)

        # Right column (Vibration amplitude & phase plots)
        for i in range(4):
            axes[i, 1].plot(t, h_t[i], 'b-', label=f"Mode {i+1} Plunge")
            axes[i, 1].plot(t, theta_t[i], 'r--', label=f"Mode {i+1} Twist")
            axes[i, 1].set_xlabel("Vibration Period")
            axes[i, 1].set_ylabel("Displacement")
            axes[i, 1].legend()
            axes[i, 1].grid()
            axes[i, 1].set_title(f"Mode {i+1} Amplitude & Phase Difference")

        # Function to update animation frames
        def update(frame):
            for i in range(4):
                trans = transforms.Affine2D().rotate_deg_around(self.a, 0, np.degrees(theta_t[i, frame])).translate(0, h_t[i, frame]) + axes[i, 0].transData
                airfoil_patches[i].set_transform(trans)
            return airfoil_patches

        # Run the animation
        ani = FuncAnimation(fig, update, frames=len(t), blit=True, interval=1000 / fps)

        plt.show()
        return HTML(ani.to_jshtml())




    
        


## Refactored code example using the classes defined above
# Instantiate the Airfoil object
airfoil = Airfoil(2, 5, 12)
airfoil.generate_naca_airfoil4()
airfoil.plot()
airfoil.update(0,0,12)
airfoil.plot()

# # Instantiate the FlutterAnalysis object
# flutter = FlutterAnalysis(20, 0.4, 2, -0.2, 0.5, -0.1, np.sqrt(6/25), mode='Steady - State Space', w_theta=100)
# flutter.compute_response()
# flutter.animate_airfoil(airfoil.coords, scale=0.1, duration=10, fps=30)
# # flutter.plot_freq_damp()
# # flutter.plot_amp_phase()
# # flutter.plot_displacement()

# print("Eigenvalues:", flutter.vals)
# print("Eigenvectors:", flutter.vecs)
# print("Omega:", flutter.omega)
# print("Zeta:", flutter.zeta)









##
# Function to generate airfoil coordinates
def naca_airfoil4(max_camber, camber_position, thickness, num_points=100, length=1, centrepos=0.5):
    """
    Generates a 4-digit NACA airfoil based on the given parameters, with variable length and centering position.

    Using Wikipedia's parameterization for a 4-digit NACA airfoil for symmetric airfoils, and NACA's original equations for cambered airfoils.

    Parameters:
    max_camber (float): Maximum camber as a percentage of the chord (0 to 9.9).
    camber_position (float): Position of maximum camber as a fraction of the chord (0 to 0.9).
    thickness (float): Maximum thickness as a percentage of the chord (0 to 40).
    num_points (int): Number of points to generate for the airfoil (default: 100).
    length (float): Length of the airfoil (chord length). Defaults to 1.
    centrepos (float): Position along the chord at which the airfoil will be centered (0 to 1).

    Returns:
    numpy.ndarray: Array of airfoil coordinates with columns for x and y coordinates.
    """
    # Convert max_camber and thickness to decimals
    m = max_camber / 100
    p = camber_position / 10
    t = thickness / 100
    
    # Generate x-coordinates with cosine spacing
    beta = np.linspace(0, np.pi, num_points)
    x = 0.5 * (1 - np.cos(beta))
    
    # Initialize y-coordinates for camber line (yc) and thickness distribution (yt)
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    # Compute yc and dyc_dx based on the position of maximum camber (p)
    for i in range(num_points):
        if x[i] < p:
            yc[i] = (m / p**2) * (2 * p * x[i] - x[i]**2)
            dyc_dx[i] = (2 * m / p**2) * (p - x[i])
        else:
            yc[i] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[i] - x[i]**2)
            dyc_dx[i] = (2 * m / (1 - p)**2) * (p - x[i])
    
    # Calculate the angle (theta) for each point
    theta = np.arctan(dyc_dx)
    
    # Calculate upper and lower surface coordinates
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Combine upper and lower surface points; start at trailing edge and move along upper surface
    # back to the leading edge, then along the lower surface to form a closed loop
    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])

    # Scale by the length parameter, then center along the chord
    x_coords *= length
    y_coords *= length
    x_coords -= centrepos * length
    
    # Return as a two-column array
    airfoil_coords = np.column_stack((x_coords, y_coords))
    
    return airfoil_coords


##
# Function to plot the airfoil
def plot_airfoil(airfoil_coords, show_chord=False):
    """
    Plot the airfoil shape based on the given coordinates.

    Parameters:
    airfoil_coords (numpy.ndarray): Array of airfoil coordinates with columns for x and y coordinates.
    show_chord (bool): Whether to show the chord length on the plot (default: False).
    """
    # Create the figure and axis
    fig, ax = plt.subplots()
    
    ax.plot(airfoil_coords[:, 0], airfoil_coords[:, 1], 'k-', lw=2)
    
    # Add chord line if requested
    if show_chord:
        ax.axhline(0, color='r', linestyle='--', lw=1)
        ax.text(0.5, -0.05, 'Chord Line', color='r', ha='center')
    
    # Set aspect ratio to be equal
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    
    plt.show()
    return fig


##
def update_airfoil(max_camber, camber_pos, chord_length, thickness, num_points=100, centrepos=0.5):
    """
    Update the global variable `general_airfoil_coordinates` with airfoil data 
    and save the updated coordinates to a .dat file. No plotting or return values.

    Parameters:
        max_camber (float): Maximum camber as a fraction of chord length.
        camber_pos (float): Position of maximum camber as a fraction of chord length.
        chord_length (float): Length of the chord (scaling factor).
        thickness (float): Maximum thickness as a fraction of chord length.
        num_points (int): Number of points used to generate the airfoil.
        centrepos (float): Center position for the airfoil along the x-axis.
    """
    global general_airfoil_coordinates

    # Generate the airfoil coordinates
    general_airfoil_coordinates = naca_airfoil4(
        max_camber=max_camber,
        camber_position=camber_pos,
        thickness=thickness,
        num_points=num_points,
        length=chord_length,
        centrepos=centrepos
    )

    # Construct the NACA code and log the update
    airfoil_code = f"{int(max_camber * 100)}{int(camber_pos * 10)}{int(thickness * 100):02d}"

    # Save to a .dat file
    np.savetxt(f"Airfoil Objects/NACA_{airfoil_code}_airfoil.dat", general_airfoil_coordinates, fmt='%.6f', delimiter=' ')

    # Optional: Provide feedback for debugging, if needed
    print(f"Coordinates saved to NACA_{airfoil_code}_airfoil.dat.")



##
# Tentative function for steady aerodynamic influence matrix only
# This functino accounts for unsteady effects and damping in the aerodynamic influence matrix
# Even though the steady case 
def coupled_flutter_response(mu,sigma,V, a,b,e,r, mode='Steady - Basic', w_theta = 100):
    """
    Compute the eignvalues and eigenvectors for anlaysis of the
    flutter response of a coupled system with the given parameters.

    Parameters:
    mu (float): Mass Ratio.
    sigma (float): Frequency Ratio.
    V (float): Reduced Velocity.
    a (float): Torsional Axis Location.
    b (float): Semi-Chord Length.
    e (float): Eccentricity - Center of Mass Location from torsional axis.
    r (float): Radius of Gyration.
    mode (string): Approximation Chosen for Aerodynamic Influence Matrix; 'Steady', 'Unsteady', 'Quasi-Steady' or 'kussner'. - kussner stc
    w_theta (float): Frequency of Torsional Vibrations, Hz.

    Returns:
    Eigen Value Problem Solution: 
    Eigenvalues and Eigenvectors to problem below:

    p**2*I - M*^-1(K* - F) = 0
    """

    # Torsional Axis Offset
    x_theta = (e - a)# * b

    # Mass and Stiffness Matrices
    M = np.array([[1, x_theta],
                  [x_theta, r**2]])
    
    K = np.array([
        [sigma**2 / V**2, 2/mu],
        [0, (r**2 / V**2) - ((2 / mu) * (a + 0.5))]])
    

    
    if mode == 'Steady - Basic':

        # Solve the eigenvalue problem
        M_inv = np.linalg.inv(M)
        A = np.matmul(M_inv, K)
        eigvals, eigvecs = np.linalg.eig(-A)

        return eigvals, eigvecs
    
    if mode == 'Steady - State Space':
        #w_theta = 100 # Frequency of Torsional Vibrations, Hz
        # Create state space representation then solve the eigenvalue problem

        
        A = np.block([
            [np.zeros_like(M), -np.eye(2)],
            [la.inv(M) @ (K), np.zeros_like(M)]
        ])

        p, eigvecs = la.eig(A)

        # Dimensionalise eigenvalues - in essence return lambda = p*u/b
        eigvals = p*V*w_theta

        # # Remove complex conjugates, retain positive frequencies i.e., p > 0
        # eigvals = eigvals[np.where(np.real(eigvals) > 0)]
        # eigvecs = eigvecs[:, np.where(np.real(eigvals) > 0)]
        
        return eigvals, eigvecs

    else:
        raise ValueError("Invalid mode selected; choose 'Steady - Basic' or 'Steady - State Space'.")

# Function to find closest points
def find_closest_points(reference, target, max_points):
    reference = np.array(reference)
    target = np.array(target)
    differences = np.abs(reference[:, None] - target[None, :])
    closest_indices = np.argmin(differences, axis=0)
    closest_points = np.unique(reference[closest_indices])
    if len(closest_points) > max_points:
        distances = np.abs(closest_points[:, None] - target).min(axis=1)
        sorted_indices = np.argsort(distances)
        closest_points = closest_points[sorted_indices][:max_points]
    return closest_points, closest_indices





# parametric form after small iterations sqrt(x^2 + y^2) = v/w * atan(y/x) + c
def archimedes_spiral(num_points,c,spiral_length):
    """
    Function to creatte a torsional spring element for the vizualizattions of flutter analysis.
    The function uses the archimedean spiral to create the spring element of variable dimensions.
    Archimedian spiral is a spiral with polar equation r = a + b*theta; from wikipedia. where r is the radius, a is the initial offset, b is the base spacing between coils and theta is the angle in radians.

    Parameters:
    num_points (int): Number of points to generate for the spring element.
    c (float): constant for the spiral equation.
    spiral_length (float): Length of the spiral element.


    Returns:
    x_coords: x-coordinates of the spring element in cartesian coordinates.
    y_coords: y-coordinates of the spring element in cartesian coordinates.
    """
    theta = np.linspace(0, spiral_length * np.pi, num_points)
    r = c * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# Linear spring element using sine wave
def linear_spring(num_points, length):
    """
    Function to create a linear spring element for the visualization of flutter analysis.
    The function uses a sine wave to create the spring element of variable dimensions.

    Parameters:
    num_points (int): Number of points to generate for the spring element.
    length (float): Length of the spring element.

    Returns:
    x_coords: x-coordinates of the spring element in cartesian coordinates.
    y_coords: y-coordinates of the spring element in cartesian coordinates.
    """
    x = np.linspace(0, length, num_points)
    y = np.sin(5 * x)
    return x, y

def linear_spring2(num_points, length):
    """
    Function to create a linear spring element using a triangular wave for visualization of flutter analysis.

    Parameters:
    num_points (int): Number of points to generate for the spring element.
    length (float): Length of the spring element.

    Returns:
    x_coords: x-coordinates of the spring element in cartesian coordinates.
    y_coords: y-coordinates of the spring element in cartesian coordinates.
    """
    # Generate x-coordinates
    x = np.linspace(0, length, num_points)

    # Define the period of the triangular wave as a fraction of the total length
    period = length / 5  # Adjust the denominator to control the number of cycles
    amplitude = 0.1 * length  # Set amplitude relative to the spring length

    # Calculate the fractional position within each period
    fractional_position = (x % period) / period

    # Construct the triangular wave
    y = 2 * amplitude * np.abs(fractional_position - 0.5)

    return x, y

# Define scaling matrix for compression/stretching
def scale_spring(x, y, scale_factor):
    scaling_matrix = np.array([[scale_factor, 0],
                               [0, 1]])
    points = np.vstack((x, y))  # Combine x and y into a single matrix
    transformed_points = scaling_matrix @ points  # Apply scaling
    return transformed_points[0], transformed_points[1]


## Animation Functions, animating airfoil, archimedes spiral for torsional spring, and XX for linear spring

def animate_flutter(airfoil_coords, mu, sigma, V, a, b, e, r, mode='State Space', scale=0.1, duration=10, fps=30, properties={'airfoil_color': 'lightblue', 'transparency': 50, 'angled_lines': True, 'angled_text': True, 'annotated_text_color': 'black', 'show_chord': True}):
    """
    Animate the flutter response of a coupled system with the given parameters.

    Additional Parameters:
    airfoil_coords (numpy.ndarray): Array of airfoil coordinates with columns for x and y coordinates.
    scale (float): Scaling factor for the animation (default: 0.1).
    duration (float): Duration of the animation in seconds (default: 10).
    fps (int): Frames per second for the animation (default: 30).
    properties (dict): Dictionary of properties for the animation.
        properties['airfoil_color'] (str): Color for the airfoil shape (default: 'lightblue').
        properties['angled_lines'] (bool): Whether to include angled lines for reference (default: True).
        properties['angled_text'] (bool): Whether to include annotated text for the displacement angle (default: True).
        properties['annotated_text_color'] (str): Color for the annotated text (default: 'black').
        properties['show_chord'] (bool): Whether to show the chord length on the plot (default: False).

    Returns:
    HTML: Animation of the flutter response.
    """
    vals, vecs = coupled_flutter_response(mu, sigma, V, a, b, e, r, mode)
    
    # Extract first eigenvalue and eigenvector pair

    val = vals[0]
    vec = vecs[:, 0]
    


    real_parts = val.real # Gamma component
    imag_parts = val.imag # Omega component

    # Calculate phase differences from eigenvectors
    phase_diffs = np.angle(vecs[1] / vecs[0])

    fig, axs = plt.subplots(1, 2)
    plt.figsize = (20, 20)


    # Add space between subplots
    plt.subplots_adjust(wspace=0.75)

    t = np.linspace(0, duration, duration * fps)

    # Calculate values of interest
    airfoil_length = np.max(airfoil_coords[:, 0]) - np.min(airfoil_coords[:, 0])
    alpha = properties['transparency'] / 100

    # Plot setup
    airfoil_patch = Polygon(airfoil_coords, closed=True, edgecolor='k', facecolor=properties['airfoil_color'], alpha=alpha)

    axs[0].add_patch(airfoil_patch)
    axs[0].set_xlim(-2 * airfoil_length, 2 * airfoil_length)
    axs[0].set_ylim(-2 * airfoil_length, 2 * airfoil_length)
    axs[0].set_aspect('equal')
    
    axs[0].set_title(f"Coupled Flutter Animation - ({mode})")
    axs[0].set_xlabel("Angular Displacement")
    axs[0].set_ylabel("Vertical Displacement")

    axs[1].set_title(f"Phase Difference Between Modes - ({mode})")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Phase Difference (radians)")
    # Ensure t and phase_diffs have the same length
    min_length = min(len(t), len(phase_diffs))
    axs[1].plot(t[:min_length], phase_diffs[:min_length])


    if properties['angled_text']:
        # Adding an annotation for displacement angle
        angle_text = axs[0].text(0.98, 0.95, '', transform=axs[0].transAxes, fontsize=10, color=properties['annotated_text_color'], ha='right')
        vertical_text = axs[0].text(0.98, 0.90, '', transform=axs[0].transAxes, fontsize=10, color=properties['annotated_text_color'], ha='right')
        ax2 = axs[0].twinx()
        ax2.set_ylabel('Displacement Angle (degrees)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    # Function to update each frame
    def update(frame):
        h_t = np.exp(real_parts[0] * frame) * np.cos(imag_parts[0] * frame)  # Scale for visibility
        theta_t = np.exp(real_parts[1] * frame) * np.cos(imag_parts[1] * frame)  # Scale for visibility
        
        # Define transformation: translation + rotation, for the airfoil
        trans = transforms.Affine2D().rotate_deg_around(a, 0, np.degrees(theta_t)).translate(0, h_t) + axs[0].transData
        airfoil_patch.set_transform(trans) 
        
        # Update displacement angle text with +XX for positive angles and -XX for negatives
        angle_text.set_text(f'Displacement Angle: {"+" if theta_t >= 0 else "-"}{abs(np.degrees(theta_t)):.2f}°')

        # Update vertical displacement text with +XX for positive displacements and -XX for negatives
        vertical_text.set_text(f'Vertical Displacement: {"+" if h_t >= 0 else "-"}{abs(h_t):.2f}')

        # # Place a dot on the second plot to show the current phase difference, and update the plot
        # axs[1].lines[0].set_data(t[:frame], phase_diffs[:frame])
        # axs[1].lines[0].set_marker('o')
        # axs[1].lines[0].set_markerfacecolor('r')
        # axs[1].lines[0].set_markeredgecolor('r')
        # axs[1].lines[0].set_markersize(5)

        
        return airfoil_patch, angle_text, vertical_text, axs[1].lines[0]
    
    ani = FuncAnimation(fig, update, frames=t, blit=True, interval=50)
    plt.show()
    return HTML(ani.to_jshtml())


# quick animation to show how the spring element's length can be varied
def animate_torsional_spring(num_points,c,spiral_length, duration=10, fps=30):
	"""
	Function to animate the torsional spring element for the vizualizattions of flutter analysis.
	The function uses the archimedean spiral to create the spring element of variable dimensions.
	Archimedian spiral is a spiral with polar equation r = a + b*theta; from wikipedia. where r is the radius, a is the initial offset, b is the base spacing between coils and theta is the angle in radians.

	Parameters:
	num_points (int): Number of points to generate for the spring element.
	c (float): constant for the spiral equation.
	spiral_length (float): Length of the spiral element.
	duration (float): Duration of the animation in seconds (default: 10).
	fps (int): Frames per second for the animation (default: 30).

	Returns:
	HTML: Animation of the torsional spring element.
	"""
	fig, ax = plt.subplots()
	t = np.linspace(0, duration, duration * fps)
	x, y = archimedes_spiral(num_points,c,spiral_length)
	spring, = ax.plot(x, y)
	ax.set_aspect('equal')
	ax.set_title("Torsional Spring Element")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	
	# Function to update spring position for each frame
	def update(frame):
		x, y = archimedes_spiral(num_points,c,spiral_length * frame / duration)
		spring.set_xdata(x)
		spring.set_ydata(y)
		return spring,
	
	ani = FuncAnimation(fig, update, frames=t, blit=True, interval=50)
	#plt.show()
	return HTML(ani.to_jshtml())