import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
import base64
from io import BytesIO
import tempfile
import os
import sympy as sp
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Bode & Nyquist Visualizer",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None
if 'num_coeffs' not in st.session_state:
    st.session_state.num_coeffs = None
if 'den_coeffs' not in st.session_state:
    st.session_state.den_coeffs = None
if 'show_animation' not in st.session_state:
    st.session_state.show_animation = False

class NyquistVisualizer:
    def __init__(self, num=None, den=None, min_freq=-2, max_freq=2):
        if num is not None and den is not None:
            self.sys = signal.TransferFunction(num, den)
        else:
            raise ValueError("Must provide num and den coefficients")
        
        # Store frequency range
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        # Store coefficients for pole/zero analysis
        self.num_coeffs = num
        self.den_coeffs = den
        
        # Frequency range for plots
        self.w = np.logspace(min_freq, max_freq, 1000)
        
        # Calculate frequency response
        self.w, self.mag, self.phase = signal.bode(self.sys, self.w)
        self.mag_linear = 10**(self.mag / 20)
        
        # Calculate Nyquist points
        phase_rad = np.radians(self.phase)
        self.nyquist_real = self.mag_linear * np.cos(phase_rad)
        self.nyquist_imag = self.mag_linear * np.sin(phase_rad)
        
        # Store for animation (fewer points for speed)
        self.w_anim = np.logspace(min_freq, max_freq, 200)
        self.w_anim, self.mag_anim, self.phase_anim = signal.bode(self.sys, self.w_anim)
        self.mag_linear_anim = 10**(self.mag_anim / 20)
        phase_rad_anim = np.radians(self.phase_anim)
        self.nyquist_real_anim = self.mag_linear_anim * np.cos(phase_rad_anim)
        self.nyquist_imag_anim = self.mag_linear_anim * np.sin(phase_rad_anim)
    
    def get_symbolic_expressions(self, num_coeffs, den_coeffs):
        """Compute symbolic real and imaginary parts of G(jω)"""
        try:
            # Define symbols
            s, omega = sp.symbols('s omega', real=True)
            
            # Create numerator and denominator polynomials
            num_poly = sum(c * s**i for i, c in enumerate(reversed(num_coeffs)))
            den_poly = sum(c * s**i for i, c in enumerate(reversed(den_coeffs)))
            
            # Create transfer function
            G = num_poly / den_poly
            
            # Substitute s = jω
            G_jw = G.subs(s, sp.I * omega)
            
            # Separate real and imaginary parts
            real_part = sp.simplify(sp.re(G_jw))
            imag_part = sp.simplify(sp.im(G_jw))
            
            # Convert to LaTeX
            G_latex = sp.latex(G)
            real_latex = sp.latex(real_part)
            imag_latex = sp.latex(imag_part)
            
            return G_latex, real_latex, imag_latex
            
        except Exception as e:
            st.error(f"Error computing symbolic expressions: {str(e)}")
            return None, None, None
    
    def plot_bode(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.semilogx(self.w, self.mag, 'b', linewidth=2)
        ax1.set_ylabel('Magnitude [dB]', fontsize=12)
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.set_title(f'Bode Diagram (ω: 10^{{{self.min_freq}}} to 10^{{{self.max_freq}}} rad/s)', fontsize=14)
        ax1.set_xlim([10**self.min_freq, 10**self.max_freq])
        
        ax2.semilogx(self.w, self.phase, 'r', linewidth=2)
        ax2.set_ylabel('Phase [deg]', fontsize=12)
        ax2.set_xlabel('Frequency [rad/s]', fontsize=12)
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.set_title(f'Phase Response', fontsize=14)
        ax2.set_xlim([10**self.min_freq, 10**self.max_freq])
        
        plt.tight_layout()
        return fig
    
    def get_poles_on_imaginary_axis(self):
        """Find poles on the imaginary axis (including origin)"""
        # Get poles of the system
        poles = np.roots(self.den_coeffs)
        
        # Check for poles on imaginary axis (real part = 0)
        imag_poles = []
        for pole in poles:
            if abs(pole.real) < 1e-10 and abs(pole.imag) > 1e-10:
                imag_poles.append(abs(pole.imag))
            elif abs(pole) < 1e-10:  # Pole at origin
                imag_poles.append(0)
        
        return sorted(imag_poles)
    
    def plot_complete_nyquist(self):
        # Check for poles on imaginary axis
        imag_poles = self.get_poles_on_imaginary_axis()
        
        if not imag_poles:
            # No poles on imaginary axis - use standard Nyquist plot
            return self._plot_standard_nyquist()
        else:
            # Poles on imaginary axis - handle with detour
            return self._plot_nyquist_with_detour(imag_poles)
    
    def _plot_standard_nyquist(self):
        """Plot Nyquist for systems without poles on imaginary axis"""
        # Create dense frequency array
        w_dense = np.logspace(self.min_freq, self.max_freq, 5000)
        w_dense, mag_dense, phase_dense = signal.bode(self.sys, w_dense)
        mag_linear_dense = 10**(mag_dense / 20)
        phase_rad_dense = np.unwrap(np.radians(phase_dense))
        
        nyquist_real_dense = mag_linear_dense * np.cos(phase_rad_dense)
        nyquist_imag_dense = mag_linear_dense * np.sin(phase_rad_dense)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot Nyquist with dense points
        ax.plot(nyquist_real_dense, nyquist_imag_dense, 'b-', linewidth=2, label='ω: 0 → ∞')
        ax.plot(nyquist_real_dense, -nyquist_imag_dense, 'b--', linewidth=1, alpha=0.5, label='ω: -∞ → 0')
        
        # Mark (-1, 0)
        ax.plot(-1, 0, 'rx', markersize=12, markeredgewidth=2, label='(-1, 0)')
        ax.text(-1.1, 0.1, '(-1, 0)', fontsize=12, color='red')
        
        self._set_nyquist_axis_limits(ax, nyquist_real_dense, nyquist_imag_dense)
        
        ax.set_xlabel('Real', fontsize=12)
        ax.set_ylabel('Imaginary', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title('Nyquist Diagram (No poles on imaginary axis)', fontsize=14)
        ax.legend(loc='best')
        
        plt.tight_layout()
        return fig
    
    def _plot_nyquist_with_detour(self, imag_poles):
        """Plot Nyquist for systems with poles on imaginary axis"""
        # Create multiple frequency segments to avoid poles
        all_real = []
        all_imag = []
        all_labels = []
        
        # Define detour radius (small epsilon)
        epsilon = 1e-2
        
        # Sort poles including 0 if present
        poles_to_avoid = sorted(imag_poles)
        
        # Start from min_freq
        current_freq = 10**self.min_freq
        
        for i, pole_freq in enumerate(poles_to_avoid):
            if pole_freq == 0:
                # Handle pole at origin with detour
                # Left of pole (negative epsilon to min_freq)
                if current_freq < epsilon:
                    w_left = np.logspace(np.log10(epsilon), self.min_freq, 1000)[::-1]
                    w_left, mag_left, phase_left = signal.bode(self.sys, w_left)
                    mag_linear_left = 10**(mag_left / 20)
                    phase_rad_left = np.unwrap(np.radians(phase_left))
                    
                    all_real.append(mag_linear_left * np.cos(phase_rad_left))
                    all_imag.append(mag_linear_left * np.sin(phase_rad_left))
                    all_labels.append('ω: ε⁻ → 0⁻' if i == 0 else '')
                
                # Right of pole (epsilon to next pole or max_freq)
                next_pole = poles_to_avoid[i+1] if i+1 < len(poles_to_avoid) else 10**self.max_freq
                w_right = np.logspace(np.log10(epsilon), np.log10(min(next_pole - epsilon, 10**self.max_freq)), 1000)
                w_right, mag_right, phase_right = signal.bode(self.sys, w_right)
                mag_linear_right = 10**(mag_right / 20)
                phase_rad_right = np.unwrap(np.radians(phase_right))
                
                all_real.append(mag_linear_right * np.cos(phase_rad_right))
                all_imag.append(mag_linear_right * np.sin(phase_rad_right))
                all_labels.append('ω: 0⁺ → ...' if i == 0 else '')
                
                current_freq = next_pole + epsilon
            else:
                # Handle pole at ω = pole_freq
                # Below pole
                w_below = np.logspace(np.log10(current_freq), np.log10(pole_freq - epsilon), 1000)
                w_below, mag_below, phase_below = signal.bode(self.sys, w_below)
                mag_linear_below = 10**(mag_below / 20)
                phase_rad_below = np.unwrap(np.radians(phase_below))
                
                all_real.append(mag_linear_below * np.cos(phase_rad_below))
                all_imag.append(mag_linear_below * np.sin(phase_rad_below))
                all_labels.append('')
                
                # Above pole
                next_pole = poles_to_avoid[i+1] if i+1 < len(poles_to_avoid) else 10**self.max_freq
                w_above = np.logspace(np.log10(pole_freq + epsilon), np.log10(min(next_pole, 10**self.max_freq)), 1000)
                w_above, mag_above, phase_above = signal.bode(self.sys, w_above)
                mag_linear_above = 10**(mag_above / 20)
                phase_rad_above = np.unwrap(np.radians(phase_above))
                
                all_real.append(mag_linear_above * np.cos(phase_rad_above))
                all_imag.append(mag_linear_above * np.sin(phase_rad_above))
                all_labels.append('')
                
                current_freq = next_pole
        
        # Plot last segment if any
        if current_freq < 10**self.max_freq:
            w_last = np.logspace(np.log10(current_freq), self.max_freq, 1000)
            w_last, mag_last, phase_last = signal.bode(self.sys, w_last)
            mag_linear_last = 10**(mag_last / 20)
            phase_rad_last = np.unwrap(np.radians(phase_last))
            
            all_real.append(mag_linear_last * np.cos(phase_rad_last))
            all_imag.append(mag_linear_last * np.sin(phase_rad_last))
            all_labels.append('')
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot all segments
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_real)))
        for i, (real_seg, imag_seg, label) in enumerate(zip(all_real, all_imag, all_labels)):
            if len(real_seg) > 0:
                ax.plot(real_seg, imag_seg, '-', color=colors[i % len(colors)], 
                       linewidth=2, alpha=0.8, label=label if label else f'Segment {i+1}')
                # Plot mirror for negative frequencies
                ax.plot(real_seg, -imag_seg, '--', color=colors[i % len(colors)], 
                       linewidth=1, alpha=0.4)
        
        # Mark (-1, 0)
        ax.plot(-1, 0, 'rx', markersize=12, markeredgewidth=2, label='(-1, 0)')
        ax.text(-1.1, 0.1, '(-1, 0)', fontsize=12, color='red')
        
        # Combine all data for axis limits
        all_real_combined = np.concatenate([segment for segment in all_real if len(segment) > 0])
        all_imag_combined = np.concatenate([segment for segment in all_imag if len(segment) > 0])
        
        self._set_nyquist_axis_limits(ax, all_real_combined, all_imag_combined)
        
        ax.set_xlabel('Real', fontsize=12)
        ax.set_ylabel('Imaginary', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_aspect('equal')
        
        if imag_poles:
            pole_text = ', '.join([f'ω={p:.2f}' if p > 0 else 'ω=0' for p in imag_poles])
            ax.set_title(f'Nyquist Diagram (Poles at: {pole_text})', fontsize=14)
        else:
            ax.set_title('Nyquist Diagram', fontsize=14)
        
        # Simplified legend
        ax.legend(['Positive ω', '(-1,0) point'], loc='best')
        
        plt.tight_layout()
        return fig
    
    def _set_nyquist_axis_limits(self, ax, real_data, imag_data):
        """Set intelligent axis limits for Nyquist plot"""
        # Include critical point and all data
        x_data = np.concatenate([real_data, [-1, 1, 0]])
        y_data = np.concatenate([imag_data, [-1, 1, 0]])
        
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        
        # Add padding
        padding = 0.2
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Ensure minimum range
        if x_range < 0.1:
            x_range = 0.1
            x_min = -0.05
            x_max = 0.05
        if y_range < 0.1:
            y_range = 0.1
            y_min = -0.05
            y_max = 0.05
        
        # Apply padding
        x_min -= x_range * padding
        x_max += x_range * padding
        y_min -= y_range * padding
        y_max += y_range * padding
        
        # Ensure we include (-1,0) and origin
        x_min = min(x_min, -1.5)
        x_max = max(x_max, 0.5)
        y_min = min(y_min, -1.5)
        y_max = max(y_max, 1.5)
        
        # Make square aspect ratio
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        max_range = max(x_max - x_min, y_max - y_min)
        
        ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
        ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    
    def create_fast_animation(self, num_frames=40):
        """
        Create animation and save as GIF using temp file
        """
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            st.error("Please install matplotlib and pillow")
            return None
        
        # Check for poles on imaginary axis - use simplified animation if present
        imag_poles = self.get_poles_on_imaginary_axis()
        if imag_poles:
            return self._create_simplified_animation(num_frames)
        
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        
        # Bode plots on left
        ax1 = plt.subplot2grid((2, 2), (0, 0))  # Magnitude
        ax2 = plt.subplot2grid((2, 2), (1, 0))  # Phase
        ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)  # Nyquist
        
        # Setup Bode plots
        ax1.semilogx(self.w_anim, self.mag_anim, 'b', alpha=0.3, linewidth=1)
        ax1.set_title('Bode - Magnitude', fontsize=12)
        ax1.set_ylabel('Magnitude [dB]')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([10**self.min_freq, 10**self.max_freq])
        
        ax2.semilogx(self.w_anim, self.phase_anim, 'r', alpha=0.3, linewidth=1)
        ax2.set_title('Bode - Phase', fontsize=12)
        ax2.set_ylabel('Phase [deg]')
        ax2.set_xlabel('Frequency [rad/s]')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([10**self.min_freq, 10**self.max_freq])
        
        # Setup Nyquist plot with proper scaling
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', alpha=0.3)
        ax3.axvline(x=0, color='k', alpha=0.3)
        ax3.set_xlabel('Real')
        ax3.set_ylabel('Imaginary')
        ax3.set_title('Nyquist Construction', fontsize=12)
        
        # Calculate nice limits
        all_real = list(self.nyquist_real_anim) + [-1, 1, 0]
        all_imag = list(self.nyquist_imag_anim) + [-1, 1, 0]
        
        x_min, x_max = min(all_real), max(all_real)
        y_min, y_max = min(all_imag), max(all_imag)
        
        margin = 0.2
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Ensure square plot
        plot_range = max(x_range, y_range, 0.1) * (1 + margin)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Adjust center if near origin
        if abs(center_x) < plot_range/4:
            center_x = 0
        if abs(center_y) < plot_range/4:
            center_y = 0
        
        ax3.set_xlim(center_x - plot_range/2, center_x + plot_range/2)
        ax3.set_ylim(center_y - plot_range/2, center_y + plot_range/2)
        
        # Animation elements
        mag_point, = ax1.plot([], [], 'bo', markersize=8)
        phase_point, = ax2.plot([], [], 'ro', markersize=8)
        
        circle = plt.Circle((0, 0), 0, fill=False, color='blue', alpha=0.5, linewidth=2)
        ax3.add_patch(circle)
        
        phase_line, = ax3.plot([], [], 'r--', alpha=0.7, linewidth=2)
        nyquist_point, = ax3.plot([], [], 'go', markersize=10, markeredgecolor='k', markeredgewidth=2)
        nyquist_trajectory, = ax3.plot([], [], 'g-', alpha=0.7, linewidth=2)
        
        text_box = ax3.text(0.02, 0.98, '', transform=ax3.transAxes, fontsize=10,
                           verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Pre-calculate frame indices
        indices = np.linspace(0, len(self.w_anim)-1, min(num_frames, len(self.w_anim)), dtype=int)
        
        def init():
            mag_point.set_data([], [])
            phase_point.set_data([], [])
            phase_line.set_data([], [])
            nyquist_point.set_data([], [])
            nyquist_trajectory.set_data([], [])
            circle.set_radius(0)
            text_box.set_text('')
            return mag_point, phase_point, circle, phase_line, nyquist_point, nyquist_trajectory, text_box
        
        def animate(i):
            idx = indices[i]
            
            freq = self.w_anim[idx]
            mag_db = self.mag_anim[idx]
            mag_lin = self.mag_linear_anim[idx]
            phase_deg = self.phase_anim[idx]
            phase_rad = np.radians(phase_deg)
            
            # Update Bode points
            mag_point.set_data([freq], [mag_db])
            phase_point.set_data([freq], [phase_deg])
            
            # Update circle
            circle.set_radius(mag_lin)
            
            # Update phase line
            x_end = mag_lin * np.cos(phase_rad)
            y_end = mag_lin * np.sin(phase_rad)
            phase_line.set_data([0, x_end], [0, y_end])
            
            # Update point
            nyquist_point.set_data([x_end], [y_end])
            
            # Update trajectory
            nyquist_trajectory.set_data(self.nyquist_real_anim[:idx+1], self.nyquist_imag_anim[:idx+1])
            
            # Update text
            text_box.set_text(f'Frequency: {freq:.2f} rad/s\nMagnitude: {mag_lin:.3f}\nPhase: {phase_deg:.1f}°')
            
            return mag_point, phase_point, circle, phase_line, nyquist_point, nyquist_trajectory, text_box
        
        # Create animation
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=len(indices), interval=50, blit=True)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Save animation to temp file
            writer = PillowWriter(fps=15)
            anim.save(temp_path, writer=writer)
            
            # Read temp file into BytesIO
            with open(temp_path, 'rb') as f:
                gif_data = f.read()
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Create BytesIO object
            gif_buffer = BytesIO(gif_data)
            gif_buffer.seek(0)
            
            plt.close(fig)
            return gif_buffer
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            plt.close(fig)
            st.error(f"Animation error: {str(e)}")
            return None
    
    def _create_simplified_animation(self, num_frames=40):
        """Create simplified animation for systems with poles on imaginary axis"""
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            st.error("Please install matplotlib and pillow")
            return None
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Bode plot on left
        ax1.semilogx(self.w_anim, self.mag_anim, 'b', alpha=0.3, linewidth=1)
        ax1.set_title('Bode - Magnitude', fontsize=12)
        ax1.set_ylabel('Magnitude [dB]')
        ax1.set_xlabel('Frequency [rad/s]')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([10**self.min_freq, 10**self.max_freq])
        
        # Simplified Nyquist on right
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', alpha=0.3)
        ax2.axvline(x=0, color='k', alpha=0.3)
        ax2.set_xlabel('Real')
        ax2.set_ylabel('Imaginary')
        ax2.set_title('Nyquist (Simplified)', fontsize=12)
        
        # Plot static Nyquist curve
        ax2.plot(self.nyquist_real_anim, self.nyquist_imag_anim, 'g-', alpha=0.5, linewidth=2)
        ax2.plot(-1, 0, 'rx', markersize=12, markeredgewidth=2)
        ax2.text(-1.1, 0.1, '(-1, 0)', fontsize=12, color='red')
        
        # Animation elements
        mag_point, = ax1.plot([], [], 'bo', markersize=8)
        nyquist_point, = ax2.plot([], [], 'go', markersize=10, markeredgecolor='k', markeredgewidth=2)
        
        text_box = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, fontsize=10,
                           verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set limits
        all_real = list(self.nyquist_real_anim) + [-1, 1, 0]
        all_imag = list(self.nyquist_imag_anim) + [-1, 1, 0]
        
        x_min, x_max = min(all_real), max(all_real)
        y_min, y_max = min(all_imag), max(all_imag)
        
        margin = 0.2
        plot_range = max(x_max - x_min, y_max - y_min, 0.1) * (1 + margin)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        ax2.set_xlim(center_x - plot_range/2, center_x + plot_range/2)
        ax2.set_ylim(center_y - plot_range/2, center_y + plot_range/2)
        
        # Pre-calculate frame indices
        indices = np.linspace(0, len(self.w_anim)-1, min(num_frames, len(self.w_anim)), dtype=int)
        
        def init():
            mag_point.set_data([], [])
            nyquist_point.set_data([], [])
            text_box.set_text('')
            return mag_point, nyquist_point, text_box
        
        def animate(i):
            idx = indices[i]
            
            freq = self.w_anim[idx]
            mag_db = self.mag_anim[idx]
            mag_lin = self.mag_linear_anim[idx]
            phase_deg = self.phase_anim[idx]
            phase_rad = np.radians(phase_deg)
            
            # Update Bode point
            mag_point.set_data([freq], [mag_db])
            
            # Update Nyquist point
            x_end = mag_lin * np.cos(phase_rad)
            y_end = mag_lin * np.sin(phase_rad)
            nyquist_point.set_data([x_end], [y_end])
            
            # Update text
            text_box.set_text(f'Frequency: {freq:.2f} rad/s\nMagnitude: {mag_lin:.3f}\nPhase: {phase_deg:.1f}°')
            
            return mag_point, nyquist_point, text_box
        
        # Create animation
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=len(indices), interval=50, blit=True)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Save animation to temp file
            writer = PillowWriter(fps=15)
            anim.save(temp_path, writer=writer)
            
            # Read temp file into BytesIO
            with open(temp_path, 'rb') as f:
                gif_data = f.read()
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Create BytesIO object
            gif_buffer = BytesIO(gif_data)
            gif_buffer.seek(0)
            
            plt.close(fig)
            return gif_buffer
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            plt.close(fig)
            st.error(f"Animation error: {str(e)}")
            return None

# ... [rest of the app code remains the same] ...
