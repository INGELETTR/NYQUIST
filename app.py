import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import control as ct
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

class ControlVisualizer:
    def __init__(self, num=None, den=None, min_freq=-2, max_freq=2):
        if num is not None and den is not None:
            # Create transfer function using control library
            self.sys = ct.TransferFunction(num, den)
        else:
            raise ValueError("Must provide num and den coefficients")
        
        # Store frequency range
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        # Store coefficients
        self.num_coeffs = num
        self.den_coeffs = den
        
        # Generate frequency array for Bode
        self.w = np.logspace(min_freq, max_freq, 1000)
        
        # Calculate Bode plot data using control library
        self.mag, self.phase, self.w_bode = ct.bode(self.sys, self.w, plot=False)
        # Convert magnitude to dB
        self.mag_db = 20 * np.log10(self.mag)
        
        # Get Nyquist data using control library
        try:
            # Try to get Nyquist data from control library
            self.nyquist_real, self.nyquist_imag, self.nyquist_omega = ct.nyquist(self.sys, plot=False)
        except:
            # Fallback: compute manually
            nyquist_response = self._compute_nyquist_response(self.w)
            self.nyquist_real = np.real(nyquist_response)
            self.nyquist_imag = np.imag(nyquist_response)
            self.nyquist_omega = self.w
        
        # For animation (fewer points for speed)
        self.w_anim = np.logspace(min_freq, max_freq, 200)
        self.mag_anim, self.phase_anim, _ = ct.bode(self.sys, self.w_anim, plot=False)
        self.mag_db_anim = 20 * np.log10(self.mag_anim)
        
        # Calculate complex response for animation
        self.nyquist_response_anim = self._compute_nyquist_response(self.w_anim)
        self.nyquist_real_anim = np.real(self.nyquist_response_anim)
        self.nyquist_imag_anim = np.imag(self.nyquist_response_anim)
    
    def _compute_nyquist_response(self, frequencies):
        """Compute frequency response G(jω) for given frequencies"""
        response = []
        for w in frequencies:
            # Evaluate transfer function at s = jω
            resp = self.sys(1j * w)
            response.append(resp)
        return np.array(response)
    
    def get_symbolic_expressions(self, num_coeffs, den_coeffs):
        """Compute symbolic real and imaginary parts of G(jω)"""
        try:
            # Define symbols
            s = sp.symbols('s')
            omega = sp.symbols('omega', real=True)
            
            # Create numerator and denominator polynomials
            num_poly = sum(c * s**i for i, c in enumerate(reversed(num_coeffs)))
            den_poly = sum(c * s**i for i, c in enumerate(reversed(den_coeffs)))
            
            # Create transfer function
            G = num_poly / den_poly
            
            # Get G(jω) and simplify
            G_jw = sp.simplify(G.subs(s, sp.I * omega))
            
            # Use as_real_imag() method to separate
            real_part, imag_part = G_jw.as_real_imag()
            
            # Simplify further
            real_part = sp.simplify(real_part)
            imag_part = sp.simplify(imag_part)
            
            # Convert to LaTeX
            G_latex = sp.latex(G)
            real_latex = sp.latex(real_part)
            imag_latex = sp.latex(imag_part)
            
            return G_latex, real_latex, imag_latex
            
        except Exception as e:
            st.error(f"Error computing symbolic expressions: {str(e)}")
            return None, None, None
    
    def plot_bode(self):
        """Plot Bode diagram using control library's bode function"""
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Use control library's bode function for proper plotting
        # Magnitude plot
        ax1.semilogx(self.w_bode, self.mag_db, 'b', linewidth=2)
        ax1.set_ylabel('Magnitude [dB]', fontsize=12)
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.set_title(f'Bode Diagram', fontsize=14)
        ax1.set_xlim([10**self.min_freq, 10**self.max_freq])
        
        # Add magnitude grid lines
        ax1.minorticks_on()
        ax1.grid(True, which='major', linestyle='-', alpha=0.7)
        ax1.grid(True, which='minor', linestyle=':', alpha=0.5)
        
        # Phase plot
        ax2.semilogx(self.w_bode, self.phase, 'r', linewidth=2)
        ax2.set_ylabel('Phase [deg]', fontsize=12)
        ax2.set_xlabel('Frequency [rad/s]', fontsize=12)
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.set_title(f'Phase Response', fontsize=14)
        ax2.set_xlim([10**self.min_freq, 10**self.max_freq])
        
        # Add phase grid lines
        ax2.minorticks_on()
        ax2.grid(True, which='major', linestyle='-', alpha=0.7)
        ax2.grid(True, which='minor', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_nyquist(self):
        """Plot Nyquist diagram using control library's nyquist function"""
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot the Nyquist curve from control library
        ax.plot(self.nyquist_real, self.nyquist_imag, 'b-', linewidth=2, label='ω: 0 → ∞')
        
        # Also plot the mirror for negative frequencies (complex conjugate)
        ax.plot(self.nyquist_real, -self.nyquist_imag, 'b--', linewidth=1, alpha=0.5, label='ω: -∞ → 0')
        
        # Mark (-1, 0) point
        ax.plot(-1, 0, 'rx', markersize=12, markeredgewidth=2, label='(-1, 0)', zorder=5)
        ax.text(-1.1, 0.1, '(-1, 0)', fontsize=12, color='red', zorder=5)
        
        # Add arrow showing direction (if we have enough points)
        if len(self.nyquist_real) > 10:
            mid_idx = len(self.nyquist_real) // 2
            ax.annotate('', 
                       xy=(self.nyquist_real[mid_idx], self.nyquist_imag[mid_idx]),
                       xytext=(self.nyquist_real[mid_idx-5], self.nyquist_imag[mid_idx-5]),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=1))
        
        # Set axis properties
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Real', fontsize=12)
        ax.set_ylabel('Imaginary', fontsize=12)
        ax.set_aspect('equal')
        ax.set_title('Nyquist Diagram', fontsize=14)
        
        # Set appropriate limits
        self._set_nyquist_limits(ax)
        
        ax.legend(loc='best')
        plt.tight_layout()
        return fig
    
    def _set_nyquist_limits(self, ax):
        """Set appropriate axis limits for Nyquist plot"""
        # Get data bounds
        x_data = np.concatenate([self.nyquist_real, [-1, 1, 0]])
        y_data = np.concatenate([self.nyquist_imag, [-1, 1, 0]])
        
        # Check if we have large values (infinity case)
        max_abs_real = np.max(np.abs(self.nyquist_real))
        max_abs_imag = np.max(np.abs(self.nyquist_imag))
        
        if max_abs_real > 100 or max_abs_imag > 100:
            # System goes to infinity - show reasonable window
            # Find finite points
            finite_mask = (np.abs(self.nyquist_real) < 100) & (np.abs(self.nyquist_imag) < 100)
            
            if np.any(finite_mask):
                finite_real = self.nyquist_real[finite_mask]
                finite_imag = self.nyquist_imag[finite_mask]
                x_min, x_max = np.min(finite_real), np.max(finite_real)
                y_min, y_max = np.min(finite_imag), np.max(finite_imag)
            else:
                x_min, x_max = -10, 10
                y_min, y_max = -10, 10
        else:
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
    
    def create_animation(self, num_frames=40):
        """Create animation showing Bode to Nyquist transformation"""
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            st.error("Please install matplotlib and pillow")
            return None
        
        # Create figure with subplots
        fig = plt.figure(figsize=(14, 10))
        
        # Bode plots on left
        ax1 = plt.subplot2grid((2, 2), (0, 0))  # Magnitude
        ax2 = plt.subplot2grid((2, 2), (1, 0))  # Phase
        ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)  # Nyquist
        
        # Setup Bode magnitude plot
        ax1.semilogx(self.w_anim, self.mag_db_anim, 'b', alpha=0.3, linewidth=1)
        ax1.set_title('Bode - Magnitude', fontsize=12)
        ax1.set_ylabel('Magnitude [dB]')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([10**self.min_freq, 10**self.max_freq])
        
        # Setup Bode phase plot
        ax2.semilogx(self.w_anim, self.phase_anim, 'r', alpha=0.3, linewidth=1)
        ax2.set_title('Bode - Phase', fontsize=12)
        ax2.set_ylabel('Phase [deg]')
        ax2.set_xlabel('Frequency [rad/s]')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([10**self.min_freq, 10**self.max_freq])
        
        # Setup Nyquist plot
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', alpha=0.3)
        ax3.axvline(x=0, color='k', alpha=0.3)
        ax3.set_xlabel('Real')
        ax3.set_ylabel('Imaginary')
        ax3.set_title('Nyquist Construction Animation', fontsize=12)
        
        # Plot static Nyquist curve
        ax3.plot(self.nyquist_real_anim, self.nyquist_imag_anim, 'g-', alpha=0.3, linewidth=1)
        ax3.plot(-1, 0, 'rx', markersize=12, markeredgewidth=2)
        ax3.text(-1.1, 0.1, '(-1, 0)', fontsize=12, color='red')
        
        # Set Nyquist plot limits
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
        
        # Circle for magnitude in Nyquist plot
        circle = plt.Circle((0, 0), 0, fill=False, color='blue', alpha=0.5, linewidth=2)
        ax3.add_patch(circle)
        
        # Phase line in Nyquist plot
        phase_line, = ax3.plot([], [], 'r--', alpha=0.7, linewidth=2)
        
        # Current point in Nyquist plot
        nyquist_point, = ax3.plot([], [], 'go', markersize=10, markeredgecolor='k', markeredgewidth=2)
        
        # Trajectory in Nyquist plot
        nyquist_trajectory, = ax3.plot([], [], 'g-', alpha=0.7, linewidth=2)
        
        # Text box for information
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
            mag_db = self.mag_db_anim[idx]
            mag_lin = self.mag_anim[idx]
            phase_deg = self.phase_anim[idx]
            phase_rad = np.radians(phase_deg)
            
            # Update Bode points
            mag_point.set_data([freq], [mag_db])
            phase_point.set_data([freq], [phase_deg])
            
            # Update circle (magnitude in complex plane)
            circle.set_radius(mag_lin)
            
            # Update phase line
            x_end = mag_lin * np.cos(phase_rad)
            y_end = mag_lin * np.sin(phase_rad)
            phase_line.set_data([0, x_end], [0, y_end])
            
            # Update current point
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

# Main App
st.title("📈 Bode & Nyquist Visualizer")

# Parse coefficients function
def parse_coeffs(coefficient_string):
    """Parse comma-separated coefficients into list of floats"""
    if not coefficient_string:
        return [1.0]
    
    # Remove brackets if present
    coeff_str = coefficient_string.strip("[]")
    
    # Split by comma and convert to float
    try:
        coeffs = [float(x.strip()) for x in coeff_str.split(',') if x.strip()]
        return coeffs if coeffs else [1.0]
    except ValueError:
        st.error(f"Invalid coefficients: {coefficient_string}")
        return [1.0]

# Input section
st.markdown("### Enter Transfer Function Coefficients")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Numerator Coefficients**")
    st.markdown("Enter coefficients for highest to lowest power of s")
    
    # Use session state to remember the input
    if 'num_input' not in st.session_state:
        st.session_state.num_input = "1"
    
    num_input = st.text_input(
        "Numerator [aₙ, aₙ₋₁, ..., a₁, a₀]:",
        value=st.session_state.num_input,
        key="num_input_widget",
        help="Comma-separated coefficients, e.g., '1' for 1, '20,0,20' for 20s²+20"
    )
    
    # Update session state
    st.session_state.num_input = num_input

with col2:
    st.markdown("**Denominator Coefficients**")
    st.markdown("Enter coefficients for highest to lowest power of s")
    
    if 'den_input' not in st.session_state:
        st.session_state.den_input = "1, 1"
    
    den_input = st.text_input(
        "Denominator [bₘ, bₘ₋₁, ..., b₁, b₀]:",
        value=st.session_state.den_input,
        key="den_input_widget",
        help="Comma-separated coefficients, e.g., '1,1' for s+1, '1,0.5,1' for s²+0.5s+1"
    )
    
    # Update session state
    st.session_state.den_input = den_input

# Sidebar for frequency range
with st.sidebar:
    st.markdown("### Frequency Range")
    st.markdown("Set the frequency range for the Bode plot (in decades):")
    
    col1, col2 = st.columns(2)
    with col1:
        if 'min_freq' not in st.session_state:
            st.session_state.min_freq = -2.0
        
        min_freq = st.number_input("Min frequency (10^x)", 
                                  value=st.session_state.min_freq, 
                                  min_value=-5.0, 
                                  max_value=5.0, 
                                  step=0.5,
                                  key="min_freq_input",
                                  help="Minimum frequency exponent (10^min)")
        st.session_state.min_freq = min_freq
        
    with col2:
        if 'max_freq' not in st.session_state:
            st.session_state.max_freq = 2.0
        
        max_freq = st.number_input("Max frequency (10^x)", 
                                  value=st.session_state.max_freq, 
                                  min_value=-5.0, 
                                  max_value=5.0, 
                                  step=0.5,
                                  key="max_freq_input",
                                  help="Maximum frequency exponent (10^max)")
        st.session_state.max_freq = max_freq
    
    if min_freq >= max_freq:
        st.error("Minimum frequency must be less than maximum frequency")
        min_freq, max_freq = -2.0, 2.0
        st.session_state.min_freq = min_freq
        st.session_state.max_freq = max_freq
    
    st.markdown("---")
    st.markdown("### Quick Examples")
    
    examples = [
        ("First Order", "[1]", "[1, 1]", "1/(s+1)"),
        ("Second Order", "[1]", "[1, 0.5, 1]", "1/(s²+0.5s+1)"),
        ("Integrator", "[1]", "[1, 0]", "1/s"),
        ("Differentiator", "[1, 0]", "[1]", "s"),
        ("20*(s²+1)", "[20, 0, 20]", "[1]", "20s²+20"),
        ("s*(s+100)", "[1, 100, 0]", "[1]", "s²+100s"),
    ]
    
    for name, num_ex, den_ex, desc in examples:
        if st.button(f"{name}: {desc}", key=f"ex_{name}"):
            st.session_state.num_input = num_ex.strip("[]").replace(" ", "")
            st.session_state.den_input = den_ex.strip("[]").replace(" ", "")
            st.session_state.visualizer = None  # Reset visualizer
            st.rerun()

# Main generate button
if st.button("Generate Plots", type="primary", use_container_width=True):
    try:
        # Parse coefficients
        num_coeffs = parse_coeffs(st.session_state.num_input)
        den_coeffs = parse_coeffs(st.session_state.den_input)
        
        # Store in session state
        st.session_state.num_coeffs = num_coeffs
        st.session_state.den_coeffs = den_coeffs
        
        # Create and store visualizer using control library
        st.session_state.visualizer = ControlVisualizer(
            num=num_coeffs, 
            den=den_coeffs, 
            min_freq=st.session_state.min_freq, 
            max_freq=st.session_state.max_freq
        )
        
        # Reset animation flag
        st.session_state.show_animation = False
        
        # Show success
        st.success("✓ Plots generated successfully!")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Make sure your coefficients are valid numbers separated by commas.")

# If we have a visualizer in session state, show the plots
if st.session_state.visualizer is not None:
    visualizer = st.session_state.visualizer
    
    # Display the transfer function and real/imag parts
    def format_poly(coeffs, var='s'):
        """Format coefficients as polynomial string"""
        n = len(coeffs)
        terms = []
        for i, coeff in enumerate(coeffs):
            if abs(coeff) > 1e-10:
                power = n - i - 1
                if power == 0:
                    terms.append(f"{coeff:.4g}")
                elif power == 1:
                    terms.append(f"{coeff:.4g}{var}")
                else:
                    terms.append(f"{coeff:.4g}{var}^{power}")
        if not terms:
            return "0"
        return " + ".join(terms).replace("+ -", "- ")
    
    # Get symbolic expressions
    G_latex, real_latex, imag_latex = visualizer.get_symbolic_expressions(
        st.session_state.num_coeffs, st.session_state.den_coeffs
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Transfer Function:**")
        st.markdown(f"$$G(s) = \\frac{{{format_poly(st.session_state.num_coeffs)}}}{{{format_poly(st.session_state.den_coeffs)}}}$$")
    
    with col2:
        if G_latex:
            st.markdown(f"**Transfer Function (LaTeX):**")
            st.markdown(f"$$G(s) = {G_latex}$$")
    
    if real_latex and imag_latex:
        st.markdown("**Real and Imaginary Parts:**")
        st.markdown(f"$$\\text{{Re}}[G(j\\omega)] = {real_latex}$$")
        st.markdown(f"$$\\text{{Im}}[G(j\\omega)] = {imag_latex}$$")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Bode Plot", "Nyquist Diagram", "Animation"])
    
    with tab1:
        fig_bode = visualizer.plot_bode()
        st.pyplot(fig_bode)
        plt.close(fig_bode)
        
        st.info(f"Frequency range: 10^{{{st.session_state.min_freq}}} to 10^{{{st.session_state.max_freq}}} rad/s")
    
    with tab2:
        fig_nyquist = visualizer.plot_nyquist()
        st.pyplot(fig_nyquist)
        plt.close(fig_nyquist)
        
        with st.expander("Stability Information"):
            st.markdown("""
            **Nyquist Stability Criterion:**
            
            The **(-1, 0)** point (marked in red) is critical for stability analysis.
            
            1. **Clockwise encirclements** of (-1, 0) indicate **instability**
            2. **No encirclement** → system is stable (if open-loop stable)
            3. **Number of encirclements** = number of unstable poles
            
            The plot shows both positive frequencies (solid line) and 
            negative frequencies (dashed line, complex conjugate).
            
            **Note:** The control library handles poles on the imaginary axis and infinity better.
            """)
    
    with tab3:
        st.markdown("### Bode to Nyquist Animation")
        st.markdown("Watch how the Bode magnitude and phase combine to form the Nyquist plot.")
        
        col1, col2 = st.columns(2)
        with col1:
            if 'num_frames' not in st.session_state:
                st.session_state.num_frames = 40
            
            num_frames = st.slider("Number of animation frames", 20, 60, 
                                  st.session_state.num_frames, key="num_frames_slider")
            st.session_state.num_frames = num_frames
        
        with col2:
            if st.button("🎬 Generate Animation", type="primary", use_container_width=True, key="anim_btn"):
                st.session_state.show_animation = True
        
        # Show animation if flag is set
        if st.session_state.show_animation:
            with st.spinner("Creating animation..."):
                gif_buffer = visualizer.create_animation(st.session_state.num_frames)
                
                if gif_buffer:
                    # Display the GIF
                    st.markdown("### Bode to Nyquist Construction")
                    
                    # Convert to base64 for embedding
                    gif_base64 = base64.b64encode(gif_buffer.read()).decode()
                    gif_buffer.seek(0)
                    
                    # Display with HTML (centered)
                    st.markdown(
                        f'<div style="text-align: center;">'
                        f'<img src="data:image/gif;base64,{gif_base64}" alt="nyquist animation" width="800">'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Download button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            "⬇️ Download GIF",
                            data=gif_buffer,
                            file_name="bode_to_nyquist.gif",
                            mime="image/gif",
                            use_container_width=True
                        )
                else:
                    st.error("Could not create animation. Make sure pillow is installed: pip install pillow")
        
        # Always show explanation
        with st.expander("How the animation works"):
            st.markdown("""
            **Animation Elements:**
            
            - **Blue circle**: Radius = magnitude |G(jω)| from Bode plot
            - **Red dashed line**: Angle = phase ∠G(jω) from Bode plot  
            - **Green point**: Intersection = G(jω) in complex plane (Nyquist point)
            - **Green line**: Traces the complete Nyquist plot as ω increases
            
            **What's happening:**
            1. As frequency ω increases, the magnitude and phase change
            2. The blue circle grows/shrinks based on magnitude
            3. The red line rotates based on phase
            4. Their intersection traces the Nyquist plot
            
            This shows how frequency response (Bode) transforms to complex plane (Nyquist).
            """)

# Clear button to reset everything
if st.button("Clear All", type="secondary"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Quick help
with st.expander("📋 How to enter coefficients"):
    st.markdown("""
    **Coefficient Format:**
    
    Enter coefficients from highest power to constant term, separated by commas.
    
    **Examples:**
    
    | System | Transfer Function | Numerator | Denominator |
    |--------|-------------------|-----------|-------------|
    | First order | 1/(s+1) | `1` | `1, 1` |
    | Second order | 1/(s²+0.5s+1) | `1` | `1, 0.5, 1` |
    | Integrator | 1/s | `1` | `1, 0` |
    | Differentiator | s | `1, 0` | `1` |
    | 20*(s²+1) | 20s²+20 | `20, 0, 20` | `1` |
    | s*(s+100) | s²+100s | `1, 100, 0` | `1` |
    
    **Note:** The denominator must have at least one non-zero coefficient.
    """)

# Requirements info
with st.expander("🔧 Installation"):
    st.code("pip install streamlit numpy matplotlib control sympy pillow")
    st.markdown("""
    **Note:** The `control` library is specifically designed for control systems and provides
    better handling of Nyquist plots, especially for systems with poles on the imaginary axis.
    """)
