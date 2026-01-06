import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
import base64
from io import BytesIO
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Bode & Nyquist Visualizer",
    page_icon="📈",
    layout="wide"
)

def parse_transfer_function(tf_str):
    """
    Fixed transfer function parser
    """
    # Clean input - replace ^ with ** for Python
    tf_str = tf_str.replace(' ', '').replace('^', '**')
    
    # Handle cases like 20*(s**2+1) or s*(s+100)
    if '/' in tf_str:
        num_str, den_str = tf_str.split('/')
    else:
        num_str = tf_str
        den_str = "1"
    
    # Remove outer parentheses
    num_str = num_str.strip('()')
    den_str = den_str.strip('()')
    
    # Parse polynomials
    num_coeffs = parse_polynomial_simple(num_str)
    den_coeffs = parse_polynomial_simple(den_str)
    
    return num_coeffs, den_coeffs

def parse_polynomial_simple(expr):
    """
    Simple but robust polynomial parser
    Handles: 20*(s**2+1), s*(s+100), s**2+3*s+2, etc.
    """
    # If empty or just a number
    if not expr or expr == '0':
        return [0.0]
    if expr.replace('.', '').replace('-', '').isdigit():
        return [float(expr)]
    
    # Expand multiplication if needed
    if '*' in expr and '(' in expr:
        # Handle 20*(s**2+1) or s*(s+100)
        parts = expr.split('*')
        if len(parts) == 2:
            coeff_str, poly_str = parts
            poly_str = poly_str.strip('()')
            
            # Parse the polynomial inside parentheses
            poly_coeffs = parse_polynomial_simple(poly_str)
            
            # Multiply all coefficients by the scalar
            try:
                coeff = float(coeff_str)
                return [coeff * c for c in poly_coeffs]
            except:
                # coeff_str might be 's', handle s*(s+100)
                if coeff_str == 's':
                    # s * (s + 100) = s² + 100s
                    return [1.0, 100.0, 0.0]
    
    # Handle direct polynomials like s**2+1 or s+1
    # Convert s**2 to s^2 for easier parsing
    expr = expr.replace('**', '^')
    
    # Split into terms
    terms = []
    current_term = ''
    for char in expr:
        if char in '+-' and current_term:
            terms.append(current_term)
            current_term = char
        else:
            current_term += char
    if current_term:
        terms.append(current_term)
    
    # If first term doesn't start with +/-, assume positive
    if terms and not terms[0].startswith(('-', '+')):
        terms[0] = '+' + terms[0]
    
    # Find highest power
    max_power = 0
    for term in terms:
        if 's^' in term:
            power = int(term.split('s^')[1])
            max_power = max(max_power, power)
        elif 's' in term and '^' not in term:
            max_power = max(max_power, 1)
    
    # Initialize coefficients
    coeffs = [0.0] * (max_power + 1)
    
    # Fill coefficients
    for term in terms:
        if term == '':
            continue
            
        sign = 1 if term[0] == '+' else -1
        term = term[1:]
        
        if 's^' in term:
            # Term like 2s^2 or s^2
            parts = term.split('s^')
            coeff_str = parts[0] if parts[0] else '1'
            try:
                coeff = float(coeff_str)
            except:
                coeff = 1.0
            power = int(parts[1])
        elif 's' in term:
            # Term like 2s or s
            parts = term.split('s')
            coeff_str = parts[0] if parts[0] else '1'
            try:
                coeff = float(coeff_str)
            except:
                coeff = 1.0
            power = 1
        else:
            # Constant term
            coeff = float(term)
            power = 0
        
        coeffs[max_power - power] = sign * coeff
    
    return coeffs

class NyquistVisualizer:
    def __init__(self, num=None, den=None):
        if num is not None and den is not None:
            self.sys = signal.TransferFunction(num, den)
        else:
            raise ValueError("Must provide num and den coefficients")
        
        # Frequency range
        self.w = np.logspace(-2, 2, 400)
        
        # Calculate frequency response
        self.w, self.mag, self.phase = signal.bode(self.sys, self.w)
        self.mag_linear = 10**(self.mag / 20)
        
        # Calculate Nyquist points
        phase_rad = np.radians(self.phase)
        self.nyquist_real = self.mag_linear * np.cos(phase_rad)
        self.nyquist_imag = self.mag_linear * np.sin(phase_rad)
    
    def plot_bode(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        ax1.semilogx(self.w, self.mag, 'b', linewidth=2)
        ax1.set_ylabel('Magnitude [dB]')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.set_title('Bode Diagram')
        
        ax2.semilogx(self.w, self.phase, 'r', linewidth=2)
        ax2.set_ylabel('Phase [deg]')
        ax2.set_xlabel('Frequency [rad/s]')
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_complete_nyquist(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.plot(self.nyquist_real, self.nyquist_imag, 'b-', linewidth=2)
        ax.plot(self.nyquist_real, -self.nyquist_imag, 'b--', linewidth=1, alpha=0.5)
        ax.plot(-1, 0, 'rx', markersize=10, markeredgewidth=2)
        ax.text(-1.1, 0.1, '(-1, 0)', fontsize=10)
        
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title('Nyquist Diagram')
        
        plt.tight_layout()
        return fig
    
    def create_construction_animation(self, num_frames=30):
        """Create an animation showing how Bode plots become Nyquist plot"""
        try:
            from matplotlib.animation import FuncAnimation
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            st.error("Please install matplotlib: pip install matplotlib")
            return None
        
        # Create figure with subplots
        fig = plt.figure(figsize=(12, 8))
        
        # Bode plots on left
        ax1 = plt.subplot2grid((2, 2), (0, 0))  # Magnitude
        ax2 = plt.subplot2grid((2, 2), (1, 0))  # Phase
        ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)  # Nyquist
        
        # Setup Bode plots
        ax1.semilogx(self.w, self.mag, 'b', alpha=0.3, linewidth=1)
        ax1.set_title('Bode - Magnitude')
        ax1.set_ylabel('Magnitude [dB]')
        ax1.grid(True, alpha=0.3)
        
        ax2.semilogx(self.w, self.phase, 'r', alpha=0.3, linewidth=1)
        ax2.set_title('Bode - Phase')
        ax2.set_ylabel('Phase [deg]')
        ax2.set_xlabel('Frequency [rad/s]')
        ax2.grid(True, alpha=0.3)
        
        # Setup Nyquist plot
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', alpha=0.3)
        ax3.axvline(x=0, color='k', alpha=0.3)
        ax3.set_xlabel('Real')
        ax3.set_ylabel('Imaginary')
        ax3.set_title('Nyquist Construction')
        
        # Plot limits
        all_real = list(self.nyquist_real) + [-1, 1]
        all_imag = list(self.nyquist_imag) + [-1, 1]
        x_min, x_max = min(all_real), max(all_real)
        y_min, y_max = min(all_imag), max(all_imag)
        margin = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax3.set_xlim(x_min - margin*x_range, x_max + margin*x_range)
        ax3.set_ylim(y_min - margin*y_range, y_max + margin*y_range)
        
        # Animation elements
        mag_point, = ax1.plot([], [], 'bo', markersize=6)
        phase_point, = ax2.plot([], [], 'ro', markersize=6)
        
        circle = plt.Circle((0, 0), 0, fill=False, color='blue', alpha=0.5)
        ax3.add_patch(circle)
        
        phase_line, = ax3.plot([], [], 'r--', alpha=0.7)
        nyquist_point, = ax3.plot([], [], 'go', markersize=8, markeredgecolor='k')
        nyquist_trajectory, = ax3.plot([], [], 'g-', alpha=0.7)
        
        text_box = ax3.text(0.02, 0.98, '', transform=ax3.transAxes, fontsize=9,
                           verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Select frames
        indices = np.linspace(0, len(self.w)-1, num_frames, dtype=int)
        
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
            
            freq = self.w[idx]
            mag_db = self.mag[idx]
            mag_lin = self.mag_linear[idx]
            phase_deg = self.phase[idx]
            phase_rad = np.radians(phase_deg)
            
            # Update Bode points
            mag_point.set_data([freq], [mag_db])
            phase_point.set_data([freq], [phase_deg])
            
            # Update circle (amplitude)
            circle.set_radius(mag_lin)
            
            # Update phase line
            x_end = mag_lin * np.cos(phase_rad)
            y_end = mag_lin * np.sin(phase_rad)
            phase_line.set_data([0, x_end], [0, y_end])
            
            # Update Nyquist point
            nyquist_point.set_data([x_end], [y_end])
            
            # Update trajectory
            nyquist_trajectory.set_data(self.nyquist_real[:idx+1], self.nyquist_imag[:idx+1])
            
            # Update text
            text_box.set_text(f'f = {freq:.2f} rad/s\n|G| = {mag_lin:.2f}\n∠ = {phase_deg:.0f}°')
            
            return mag_point, phase_point, circle, phase_line, nyquist_point, nyquist_trajectory, text_box
        
        # Create animation
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=len(indices), interval=100, blit=True)
        
        # Save as GIF
        try:
            from matplotlib.animation import PillowWriter
            
            gif_buffer = BytesIO()
            writer = PillowWriter(fps=10)
            anim.save(gif_buffer, writer=writer, format='gif')
            gif_buffer.seek(0)
            
            plt.close(fig)
            return gif_buffer
        except ImportError:
            st.error("Please install pillow for GIF creation: pip install pillow")
            plt.close(fig)
            return None

# Main App
st.title("📈 Bode & Nyquist Visualizer")

# Input section
st.markdown("### Enter Transfer Function")
tf_input = st.text_input(
    "Transfer Function (s-domain):",
    value="1/(s+1)",
    help="Examples: 20*(s^2+1), s*(s+100), 1/(s+1), 1/(s^2+0.5*s+1)"
)

parse_clicked = st.button("Generate Plots", type="primary")

# Test cases
if st.checkbox("Show test cases"):
    test_cases = [
        ("20*(s^2+1)", "Should give: num=[20.0, 0.0, 20.0]"),
        ("s*(s+100)", "Should give: num=[1.0, 100.0, 0.0]"),
        ("1/(s+1)", "Simple first order"),
        ("1/(s^2+0.5*s+1)", "Second order system"),
    ]
    
    for tf, desc in test_cases:
        if st.button(f"Test: {tf}"):
            try:
                num, den = parse_transfer_function(tf)
                st.write(f"{tf}:")
                st.write(f"  Numerator: {num}")
                st.write(f"  Denominator: {den}")
            except Exception as e:
                st.error(f"Error: {e}")

# Process input
if parse_clicked or ('tf_input' in st.session_state and st.session_state.tf_input == tf_input):
    try:
        num_coeffs, den_coeffs = parse_transfer_function(tf_input)
        
        # Store in session state
        st.session_state.tf_input = tf_input
        st.session_state.num_coeffs = num_coeffs
        st.session_state.den_coeffs = den_coeffs
        
        # Display what was parsed
        st.success("✓ Transfer function parsed successfully!")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Numerator:** {num_coeffs}")
        with col2:
            st.info(f"**Denominator:** {den_coeffs}")
        
        # Create visualizer
        visualizer = NyquistVisualizer(num=num_coeffs, den=den_coeffs)
        
        # Tabs for visualizations
        tab1, tab2, tab3 = st.tabs(["Bode Plot", "Nyquist Diagram", "Nyquist Construction Animation"])
        
        with tab1:
            fig_bode = visualizer.plot_bode()
            st.pyplot(fig_bode)
            plt.close(fig_bode)
        
        with tab2:
            fig_nyquist = visualizer.plot_complete_nyquist()
            st.pyplot(fig_nyquist)
            plt.close(fig_nyquist)
            
            # Stability info
            with st.expander("About Nyquist Stability"):
                st.markdown("""
                The **(-1, 0)** point is critical for stability.
                
                **Nyquist Criterion:**
                - Clockwise encirclements of (-1, 0) indicate instability
                - Number of encirclements = unstable poles
                - No encirclement → stable (if open-loop stable)
                """)
        
        with tab3:
            st.markdown("### How Bode Plots Become a Nyquist Diagram")
            st.markdown("This animation shows the construction of the Nyquist plot from Bode diagrams.")
            
            col1, col2 = st.columns(2)
            with col1:
                num_frames = st.slider("Animation frames:", 20, 60, 30)
            with col2:
                if st.button("Generate Animation", type="primary"):
                    with st.spinner("Creating animation... (this takes a few seconds)"):
                        gif_buffer = visualizer.create_construction_animation(num_frames)
                        
                        if gif_buffer:
                            # Display GIF
                            st.markdown("**Animation:**")
                            
                            # Convert to base64 for display
                            gif_base64 = base64.b64encode(gif_buffer.read()).decode()
                            gif_buffer.seek(0)
                            
                            # Display with HTML for better control
                            st.markdown(
                                f'<img src="data:image/gif;base64,{gif_base64}" alt="nyquist animation" width="800">',
                                unsafe_allow_html=True
                            )
                            
                            # Download button
                            st.download_button(
                                label="Download GIF",
                                data=gif_buffer,
                                file_name="nyquist_construction.gif",
                                mime="image/gif"
                            )
                        else:
                            st.error("Could not create animation. Make sure pillow is installed: pip install pillow")
            
            # Explanation
            st.markdown("""
            **How it works:**
            1. **Blue circle** shows the magnitude (amplitude) from Bode plot
            2. **Red line** shows the phase angle from Bode plot  
            3. **Green point** is where they intersect - this is one point on the Nyquist plot
            4. **Green line** traces out the complete Nyquist plot as frequency increases
            
            The animation shows how each frequency point from Bode plots maps to a point in the complex plane.
            """)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("""
        **Try these formats:**
        - `20*(s^2+1)`
        - `s*(s+100)`
        - `1/(s+1)`
        - `1/(s^2+0.5*s+1)`
        - `(s+1)/(s^2+2*s+3)`
        - `s^2+3*s+2`
        
        **Note:** Use `s^2` for s², `*` for multiplication
        """)

# Quick examples
st.markdown("---")
st.markdown("**Try these examples:**")

examples = [
    ("20*(s^2+1)", "Gain × (s² + 1)"),
    ("s*(s+100)", "s(s + 100)"),
    ("1/(s+1)", "First order system"),
    ("1/(s^2+0.5*s+1)", "Second order system"),
]

cols = st.columns(len(examples))
for i, (example, desc) in enumerate(examples):
    with cols[i]:
        if st.button(f"{example}", help=desc, key=f"ex{i}"):
            st.session_state.tf_input = example
            st.rerun()

st.markdown("""
**Enter your transfer function above, then click 'Generate Plots'.**

The visualizer will show:
1. **Bode plot** - Frequency response (magnitude and phase)
2. **Nyquist diagram** - Complete polar plot  
3. **Nyquist construction animation** - How Bode plots become Nyquist plot
""")
