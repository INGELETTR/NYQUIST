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
    Fixed transfer function parser that actually works
    """
    # Clean input
    tf_str = tf_str.replace(' ', '').replace('^', '**')
    
    # Handle division
    if '/' in tf_str:
        num_str, den_str = tf_str.split('/')
    else:
        num_str = tf_str
        den_str = "1"
    
    # Parse numerator and denominator
    num_coeffs = parse_expression(num_str)
    den_coeffs = parse_expression(den_str)
    
    return num_coeffs, den_coeffs

def parse_expression(expr):
    """
    Parse an expression like 20*(s**2+1) or s*(s+100)
    """
    # If it's just a constant
    try:
        return [float(expr)]
    except:
        pass
    
    # Handle multiplication with parentheses
    if '*' in expr and '(' in expr and ')' in expr:
        # Find the multiplier and the polynomial
        if expr.startswith('('):
            # Case: (s+1)*(s+2)
            parts = [part.strip('()') for part in expr.split('*')]
            # Multiply polynomials
            result = [1.0]  # Start with 1
            for part in parts:
                poly = parse_simple_poly(part)
                result = multiply_polynomials(result, poly)
            return result
        else:
            # Case: 20*(s**2+1)
            multiplier_str, poly_str = expr.split('*', 1)
            poly_str = poly_str.strip('()')
            
            # Parse multiplier
            try:
                multiplier = float(multiplier_str)
            except:
                # If multiplier is 's'
                if multiplier_str == 's':
                    poly = parse_simple_poly(poly_str)
                    return multiply_polynomials([1.0, 0.0], poly)  # s * poly
                else:
                    raise ValueError(f"Can't parse multiplier: {multiplier_str}")
            
            # Parse polynomial
            poly = parse_simple_poly(poly_str)
            
            # Multiply all coefficients by multiplier
            return [multiplier * coeff for coeff in poly]
    
    # Handle simple polynomial
    return parse_simple_poly(expr)

def parse_simple_poly(expr):
    """
    Parse a simple polynomial like s**2+1 or s+100
    """
    if expr == 's':
        return [1.0, 0.0]
    elif expr == '0':
        return [0.0]
    
    # Replace ** with ^ for easier parsing
    expr = expr.replace('**', '^')
    
    # Find all terms
    terms = []
    current = ''
    for char in expr:
        if char in '+-' and current:
            terms.append(current)
            current = char
        else:
            current += char
    if current:
        terms.append(current)
    
    # If first term doesn't start with sign, assume positive
    if terms and not terms[0].startswith(('+', '-')):
        terms[0] = '+' + terms[0]
    
    # Find highest power
    max_power = 0
    for term in terms:
        if 's^' in term:
            power = int(term.split('s^')[1])
            max_power = max(max_power, power)
        elif term.endswith('s') or ('s' in term and '*' not in term):
            max_power = max(max_power, 1)
    
    # Initialize coefficients
    coeffs = [0.0] * (max_power + 1)
    
    # Parse each term
    for term in terms:
        if not term or term in ['+', '-']:
            continue
            
        sign = 1 if term[0] == '+' else -1
        term = term[1:]
        
        if 's^' in term:
            # Term like 2s^2
            parts = term.split('s^')
            coeff = float(parts[0]) if parts[0] else 1.0
            power = int(parts[1])
        elif 's' in term:
            # Term like 2s or s
            if term == 's':
                coeff = 1.0
                power = 1
            elif term.endswith('s'):
                coeff = float(term[:-1]) if term[:-1] else 1.0
                power = 1
            elif '*' in term:
                coeff = float(term.split('*')[0])
                power = 1
        else:
            # Constant
            coeff = float(term)
            power = 0
        
        coeffs[max_power - power] = sign * coeff
    
    return coeffs

def multiply_polynomials(poly1, poly2):
    """
    Multiply two polynomials
    """
    result = [0.0] * (len(poly1) + len(poly2) - 1)
    for i, coeff1 in enumerate(poly1):
        for j, coeff2 in enumerate(poly2):
            result[i + j] += coeff1 * coeff2
    return result

class FastNyquistVisualizer:
    def __init__(self, num=None, den=None):
        if num is not None and den is not None:
            self.sys = signal.TransferFunction(num, den)
        else:
            raise ValueError("Must provide num and den coefficients")
        
        # Frequency range for display (more points for smooth plots)
        self.w = np.logspace(-2, 2, 500)
        
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
    
    def create_fast_animation(self, num_frames=40):
        """
        Create fast animation using precomputed frames
        """
        try:
            from matplotlib.animation import FuncAnimation
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            st.error("Please install matplotlib")
            return None
        
        # Use fewer points for animation for speed
        anim_w = np.logspace(-2, 2, 200)  # Fewer points for animation
        anim_w, anim_mag, anim_phase = signal.bode(self.sys, anim_w)
        anim_mag_linear = 10**(anim_mag / 20)
        anim_phase_rad = np.radians(anim_phase)
        anim_real = anim_mag_linear * np.cos(anim_phase_rad)
        anim_imag = anim_mag_linear * np.sin(anim_phase_rad)
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        
        # Bode plots on left
        ax1 = plt.subplot2grid((2, 2), (0, 0))  # Magnitude
        ax2 = plt.subplot2grid((2, 2), (1, 0))  # Phase
        ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)  # Nyquist
        
        # Setup Bode plots
        ax1.semilogx(anim_w, anim_mag, 'b', alpha=0.3, linewidth=1)
        ax1.set_title('Bode - Magnitude')
        ax1.set_ylabel('Magnitude [dB]')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([anim_w[0], anim_w[-1]])
        
        ax2.semilogx(anim_w, anim_phase, 'r', alpha=0.3, linewidth=1)
        ax2.set_title('Bode - Phase')
        ax2.set_ylabel('Phase [deg]')
        ax2.set_xlabel('Frequency [rad/s]')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([anim_w[0], anim_w[-1]])
        
        # Setup Nyquist plot
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', alpha=0.3)
        ax3.axvline(x=0, color='k', alpha=0.3)
        ax3.set_xlabel('Real')
        ax3.set_ylabel('Imaginary')
        ax3.set_title('Nyquist Construction')
        
        # Set limits
        all_real = list(anim_real) + [-1, 1]
        all_imag = list(anim_imag) + [-1, 1]
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
        nyquist_trajectory, = ax3.plot([], [], 'g-', alpha=0.7, linewidth=2)
        
        text_box = ax3.text(0.02, 0.98, '', transform=ax3.transAxes, fontsize=9,
                           verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Pre-calculate frame indices
        indices = np.linspace(0, len(anim_w)-1, min(num_frames, len(anim_w)), dtype=int)
        
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
            
            freq = anim_w[idx]
            mag_db = anim_mag[idx]
            mag_lin = anim_mag_linear[idx]
            phase_deg = anim_phase[idx]
            phase_rad = anim_phase_rad[idx]
            
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
            nyquist_trajectory.set_data(anim_real[:idx+1], anim_imag[:idx+1])
            
            # Update text
            text_box.set_text(f'f = {freq:.2f} rad/s\n|G| = {mag_lin:.2f}\n∠ = {phase_deg:.0f}°')
            
            return mag_point, phase_point, circle, phase_line, nyquist_point, nyquist_trajectory, text_box
        
        # Create animation
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=len(indices), interval=50, blit=True)
        
        # Save to BytesIO
        gif_buffer = BytesIO()
        
        try:
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=15)
            anim.save(gif_buffer, writer=writer)
            gif_buffer.seek(0)
            plt.close(fig)
            return gif_buffer
        except Exception as e:
            plt.close(fig)
            st.error(f"Animation error: {str(e)}")
            return None

# Main App
st.title("📈 Bode & Nyquist Visualizer")

# Input section at the top
st.markdown("### Enter Transfer Function")
tf_input = st.text_input(
    "Transfer Function (s-domain):",
    value="1/(s+1)",
    help="Examples: 20*(s^2+1), s*(s+100), 1/(s+1), 1/(s^2+0.5*s+1)"
)

# Test the parser with some examples
if st.checkbox("Debug parser"):
    test_cases = [
        "20*(s^2+1)",
        "s*(s+100)",
        "1/(s+1)",
        "s^2+3*s+2",
        "s*(s+1)*(s+2)",
        "5*s/(s^2+2*s+1)"
    ]
    
    for test in test_cases:
        try:
            num, den = parse_transfer_function(test)
            st.write(f"{test}: num={num}, den={den}")
        except Exception as e:
            st.write(f"{test}: ERROR - {str(e)}")

# Parse button
parse_clicked = st.button("Generate Plots", type="primary")

# Process input
if parse_clicked or ('tf_input' in st.session_state and st.session_state.tf_input == tf_input):
    try:
        num_coeffs, den_coeffs = parse_transfer_function(tf_input)
        
        # Store in session state
        st.session_state.tf_input = tf_input
        st.session_state.num_coeffs = num_coeffs
        st.session_state.den_coeffs = den_coeffs
        
        # Show what was parsed
        st.success("✓ Transfer function parsed!")
        
        # Create polynomial strings for display
        def poly_to_str(coeffs):
            terms = []
            for i, c in enumerate(coeffs):
                power = len(coeffs) - i - 1
                if abs(c) > 1e-10:  # Not zero
                    if power == 0:
                        terms.append(f"{c:.2f}")
                    elif power == 1:
                        terms.append(f"{c:.2f}s")
                    else:
                        terms.append(f"{c:.2f}s^{power}")
            return " + ".join(terms)
        
        num_str = poly_to_str(num_coeffs)
        den_str = poly_to_str(den_coeffs)
        
        st.write(f"**Numerator:** {num_str}")
        st.write(f"**Denominator:** {den_str}")
        
        # Create visualizer
        visualizer = FastNyquistVisualizer(num=num_coeffs, den=den_coeffs)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Bode Plot", "Nyquist Diagram", "Nyquist Construction"])
        
        with tab1:
            fig_bode = visualizer.plot_bode()
            st.pyplot(fig_bode)
            plt.close(fig_bode)
        
        with tab2:
            fig_nyquist = visualizer.plot_complete_nyquist()
            st.pyplot(fig_nyquist)
            plt.close(fig_nyquist)
            
            with st.expander("Stability Info"):
                st.markdown("""
                **(-1, 0)** point indicates stability margin.
                
                **Rules:**
                - Clockwise encirclements of (-1, 0) → unstable
                - Counter-clockwise encirclements → conditionally stable
                - No encirclement → stable (if open-loop stable)
                """)
        
        with tab3:
            st.markdown("### Nyquist Construction Animation")
            st.markdown("This shows how the Bode magnitude and phase combine to form the Nyquist plot.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                num_frames = st.slider("Frames", 20, 80, 40)
            with col2:
                if st.button("Generate Animation", key="anim_btn"):
                    st.session_state.generate_anim = True
            
            # Generate animation if requested
            if 'generate_anim' in st.session_state and st.session_state.generate_anim:
                with st.spinner("Creating animation (2-3 seconds)..."):
                    gif_buffer = visualizer.create_fast_animation(num_frames)
                    
                    if gif_buffer:
                        # Display the GIF
                        st.markdown("**Animation:**")
                        
                        # Convert to base64 for embedding
                        gif_base64 = base64.b64encode(gif_buffer.read()).decode()
                        gif_buffer.seek(0)
                        
                        # Display with HTML
                        st.markdown(
                            f'<img src="data:image/gif;base64,{gif_base64}" alt="nyquist animation" width="800">',
                            unsafe_allow_html=True
                        )
                        
                        # Download button
                        st.download_button(
                            "Download GIF",
                            data=gif_buffer,
                            file_name="nyquist.gif",
                            mime="image/gif"
                        )
                    else:
                        st.error("Could not create animation")
                
                # Clear the flag
                st.session_state.generate_anim = False
            
            # Explanation
            st.markdown("""
            **How to read the animation:**
            1. **Blue circle** shows the magnitude (radius = |G(jω)|)
            2. **Red dashed line** shows the phase angle ∠G(jω)
            3. **Green point** is where they intersect: G(jω) = |G|∠φ
            4. **Green line** traces the complete Nyquist plot
            
            As frequency increases, the point moves around the complex plane.
            """)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Try examples like: 1/(s+1), s*(s+100), 20*(s^2+1)")

# Quick examples
st.markdown("---")
st.markdown("**Quick Examples:**")

examples = [
    ("20*(s^2+1)", "20(s² + 1)"),
    ("s*(s+100)", "s(s + 100)"),
    ("1/(s+1)", "First order"),
    ("1/(s^2+0.5*s+1)", "Second order"),
    ("(s+1)/(s^2+2*s+3)", "With zero"),
]

cols = st.columns(len(examples))
for i, (example, desc) in enumerate(examples):
    with cols[i]:
        if st.button(example, key=f"ex_{i}"):
            st.session_state.tf_input = example
            st.rerun()

st.markdown("""
**Format notes:**
- Use `s^2` or `s**2` for s²
- Use `*` for multiplication
- Parentheses work: `20*(s^2+1)`
- Division: `(s+1)/(s^2+2*s+3)`
""")
