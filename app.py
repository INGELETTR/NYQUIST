import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
import base64
from io import BytesIO
import tempfile
import os
import re
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Bode & Nyquist Visualizer",
    page_icon="📈",
    layout="wide"
)

def parse_transfer_function(tf_str):
    """
    Robust transfer function parser that handles parentheses
    """
    # Clean input
    tf_str = tf_str.strip().replace(' ', '').replace('^', '**')
    
    # Handle division
    if '/' in tf_str:
        parts = tf_str.split('/')
        if len(parts) > 2:
            raise ValueError("Multiple division signs found")
        num_str, den_str = parts[0], parts[1]
    else:
        num_str, den_str = tf_str, "1"
    
    # Remove outer parentheses if they exist
    def strip_outer_parens(s):
        s = s.strip()
        while s.startswith('(') and s.endswith(')'):
            # Check if the parentheses are balanced
            if s.count('(') == s.count(')') and s.find('(') == 0 and s.rfind(')') == len(s)-1:
                s = s[1:-1].strip()
            else:
                break
        return s
    
    num_str = strip_outer_parens(num_str)
    den_str = strip_outer_parens(den_str)
    
    # Parse using robust method
    num_coeffs = parse_expression(num_str)
    den_coeffs = parse_expression(den_str)
    
    return num_coeffs, den_coeffs

def parse_expression(expr):
    """
    Parse mathematical expression with parentheses
    """
    if expr == '' or expr == '0':
        return [0.0]
    
    # Try direct float first
    try:
        return [float(expr)]
    except:
        pass
    
    # Handle multiplication with parentheses
    expr = expand_parentheses(expr)
    
    # Now parse the expanded expression
    return parse_polynomial(expr)

def expand_parentheses(expr):
    """
    Expand expressions with parentheses like (s+1)*(s+2) or 20*(s^2+1)
    """
    # First handle multiplication with parentheses
    while '*' in expr and '(' in expr:
        # Find pattern like a*(b) or (a)*(b)
        match = re.search(r'([^()]*)\*\(([^()]+)\)|\(([^()]+)\)\*([^()]*)', expr)
        if not match:
            break
            
        if match.group(1) and match.group(2):  # a*(b)
            a, b = match.group(1), match.group(2)
            result = multiply_terms(a, b)
            expr = expr.replace(f'{a}*({b})', result)
        elif match.group(3) and match.group(4):  # (a)*(b)
            a, b = match.group(3), match.group(4)
            result = multiply_terms(a, b)
            expr = expr.replace(f'({a})*({b})', result)
    
    return expr

def multiply_terms(a, b):
    """
    Multiply two terms, can be numbers or polynomials
    """
    # If a is empty or 1
    if a == '' or a == '1':
        return b
    
    # Try to parse a as a number
    try:
        multiplier = float(a)
        # Parse b as polynomial
        b_coeffs = parse_polynomial(b)
        result_coeffs = [multiplier * c for c in b_coeffs]
        
        # Convert back to string representation
        terms = []
        for i, coeff in enumerate(result_coeffs):
            if abs(coeff) > 1e-10:
                power = len(result_coeffs) - i - 1
                if power == 0:
                    terms.append(f"{coeff:+g}")
                elif power == 1:
                    terms.append(f"{coeff:+g}*s")
                else:
                    terms.append(f"{coeff:+g}*s**{power}")
        
        result = ''.join(terms).lstrip('+')
        return result if result else '0'
    
    except:
        # a is not a number, treat as polynomial
        a_coeffs = parse_polynomial(a)
        b_coeffs = parse_polynomial(b)
        
        # Multiply polynomials
        result_coeffs = [0.0] * (len(a_coeffs) + len(b_coeffs) - 1)
        for i, coeff1 in enumerate(a_coeffs):
            for j, coeff2 in enumerate(b_coeffs):
                result_coeffs[i + j] += coeff1 * coeff2
        
        # Convert to string
        terms = []
        for i, coeff in enumerate(result_coeffs):
            if abs(coeff) > 1e-10:
                power = len(result_coeffs) - i - 1
                if power == 0:
                    terms.append(f"{coeff:+g}")
                elif power == 1:
                    terms.append(f"{coeff:+g}*s")
                else:
                    terms.append(f"{coeff:+g}*s**{power}")
        
        result = ''.join(terms).lstrip('+')
        return result if result else '0'

def parse_polynomial(expr):
    """
    Parse polynomial without parentheses
    """
    if expr == '0' or expr == '':
        return [0.0]
    
    # Try constant first
    try:
        return [float(expr)]
    except:
        pass
    
    # Handle special cases
    if expr == 's':
        return [1.0, 0.0]
    if expr == '-s':
        return [-1.0, 0.0]
    
    # Convert to standard format
    expr = expr.replace('**', '^')
    
    # Add + at beginning if no sign
    if expr[0] not in '+-':
        expr = '+' + expr
    
    # Find all terms
    terms = re.findall(r'([+-][^+-]*)', expr)
    
    # Find maximum power
    max_power = 0
    for term in terms:
        if 's^' in term:
            power = int(term.split('s^')[1])
            max_power = max(max_power, power)
        elif '*s' in term or (term.endswith('s') and not term.endswith('^s')):
            max_power = max(max_power, 1)
    
    # Initialize coefficients
    coeffs = [0.0] * (max_power + 1)
    
    # Fill coefficients
    for term in terms:
        sign = -1 if term[0] == '-' else 1
        term = term[1:]  # Remove sign
        
        if 's^' in term:
            parts = term.split('s^')
            coeff_str = parts[0]
            coeff = float(coeff_str) if coeff_str not in ['', '+', '-'] else 1.0
            power = int(parts[1])
        elif '*s' in term:
            coeff = float(term.split('*s')[0]) if term.split('*s')[0] else 1.0
            power = 1
        elif term.endswith('s'):
            coeff_str = term[:-1]
            coeff = float(coeff_str) if coeff_str else 1.0
            power = 1
        else:
            coeff = float(term)
            power = 0
        
        coeffs[max_power - power] = sign * coeff
    
    # Remove trailing zeros
    while len(coeffs) > 1 and abs(coeffs[-1]) < 1e-10:
        coeffs.pop()
    
    return coeffs

class NyquistVisualizer:
    def __init__(self, num=None, den=None, min_freq=-2, max_freq=2):
        if num is not None and den is not None:
            self.sys = signal.TransferFunction(num, den)
        else:
            raise ValueError("Must provide num and den coefficients")
        
        # Store frequency range
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        # Frequency range for plots (more points for smooth plots)
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
    
    def plot_complete_nyquist(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot Nyquist
        ax.plot(self.nyquist_real, self.nyquist_imag, 'b-', linewidth=2, label='ω: 0 → ∞')
        ax.plot(self.nyquist_real, -self.nyquist_imag, 'b--', linewidth=1, alpha=0.5, label='ω: -∞ → 0')
        
        # Mark (-1, 0)
        ax.plot(-1, 0, 'rx', markersize=12, markeredgewidth=2, label='(-1, 0)')
        ax.text(-1.1, 0.1, '(-1, 0)', fontsize=12, color='red')
        
        # Better axis scaling
        x_data = np.concatenate([self.nyquist_real, [-1, 1, 0]])
        y_data = np.concatenate([self.nyquist_imag, [-1, 1, 0]])
        
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        
        # Add margins
        margin = 0.2
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Ensure plot is square and not too narrow
        plot_range = max(x_range, y_range, 0.1) * (1 + margin)  # At least 0.1 range
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Make sure plot includes origin and (-1,0) if they're near the edge
        if abs(center_x) < plot_range/4:
            center_x = 0
        if abs(center_y) < plot_range/4:
            center_y = 0
        
        ax.set_xlim(center_x - plot_range/2, center_x + plot_range/2)
        ax.set_ylim(center_y - plot_range/2, center_y + plot_range/2)
        
        ax.set_xlabel('Real', fontsize=12)
        ax.set_ylabel('Imaginary', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title('Nyquist Diagram', fontsize=14)
        ax.legend(loc='best')
        
        plt.tight_layout()
        return fig
    
    def create_fast_animation(self, num_frames=40):
        """
        Create animation and save as GIF using temp file
        """
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            st.error("Please install matplotlib and pillow")
            return None
        
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

# Main App
st.title("📈 Bode & Nyquist Visualizer")

# Sidebar for frequency range
with st.sidebar:
    st.markdown("### Frequency Range")
    st.markdown("Set the frequency range for the Bode plot (in decades):")
    
    col1, col2 = st.columns(2)
    with col1:
        min_freq = st.number_input("Min frequency (10^x)", 
                                  value=-2.0, 
                                  min_value=-5.0, 
                                  max_value=5.0, 
                                  step=0.5,
                                  help="Minimum frequency exponent (10^min)")
    with col2:
        max_freq = st.number_input("Max frequency (10^x)", 
                                  value=2.0, 
                                  min_value=-5.0, 
                                  max_value=5.0, 
                                  step=0.5,
                                  help="Maximum frequency exponent (10^max)")
    
    if min_freq >= max_freq:
        st.error("Minimum frequency must be less than maximum frequency")
        min_freq, max_freq = -2.0, 2.0
    
    st.markdown("---")
    st.markdown("### Examples")
    
    examples = [
        ("1/(s+1)", "First order"),
        ("20*(s^2+1)", "Gain × (s² + 1)"),
        ("s*(s+100)", "s(s + 100)"),
        ("(s+1)/(s^2+2*s+3)", "With numerator"),
        ("1/(s^2+0.5*s+1)", "Second order"),
    ]
    
    for example, desc in examples:
        if st.button(f"{example}", key=f"sidebar_{example}"):
            st.session_state.tf_input = example

# Main input section
st.markdown("### Enter Transfer Function")
tf_input = st.text_input(
    "Transfer Function (s-domain):",
    value="1/(s+1)",
    help="Examples: 20*(s^2+1), s*(s+100), (s+1)/(s^2+2*s+3), 1/(s^2+0.5*s+1)"
)

parse_clicked = st.button("Generate Plots", type="primary", use_container_width=True)

# Process input
if parse_clicked or ('tf_input' in st.session_state and st.session_state.tf_input == tf_input):
    try:
        num_coeffs, den_coeffs = parse_transfer_function(tf_input)
        
        # Store in session state
        st.session_state.tf_input = tf_input
        st.session_state.num_coeffs = num_coeffs
        st.session_state.den_coeffs = den_coeffs
        
        # Show what was parsed
        st.success("✓ Transfer function parsed successfully!")
        
        # Display polynomial in readable format
        def format_poly(coeffs):
            terms = []
            n = len(coeffs)
            for i, coeff in enumerate(coeffs):
                if abs(coeff) > 1e-10:
                    power = n - i - 1
                    if power == 0:
                        terms.append(f"{coeff:.4g}")
                    elif power == 1:
                        terms.append(f"{coeff:.4g}s")
                    else:
                        terms.append(f"{coeff:.4g}s^{power}")
            if not terms:
                return "0"
            return " + ".join(terms).replace("+ -", "- ")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Numerator:** {format_poly(num_coeffs)}")
        with col2:
            st.info(f"**Denominator:** {format_poly(den_coeffs)}")
        
        # Create visualizer with user-defined frequency range
        visualizer = NyquistVisualizer(num=num_coeffs, den=den_coeffs, 
                                      min_freq=min_freq, max_freq=max_freq)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Bode Plot", "Nyquist Diagram", "Nyquist Construction"])
        
        with tab1:
            fig_bode = visualizer.plot_bode()
            st.pyplot(fig_bode)
            plt.close(fig_bode)
            
            st.info(f"Frequency range: 10^{{{min_freq}}} to 10^{{{max_freq}}} rad/s")
        
        with tab2:
            fig_nyquist = visualizer.plot_complete_nyquist()
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
                """)
        
        with tab3:
            st.markdown("### Nyquist Construction Animation")
            st.markdown("Watch how the Bode magnitude and phase combine to form the Nyquist plot.")
            
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                num_frames = st.slider("Number of frames", 20, 60, 40)
            
            if st.button("🎬 Generate Animation", type="primary", use_container_width=True):
                with st.spinner("Creating animation..."):
                    gif_buffer = visualizer.create_fast_animation(num_frames)
                    
                    if gif_buffer:
                        # Display the GIF
                        st.markdown("### Construction Animation")
                        
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
                                file_name="nyquist_construction.gif",
                                mime="image/gif",
                                use_container_width=True
                            )
                    else:
                        st.error("Could not create animation. Make sure pillow is installed: pip install pillow")
            
            # Explanation
            with st.expander("How to interpret the animation"):
                st.markdown("""
                **Animation Elements:**
                
                - **Blue circle**: Radius equals the magnitude |G(jω)| from Bode plot
                - **Red dashed line**: Angle equals the phase ∠G(jω) from Bode plot  
                - **Green point**: Intersection = G(jω) in complex plane
                - **Green line**: Traces the complete Nyquist plot as ω increases
                
                **What's happening:**
                1. As frequency ω increases, the magnitude and phase change
                2. The blue circle grows/shrinks based on magnitude
                3. The red line rotates based on phase
                4. Their intersection traces the Nyquist plot
                
                This shows how frequency response (Bode) transforms to complex plane (Nyquist).
                """)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        
        # Show helpful examples
        st.info("""
        **Try these formats:**
        
        - Simple: `1/(s+1)`
        - With gain: `20*(s^2+1)`
        - With zero: `s*(s+100)`
        - With parentheses: `(s+1)/(s^2+2*s+3)`
        - Second order: `1/(s^2+0.5*s+1)`
        
        **Note:** Use `*` for multiplication, `^` or `**` for powers, parentheses for grouping.
        """)

# Quick examples at the bottom
st.markdown("---")
st.markdown("### Quick Examples")

examples = [
    ("1/(s+1)", "First order system"),
    ("20*(s^2+1)", "Gain × (s² + 1)"),
    ("s*(s+100)", "s(s + 100)"),
    ("(s+1)/(s^2+2*s+3)", "With zero in numerator"),
    ("1/(s^2+0.5*s+1)", "Second order system"),
]

cols = st.columns(len(examples))
for i, (example, desc) in enumerate(examples):
    with cols[i]:
        if st.button(example, key=f"ex_{i}", help=desc):
            st.session_state.tf_input = example
            st.rerun()

# Requirements info
with st.expander("Installation requirements"):
    st.code("pip install streamlit numpy matplotlib scipy pillow")
