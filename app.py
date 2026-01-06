import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')
import base64
from io import BytesIO
import tempfile
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Bode & Nyquist Visualizer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for clean look
st.markdown("""
<style>
    .stTextInput>div>div>input {
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

class FastNyquistVisualizer:
    def __init__(self, num=None, den=None):
        if num is not None and den is not None:
            self.sys = signal.TransferFunction(num, den)
        else:
            raise ValueError("Must provide num and den coefficients")
        
        # Reduced frequency points for speed
        self.w = np.logspace(-2, 2, 400)
        
        # Calculate frequency response
        self.w, self.mag, self.phase = signal.bode(self.sys, self.w)
        self.mag_linear = 10**(self.mag / 20)
        
        # Pre-calculate Nyquist points (vectorized)
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
    
    def create_construction_frames(self, num_frames=30):
        """Create static frames showing construction - much faster than animation"""
        frames = []
        
        # Select evenly spaced indices
        indices = np.linspace(0, len(self.w)-1, num_frames, dtype=int)
        
        for idx in indices:
            fig = plt.figure(figsize=(14, 5))
            
            # Get current values
            freq = self.w[idx]
            mag_db = self.mag[idx]
            mag_lin = self.mag_linear[idx]
            phase_deg = self.phase[idx]
            phase_rad = np.radians(phase_deg)
            
            # Create subplots
            ax1 = plt.subplot(1, 3, 1)
            ax2 = plt.subplot(1, 3, 2)
            ax3 = plt.subplot(1, 3, 3)
            
            # Bode Magnitude
            ax1.semilogx(self.w, self.mag, 'b', alpha=0.3, linewidth=1)
            ax1.plot(freq, mag_db, 'bo', markersize=6)
            ax1.set_title(f'Bode - Magnitude\nf = {freq:.2f} rad/s')
            ax1.set_ylabel('Magnitude [dB]')
            ax1.grid(True, alpha=0.3)
            
            # Bode Phase
            ax2.semilogx(self.w, self.phase, 'r', alpha=0.3, linewidth=1)
            ax2.plot(freq, phase_deg, 'ro', markersize=6)
            ax2.set_title(f'Bode - Phase\n∠ = {phase_deg:.0f}°')
            ax2.set_ylabel('Phase [deg]')
            ax2.set_xlabel('Frequency [rad/s]')
            ax2.grid(True, alpha=0.3)
            
            # Nyquist Construction
            ax3.set_aspect('equal')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='k', alpha=0.3)
            ax3.axvline(x=0, color='k', alpha=0.3)
            
            # Circle for amplitude
            circle = plt.Circle((0, 0), mag_lin, fill=False, color='blue', alpha=0.5)
            ax3.add_patch(circle)
            
            # Phase line
            x_end = mag_lin * np.cos(phase_rad)
            y_end = mag_lin * np.sin(phase_rad)
            ax3.plot([0, x_end], [0, y_end], 'r--', alpha=0.7)
            
            # Current point
            ax3.plot(x_end, y_end, 'go', markersize=8, markeredgecolor='k')
            
            # Trajectory so far
            ax3.plot(self.nyquist_real[:idx+1], self.nyquist_imag[:idx+1], 'g-', alpha=0.7)
            
            ax3.set_title(f'Nyquist Construction\n|G| = {mag_lin:.2f}')
            ax3.set_xlabel('Real')
            ax3.set_ylabel('Imaginary')
            
            plt.tight_layout()
            frames.append(fig)
        
        return frames

def parse_transfer_function(tf_str):
    tf_str = tf_str.replace(' ', '').lower()
    
    if '/' in tf_str:
        num_str, den_str = tf_str.split('/')
    else:
        num_str = tf_str
        den_str = "1"
    
    if num_str.startswith('(') and num_str.endswith(')'):
        num_str = num_str[1:-1]
    if den_str.startswith('(') and den_str.endswith(')'):
        den_str = den_str[1:-1]
    
    num_coeffs = parse_polynomial(num_str)
    den_coeffs = parse_polynomial(den_str)
    
    return num_coeffs, den_coeffs

def parse_polynomial(poly_str):
    if 's' not in poly_str:
        return [float(poly_str)]
    
    poly_str = poly_str.replace('(', '').replace(')', '').replace('*', '')
    
    import re
    terms = re.split('([+-])', poly_str)
    
    reconstructed_terms = []
    current_sign = '+'
    for term in terms:
        if term == '+' or term == '-':
            current_sign = term
        elif term:
            if current_sign == '-':
                reconstructed_terms.append('-' + term)
            else:
                reconstructed_terms.append(term)
    
    max_power = 0
    for term in reconstructed_terms:
        if 's^' in term:
            power = int(term.split('s^')[1])
            max_power = max(max_power, power)
        elif 's' in term and '^' not in term:
            max_power = max(max_power, 1)
    
    coeffs = [0.0] * (max_power + 1)
    
    for term in reconstructed_terms:
        sign = 1
        if term.startswith('-'):
            sign = -1
            term = term[1:]
        
        if 's^' in term:
            coeff_part, power_part = term.split('s^')
            power = int(power_part)
            coeff = float(coeff_part) if coeff_part else 1.0
        elif 's' in term:
            coeff_part = term.split('s')[0]
            coeff = float(coeff_part) if coeff_part else 1.0
            power = 1
        else:
            coeff = float(term)
            power = 0
        
        coeffs[max_power - power] = sign * coeff
    
    return coeffs

# Main App
st.title("📈 Bode & Nyquist Visualizer")

# Input section - clean and centered
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("### Enter Transfer Function")
    
    # Input with examples
    tf_input = st.text_input(
        "Transfer Function (s-domain):",
        value="1/(s+1)",
        help="Examples: 1/(s+1), 1/(s^2+0.5*s+1), (s+1)/(s^2+2*s+3)"
    )
    
    parse_clicked = st.button("Generate Plots", type="primary")

# Process input
if parse_clicked or ('tf_input' in st.session_state and st.session_state.tf_input == tf_input):
    try:
        num_coeffs, den_coeffs = parse_transfer_function(tf_input)
        
        # Store in session state
        st.session_state.tf_input = tf_input
        st.session_state.num_coeffs = num_coeffs
        st.session_state.den_coeffs = den_coeffs
        
        # Create visualizer
        visualizer = FastNyquistVisualizer(num=num_coeffs, den=den_coeffs)
        
        # Display parsed coefficients
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Numerator:** {num_coeffs}")
        with col2:
            st.info(f"**Denominator:** {den_coeffs}")
        
        # Tabs for visualizations
        tab1, tab2, tab3 = st.tabs(["Bode Plot", "Nyquist Construction", "Nyquist Diagram"])
        
        with tab1:
            fig_bode = visualizer.plot_bode()
            st.pyplot(fig_bode)
            plt.close(fig_bode)
        
        with tab2:
            st.markdown("### Nyquist Construction Steps")
            
            # Simple controls
            num_frames = st.slider("Number of steps to show:", 5, 20, 10)
            
            if st.button("Show Construction", key="construct"):
                with st.spinner("Generating..."):
                    frames = visualizer.create_construction_frames(num_frames)
                    
                    # Display frames in a grid
                    cols = st.columns(2)
                    for i, fig in enumerate(frames):
                        with cols[i % 2]:
                            st.pyplot(fig)
                            plt.close(fig)
        
        with tab3:
            fig_nyquist = visualizer.plot_complete_nyquist()
            st.pyplot(fig_nyquist)
            plt.close(fig_nyquist)
            
            # Stability info
            with st.expander("Stability Information"):
                st.markdown("""
                The **(-1, 0)** point is critical for stability analysis.
                
                **Nyquist Criterion:**
                - Closed-loop stability depends on encirclements of (-1, 0)
                - Clockwise encirclements indicate instability
                - No encirclement → stable if open-loop is stable
                """)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Try examples like: 1/(s+1), 1/(s^2+0.5*s+1), (s+1)/(s^2+2*s+3)")

# Initial state or examples
elif not parse_clicked:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Examples:**")
        st.code("1/(s+1)")
        if st.button("Try this", key="ex1"):
            st.session_state.tf_input = "1/(s+1)"
            st.rerun()
    
    with col2:
        st.markdown("**Examples:**")
        st.code("1/(s^2+0.5*s+1)")
        if st.button("Try this", key="ex2"):
            st.session_state.tf_input = "1/(s^2+0.5*s+1)"
            st.rerun()
    
    with col3:
        st.markdown("**Examples:**")
        st.code("(s+1)/(s^2+2*s+3)")
        if st.button("Try this", key="ex3"):
            st.session_state.tf_input = "(s+1)/(s^2+2*s+3)"
            st.rerun()
    
    st.markdown("""
    **Enter a transfer function in s-domain format above.**
    
    Format examples:
    - `1/(s+1)` - First order system
    - `1/(s^2+0.5*s+1)` - Second order system
    - `(s+1)/(s^2+2*s+3)` - System with zeros
    """)

# Footer
st.markdown("---")
st.caption("Control Systems Visualization Tool")
