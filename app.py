import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
import base64
from io import BytesIO
import tempfile
import os
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Bode & Nyquist Visualizer",
    page_icon="📈",
    layout="wide"
)

def parse_transfer_function(tf_str):
    """
    Simple but effective transfer function parser
    """
    # Clean input
    tf_str = tf_str.replace(' ', '').replace('^', '**')
    
    # Handle division
    if '/' in tf_str:
        num_str, den_str = tf_str.split('/')
    else:
        num_str = tf_str
        den_str = "1"
    
    # Parse using direct approach
    num_coeffs = parse_direct(num_str)
    den_coeffs = parse_direct(den_str)
    
    return num_coeffs, den_coeffs

def parse_direct(expr):
    """
    Direct parsing without overcomplication
    """
    if expr == '0':
        return [0.0]
    
    # Try constant first
    try:
        return [float(expr)]
    except:
        pass
    
    # Handle multiplication like 20*(s**2+1)
    if '*' in expr and '(' in expr:
        # Find the multiplier
        mult_end = expr.find('*(')
        if mult_end != -1:
            mult_str = expr[:mult_end]
            poly_str = expr[mult_end+2:-1]  # Remove the closing )
            
            try:
                multiplier = float(mult_str)
            except:
                if mult_str == 's':
                    multiplier = 's'
                else:
                    raise ValueError(f"Cannot parse multiplier: {mult_str}")
            
            # Parse the polynomial
            poly_coeffs = parse_polynomial(poly_str)
            
            if multiplier == 's':
                # s * (s**2 + 1) -> s**3 + s
                return [0.0] + poly_coeffs
            else:
                # 20 * (s**2 + 1) -> 20*s**2 + 20
                return [multiplier * c for c in poly_coeffs]
    
    # Handle simple polynomial
    return parse_polynomial(expr)

def parse_polynomial(expr):
    """
    Parse polynomial like s**2+1 or s+100
    """
    # Convert to simpler format
    expr = expr.replace('**', '^')
    
    # Special cases
    if expr == 's':
        return [1.0, 0.0]
    if expr == '-s':
        return [-1.0, 0.0]
    
    # Split into terms
    import re
    # Add + at the beginning if no sign
    if expr[0] not in '+-':
        expr = '+' + expr
    
    # Find terms with signs
    terms = re.findall(r'([+-]?[^+-]+)', expr)
    
    # Find max power
    max_power = 0
    for term in terms:
        if 's^' in term:
            power = int(term.split('s^')[1])
            max_power = max(max_power, power)
        elif 's' in term:
            max_power = max(max_power, 1)
    
    # Initialize coefficients
    coeffs = [0.0] * (max_power + 1)
    
    # Fill coefficients
    for term in terms:
        # Get sign
        sign = -1 if term.startswith('-') else 1
        term = term.lstrip('+-')
        
        if 's^' in term:
            # Term like 2s^2 or s^2
            if term == 's^1':
                coeff = 1.0
                power = 1
            else:
                parts = term.split('s^')
                coeff_str = parts[0]
                coeff = float(coeff_str) if coeff_str else 1.0
                power = int(parts[1])
        elif term.endswith('s'):
            # Term like 2s or s
            coeff_str = term[:-1]
            coeff = float(coeff_str) if coeff_str else 1.0
            power = 1
        else:
            # Constant
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
        self.w = np.logspace(-3, 3, 1000)  # Wider range
        
        # Calculate frequency response
        self.w, self.mag, self.phase = signal.bode(self.sys, self.w)
        self.mag_linear = 10**(self.mag / 20)
        
        # Calculate Nyquist points
        phase_rad = np.radians(self.phase)
        self.nyquist_real = self.mag_linear * np.cos(phase_rad)
        self.nyquist_imag = self.mag_linear * np.sin(phase_rad)
        
        # Store for animation (fewer points for speed)
        self.w_anim = np.logspace(-2, 2, 200)
        self.w_anim, self.mag_anim, self.phase_anim = signal.bode(self.sys, self.w_anim)
        self.mag_linear_anim = 10**(self.mag_anim / 20)
        phase_rad_anim = np.radians(self.phase_anim)
        self.nyquist_real_anim = self.mag_linear_anim * np.cos(phase_rad_anim)
        self.nyquist_imag_anim = self.mag_linear_anim * np.sin(phase_rad_anim)
    
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
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot Nyquist
        ax.plot(self.nyquist_real, self.nyquist_imag, 'b-', linewidth=2, label='ω: 0 → ∞')
        ax.plot(self.nyquist_real, -self.nyquist_imag, 'b--', linewidth=1, alpha=0.5, label='ω: -∞ → 0')
        
        # Mark (-1, 0)
        ax.plot(-1, 0, 'rx', markersize=12, markeredgewidth=2, label='(-1, 0)')
        ax.text(-1.1, 0.1, '(-1, 0)', fontsize=12, color='red')
        
        # Better axis scaling
        # Get data limits
        x_data = np.concatenate([self.nyquist_real, [-1, 1]])
        y_data = np.concatenate([self.nyquist_imag, [-1, 1]])
        
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        
        # Add margins
        margin = 0.2
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Ensure plot is square and not too narrow
        plot_range = max(x_range, y_range) * (1 + margin)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        ax.set_xlim(center_x - plot_range/2, center_x + plot_range/2)
        ax.set_ylim(center_y - plot_range/2, center_y + plot_range/2)
        
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title('Nyquist Diagram')
        ax.legend(loc='upper right')
        
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
        
        ax2.semilogx(self.w_anim, self.phase_anim, 'r', alpha=0.3, linewidth=1)
        ax2.set_title('Bode - Phase', fontsize=12)
        ax2.set_ylabel('Phase [deg]')
        ax2.set_xlabel('Frequency [rad/s]')
        ax2.grid(True, alpha=0.3)
        
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
        plot_range = max(x_range, y_range) * (1 + margin)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
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

# Input section at the top
st.markdown("### Enter Transfer Function")
tf_input = st.text_input(
    "Transfer Function (s-domain):",
    value="1/(s+1)",
    help="Examples: 20*(s^2+1), s*(s+100), 1/(s+1), 1/(s^2+0.5*s+1)"
)

# Parse button
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Numerator:** {num_coeffs}")
        with col2:
            st.info(f"**Denominator:** {den_coeffs}")
        
        # Create visualizer
        visualizer = NyquistVisualizer(num=num_coeffs, den=den_coeffs)
        
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
                num_frames = st.slider("Number of frames", 20, 60, 30)
            
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
        - Second order: `1/(s^2+0.5*s+1)`
        - With numerator: `(s+1)/(s^2+2*s+3)`
        
        **Note:** Use `*` for multiplication, `^` or `**` for powers.
        """)

# Quick examples at the bottom
st.markdown("---")
st.markdown("### Quick Examples")

examples = [
    ("1/(s+1)", "First order system"),
    ("20*(s^2+1)", "Gain × (s² + 1)"),
    ("s*(s+100)", "s(s + 100)"),
    ("1/(s^2+0.5*s+1)", "Second order system"),
    ("(s+1)/(s^2+2*s+3)", "With zero in numerator"),
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
