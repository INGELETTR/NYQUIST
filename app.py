import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')
import base64
from io import BytesIO
from matplotlib.animation import FuncAnimation
import tempfile
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Bode & Nyquist Visualizer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .stTextInput input {
        font-family: 'Courier New', monospace;
    }
    /* Make GIF smaller for faster loading */
    .gif-container {
        max-width: 800px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

class FastNyquistVisualizer:
    def __init__(self, num=None, den=None):
        """
        Initialize with transfer function.
        Optimized for speed.
        """
        if num is not None and den is not None:
            self.sys = signal.TransferFunction(num, den)
        else:
            raise ValueError("Must provide num and den coefficients")
        
        # Optimized frequency range (fewer points for animation)
        self.w = np.logspace(-2, 2, 200)  # Reduced from 1000 to 200
        
        # Calculate frequency response
        self.w, self.mag, self.phase = signal.bode(self.sys, self.w)
        
        # Convert magnitude from dB to linear
        self.mag_linear = 10**(self.mag / 20)
        
        # Pre-calculate all Nyquist points (vectorized for speed)
        phase_rad = np.radians(self.phase)
        self.nyquist_real = self.mag_linear * np.cos(phase_rad)
        self.nyquist_imag = self.mag_linear * np.sin(phase_rad)
        
        # Pre-calculate plot limits
        self._calculate_limits()
    
    def _calculate_limits(self):
        """Pre-calculate plot limits for consistency"""
        margin = 0.1
        all_real = np.concatenate([self.nyquist_real, [-1, 1]])
        all_imag = np.concatenate([self.nyquist_imag, [-1, 1]])
        
        self.x_min, self.x_max = np.min(all_real), np.max(all_real)
        self.y_min, self.y_max = np.min(all_imag), np.max(all_imag)
        
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        
        self.x_lim = (self.x_min - margin*x_range, self.x_max + margin*x_range)
        self.y_lim = (self.y_min - margin*y_range, self.y_max + margin*y_range)
    
    def plot_bode(self):
        """Plot Bode diagrams (amplitude and phase)"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Magnitude plot
        ax1.semilogx(self.w, self.mag, 'b', linewidth=2)
        ax1.set_title('Bode Diagram - Amplitude', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Magnitude [dB]', fontsize=12)
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Frequency [rad/s]', fontsize=12)
        
        # Phase plot
        ax2.semilogx(self.w, self.phase, 'r', linewidth=2)
        ax2.set_title('Bode Diagram - Phase', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Phase [deg]', fontsize=12)
        ax2.set_xlabel('Frequency [rad/s]', fontsize=12)
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def create_nyquist_animation(self, num_frames=30, fps=10):
        """
        Create a fast animation using pre-calculated data
        Returns animation HTML and optionally saves as GIF
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(14, 10))
        
        # Create subplots: Bode plots (left) and Nyquist construction (right)
        ax1 = plt.subplot2grid((2, 2), (0, 0))  # Magnitude Bode
        ax2 = plt.subplot2grid((2, 2), (1, 0))  # Phase Bode
        ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)  # Nyquist construction
        
        # Setup Bode plots (static background)
        ax1.semilogx(self.w, self.mag, 'b', alpha=0.3, linewidth=2)
        ax1.set_title('Bode Diagram - Amplitude', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Magnitude [dB]', fontsize=10)
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        ax1.set_xlim([self.w[0], self.w[-1]])
        
        ax2.semilogx(self.w, self.phase, 'r', alpha=0.3, linewidth=2)
        ax2.set_title('Bode Diagram - Phase', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Phase [deg]', fontsize=10)
        ax2.set_xlabel('Frequency [rad/s]', fontsize=10)
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)
        ax2.set_xlim([self.w[0], self.w[-1]])
        
        # Setup Nyquist plot
        ax3.set_title('Nyquist Diagram Construction', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Real', fontsize=10)
        ax3.set_ylabel('Imaginary', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_aspect('equal', adjustable='box')
        ax3.set_xlim(self.x_lim)
        ax3.set_ylim(self.y_lim)
        
        # Create initial empty plot elements
        mag_point, = ax1.plot([], [], 'bo', markersize=8)
        phase_point, = ax2.plot([], [], 'ro', markersize=8)
        
        # Nyquist elements
        circle = plt.Circle((0, 0), 0, fill=False, color='blue', linewidth=2, alpha=0.5)
        ax3.add_patch(circle)
        
        phase_line, = ax3.plot([], [], 'r--', linewidth=2, alpha=0.7)
        nyquist_point, = ax3.plot([], [], 'go', markersize=10, markeredgecolor='k', markeredgewidth=2)
        nyquist_trajectory, = ax3.plot([], [], 'g-', linewidth=2, alpha=0.7)
        
        # Text box
        text_box = ax3.text(0.02, 0.98, '', transform=ax3.transAxes, fontsize=10,
                           verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Pre-calculate animation indices (fewer frames for speed)
        indices = np.linspace(0, len(self.w)-1, min(num_frames, len(self.w)), dtype=int)
        
        def init():
            """Initialize animation"""
            mag_point.set_data([], [])
            phase_point.set_data([], [])
            phase_line.set_data([], [])
            nyquist_point.set_data([], [])
            nyquist_trajectory.set_data([], [])
            circle.set_radius(0)
            text_box.set_text('')
            return (mag_point, phase_point, circle, phase_line, 
                    nyquist_point, nyquist_trajectory, text_box)
        
        def animate(i):
            """Update animation frame"""
            idx = indices[i]
            
            # Get current values
            freq = self.w[idx]
            mag_db = self.mag[idx]
            mag_lin = self.mag_linear[idx]
            phase_deg = self.phase[idx]
            phase_rad = np.radians(phase_deg)
            
            # Update Bode points
            mag_point.set_data([freq], [mag_db])
            phase_point.set_data([freq], [phase_deg])
            
            # Update circle
            circle.set_radius(mag_lin)
            
            # Update phase line
            x_end = 1.5 * mag_lin * np.cos(phase_rad)
            y_end = 1.5 * mag_lin * np.sin(phase_rad)
            phase_line.set_data([0, x_end], [0, y_end])
            
            # Update Nyquist point
            x_point = self.nyquist_real[idx]
            y_point = self.nyquist_imag[idx]
            nyquist_point.set_data([x_point], [y_point])
            
            # Update trajectory
            nyquist_trajectory.set_data(self.nyquist_real[:idx+1], 
                                       self.nyquist_imag[:idx+1])
            
            # Update text
            text_box.set_text(f'Frequency: {freq:.3f} rad/s\n'
                            f'Amplitude: {mag_lin:.3f}\n'
                            f'Phase: {phase_deg:.1f}°\n'
                            f'Point: ({x_point:.3f}, {y_point:.3f})')
            
            return (mag_point, phase_point, circle, phase_line, 
                    nyquist_point, nyquist_trajectory, text_box)
        
        # Create animation
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=len(indices), interval=1000/fps, 
                           blit=True)
        
        # Save to BytesIO as HTML (much faster than GIF)
        from matplotlib.animation import HTMLWriter
        import tempfile
        
        # Create HTML animation
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            html_path = tmp.name
        
        # Use HTML writer for better performance
        writer = HTMLWriter(fps=fps)
        anim.save(html_path, writer=writer)
        
        # Read HTML content
        with open(html_path, 'r') as f:
            html_content = f.read()
        
        # Clean up
        Path(html_path).unlink()
        
        plt.close(fig)
        return html_content
    
    def plot_complete_nyquist(self):
        """Plot the complete Nyquist diagram"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot Nyquist diagram
        ax.plot(self.nyquist_real, self.nyquist_imag, 'b-', linewidth=2)
        ax.plot(self.nyquist_real, -self.nyquist_imag, 'b--', linewidth=1, alpha=0.5)
        
        # Mark (-1, 0) point for stability analysis
        ax.plot(-1, 0, 'rx', markersize=12, markeredgewidth=2)
        ax.text(-1.1, 0.1, '(-1, 0)', fontsize=12)
        
        ax.set_title('Complete Nyquist Diagram', fontsize=14, fontweight='bold')
        ax.set_xlabel('Real', fontsize=12)
        ax.set_ylabel('Imaginary', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        return fig
    
    def create_static_construction_frames(self, num_frames=6):
        """Create a few static frames for quick preview (no animation)"""
        frames = []
        indices = np.linspace(0, len(self.w)-1, num_frames, dtype=int)
        
        for idx in indices:
            fig = self._create_static_frame(idx)
            frames.append(fig)
        
        return frames
    
    def _create_static_frame(self, idx):
        """Create a single static frame"""
        fig = plt.figure(figsize=(12, 8))
        
        # Get current values
        freq = self.w[idx]
        mag_db = self.mag[idx]
        mag_lin = self.mag_linear[idx]
        phase_deg = self.phase[idx]
        phase_rad = np.radians(phase_deg)
        
        # Create subplots
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
        
        # Bode plots
        ax1.semilogx(self.w, self.mag, 'b', alpha=0.3, linewidth=2)
        ax1.plot(freq, mag_db, 'bo', markersize=8)
        ax1.set_title('Bode - Magnitude')
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        
        ax2.semilogx(self.w, self.phase, 'r', alpha=0.3, linewidth=2)
        ax2.plot(freq, phase_deg, 'ro', markersize=8)
        ax2.set_title('Bode - Phase')
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # Nyquist construction
        ax3.set_aspect('equal')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.axhline(y=0, color='k', alpha=0.3)
        ax3.axvline(x=0, color='k', alpha=0.3)
        ax3.set_xlim(self.x_lim)
        ax3.set_ylim(self.y_lim)
        
        # Circle
        circle = plt.Circle((0, 0), mag_lin, fill=False, color='blue', alpha=0.5)
        ax3.add_patch(circle)
        
        # Phase line
        x_end = 1.5 * mag_lin * np.cos(phase_rad)
        y_end = 1.5 * mag_lin * np.sin(phase_rad)
        ax3.plot([0, x_end], [0, y_end], 'r--', alpha=0.7)
        
        # Point and trajectory
        x_point = mag_lin * np.cos(phase_rad)
        y_point = mag_lin * np.sin(phase_rad)
        ax3.plot(x_point, y_point, 'go', markersize=10, markeredgecolor='k')
        ax3.plot(self.nyquist_real[:idx+1], self.nyquist_imag[:idx+1], 'g-', alpha=0.7)
        
        ax3.text(0.02, 0.98, f'f={freq:.2f} rad/s\n|G|={mag_lin:.2f}\n∠={phase_deg:.0f}°',
                transform=ax3.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig

def parse_transfer_function(tf_str):
    """Parse transfer function string to coefficients"""
    # Remove spaces and convert to lowercase
    tf_str = tf_str.replace(' ', '').lower()
    
    # Check if there's a denominator
    if '/' in tf_str:
        num_str, den_str = tf_str.split('/')
    else:
        num_str = tf_str
        den_str = "1"
    
    # Remove outer parentheses if present
    if num_str.startswith('(') and num_str.endswith(')'):
        num_str = num_str[1:-1]
    if den_str.startswith('(') and den_str.endswith(')'):
        den_str = den_str[1:-1]
    
    # Parse numerator and denominator
    num_coeffs = parse_polynomial(num_str)
    den_coeffs = parse_polynomial(den_str)
    
    return num_coeffs, den_coeffs

def parse_polynomial(poly_str):
    """Parse polynomial string to list of coefficients"""
    # Handle constant case
    if 's' not in poly_str:
        return [float(poly_str)]
    
    # Remove parentheses if present
    poly_str = poly_str.replace('(', '').replace(')', '')
    poly_str = poly_str.replace('*', '')
    
    # Split by '+' and '-' (keeping the signs)
    import re
    terms = re.split('([+-])', poly_str)
    
    # Reconstruct terms with signs
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
    
    # Find highest power
    max_power = 0
    for term in reconstructed_terms:
        if 's^' in term:
            power = int(term.split('s^')[1])
            max_power = max(max_power, power)
        elif 's' in term and '^' not in term:
            max_power = max(max_power, 1)
    
    # Initialize coefficient array
    coeffs = [0.0] * (max_power + 1)
    
    # Parse each term
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

def main():
    """Main Streamlit app"""
    st.markdown('<h1 class="main-header">📈 Fast Bode & Nyquist Visualizer</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'animation_html' not in st.session_state:
        st.session_state.animation_html = None
    
    # Sidebar for input and controls
    with st.sidebar:
        st.markdown('<h2 class="sub-header">Transfer Function</h2>', unsafe_allow_html=True)
        
        # Input options
        input_option = st.radio(
            "Input method:",
            ["Manual Input", "Predefined Examples"]
        )
        
        if input_option == "Manual Input":
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Enter transfer function in one of these formats:**")
            st.markdown("- `1/(s+1)`")
            st.markdown("- `1/(s^2+0.5*s+1)`")
            st.markdown("- `(s+1)/(s^2+2*s+3)`")
            st.markdown("- `s/(s^2+1)`")
            st.markdown('</div>', unsafe_allow_html=True)
            
            tf_input = st.text_input(
                "Transfer function:",
                value="1/(s+1)",
                help="Enter transfer function in s-domain"
            )
            
            if st.button("Parse Transfer Function", type="primary"):
                st.session_state.tf_input = tf_input
                st.session_state.animation_html = None  # Clear previous animation
                st.rerun()
        
        else:  # Predefined Examples
            example = st.selectbox(
                "Choose an example:",
                [
                    "First Order System: 1/(s+1)",
                    "Second Order System: 1/(s^2+0.5*s+1)",
                    "Lead Compensator: (s+1)/(s+2)",
                    "Integrator: 1/s",
                    "Double Integrator: 1/s^2",
                    "Oscillator: 1/(s^2+1)",
                    "Complex System: (s+2)/(s^3+3*s^2+3*s+1)"
                ]
            )
            
            example_map = {
                "First Order System: 1/(s+1)": "1/(s+1)",
                "Second Order System: 1/(s^2+0.5*s+1)": "1/(s^2+0.5*s+1)",
                "Lead Compensator: (s+1)/(s+2)": "(s+1)/(s+2)",
                "Integrator: 1/s": "1/s",
                "Double Integrator: 1/s^2": "1/s^2",
                "Oscillator: 1/(s^2+1)": "1/(s^2+1)",
                "Complex System: (s+2)/(s^3+3*s^2+3*s+1)": "(s+2)/(s^3+3*s^2+3*s+1)"
            }
            
            tf_input = example_map[example]
            if st.button("Use This Example", type="primary"):
                st.session_state.tf_input = tf_input
                st.session_state.animation_html = None  # Clear previous animation
                st.rerun()
        
        # Display parsed coefficients
        if 'tf_input' in st.session_state:
            try:
                num_coeffs, den_coeffs = parse_transfer_function(st.session_state.tf_input)
                
                st.markdown('<h2 class="sub-header">Parsed Coefficients</h2>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Numerator", str(num_coeffs))
                with col2:
                    st.metric("Denominator", str(den_coeffs))
                
                # Store in session state
                st.session_state.num_coeffs = num_coeffs
                st.session_state.den_coeffs = den_coeffs
                st.session_state.parsed_successfully = True
                
            except Exception as e:
                st.error(f"Error parsing transfer function: {e}")
                st.session_state.parsed_successfully = False
        else:
            st.info("Enter a transfer function and click 'Parse' to begin")
            st.session_state.parsed_successfully = False
    
    # Main content area
    if 'parsed_successfully' in st.session_state and st.session_state.parsed_successfully:
        try:
            # Create visualizer instance
            visualizer = FastNyquistVisualizer(
                num=st.session_state.num_coeffs,
                den=st.session_state.den_coeffs
            )
            
            # Display current transfer function
            st.markdown(f"### Current Transfer Function: `{st.session_state.tf_input}`")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Bode Plot", "⚡ Quick Preview", "🔄 Interactive Animation", "🎯 Complete Nyquist"])
            
            with tab1:
                st.markdown('<h2 class="sub-header">Bode Diagram</h2>', unsafe_allow_html=True)
                fig_bode = visualizer.plot_bode()
                st.pyplot(fig_bode)
                plt.close(fig_bode)
            
            with tab2:
                st.markdown('<h2 class="sub-header">Quick Construction Preview</h2>', unsafe_allow_html=True)
                st.info("Showing key frames of the construction (fast to generate)")
                
                # Create a few static frames
                num_frames = st.slider("Number of preview frames:", 3, 12, 6)
                
                if st.button("Generate Preview", type="primary"):
                    with st.spinner("Creating preview frames..."):
                        frames = visualizer.create_static_construction_frames(num_frames)
                        
                        # Display frames in columns
                        cols = st.columns(min(3, len(frames)))
                        for i, (col, frame) in enumerate(zip(cols, frames)):
                            with col:
                                st.pyplot(frame)
                                plt.close(frame)
            
            with tab3:
                st.markdown('<h2 class="sub-header">Interactive Animation</h2>', unsafe_allow_html=True)
                
                # Animation controls
                col1, col2 = st.columns(2)
                with col1:
                    num_frames = st.slider(
                        "Animation frames:",
                        10, 60, 30,
                        help="Fewer frames = faster generation"
                    )
                with col2:
                    fps = st.slider(
                        "Frames per second:",
                        5, 30, 10,
                        help="Lower FPS = smaller file size"
                    )
                
                # Generate or load cached animation
                if st.button("Generate Animation", type="primary") or st.session_state.animation_html:
                    if not st.session_state.animation_html:
                        with st.spinner("Creating animation (this may take a few seconds)..."):
                            # Generate HTML animation
                            st.session_state.animation_html = visualizer.create_nyquist_animation(
                                num_frames=num_frames,
                                fps=fps
                            )
                    
                    # Display the animation
                    st.markdown("### Nyquist Construction Animation")
                    st.markdown('<div class="gif-container">', unsafe_allow_html=True)
                    st.components.v1.html(st.session_state.animation_html, height=600)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Instructions
                    st.info("💡 **Tip**: The animation is interactive! You can:")
                    st.markdown("""
                    - **Pause/Play**: Click the animation
                    - **Step through frames**: Use the slider
                    - **Adjust speed**: Use the speed control
                    """)
            
            with tab4:
                st.markdown('<h2 class="sub-header">Complete Nyquist Diagram</h2>', unsafe_allow_html=True)
                fig_nyquist = visualizer.plot_complete_nyquist()
                st.pyplot(fig_nyquist)
                plt.close(fig_nyquist)
                
                # Stability analysis
                st.markdown("### Stability Analysis")
                st.info("""
                The **(-1, 0)** point is marked in red on the Nyquist diagram.
                
                **Nyquist Stability Criterion:**
                - If the Nyquist plot encircles (-1, 0) clockwise, the closed-loop system is unstable.
                - The number of encirclements relates to the number of unstable poles.
                - No encirclement of (-1, 0) indicates stability if the open-loop system is stable.
                """)
        
        except Exception as e:
            st.error(f"Error creating visualizations: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the Fast Bode & Nyquist Visualizer! 🚀
        
        This optimized tool provides **instant visualizations** of control systems:
        
        **⚡ Performance Optimizations:**
        - **Vectorized calculations** for speed
        - **HTML animations** instead of slow GIFs
        - **Smart caching** to avoid regeneration
        - **Quick preview mode** for fast feedback
        
        **📊 Features:**
        1. **Bode Plots** - Frequency response analysis
        2. **Quick Preview** - Static frames showing construction
        3. **Interactive Animation** - HTML5 animation with playback controls
        4. **Complete Nyquist** - Full diagram with stability analysis
        
        ### How to Use:
        1. Enter a transfer function in the sidebar
        2. Click **'Parse Transfer Function'**
        3. Explore the different visualization tabs
        
        ⚠️ **For best performance:**
        - Start with the **Quick Preview** tab
        - Use fewer frames for complex systems
        - The animation caches automatically - no need to regenerate!
        """)
        
        # Quick examples
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Try 1/(s+1)"):
                st.session_state.tf_input = "1/(s+1)"
                st.rerun()
        with col2:
            if st.button("Try 1/(s^2+0.5*s+1)"):
                st.session_state.tf_input = "1/(s^2+0.5*s+1)"
                st.rerun()
        with col3:
            if st.button("Try 1/(s^2+1)"):
                st.session_state.tf_input = "1/(s^2+1)"
                st.rerun()

if __name__ == "__main__":
    main()
