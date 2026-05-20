import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

# =========================
# DEVICE
# =========================
device = torch.device("cpu")
torch.set_grad_enabled(False)
print(f"\nUsing Device: {device}")

# =========================
# LOAD MODELS
# =========================
encoder = VGGEncoder('vgg_normalised.pth').to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load('decoder_epoch_20.pth', map_location=device))
encoder.eval()
decoder.eval()
print("\nModels Loaded Successfully!")

# =========================
# STYLE TRANSFER FUNCTION
# =========================
def style_transfer(content_image, style_image, alpha):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    content_tensor = transform(content_image).unsqueeze(0).to(device)
    style_tensor = transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feat = encoder(content_tensor, is_test=True)
        style_feat = encoder(style_tensor, is_test=True)
        t = adaptive_instance_normalization(content_feat, style_feat)
        t = alpha * t + (1 - alpha) * content_feat
        stylized_image = decoder(t)

    stylized_image = stylized_image.cpu().squeeze(0).clamp(0, 1)
    return transforms.ToPILImage()(stylized_image)

# =========================
# CUSTOM CSS (from HTML theme)
# =========================
custom_css = """
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

/* ── Root & Body ── */
:root {
    --indigo:   #6366f1;
    --purple:   #a855f7;
    --pink:     #f472b6;
    --slate-50: #f8fafc;
    --slate-200:#e2e8f0;
    --slate-400:#94a3b8;
    --slate-600:#475569;
    --slate-700:#334155;
    --slate-800:#1e293b;
    --slate-900:#0f172a;
}

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--slate-900) !important;
    color: var(--slate-200) !important;
    background-image:
        linear-gradient(rgba(15,23,42,0.92), rgba(15,23,42,0.97)),
        url('https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?q=80&w=1974&auto=format&fit=crop') !important;
    background-size: cover !important;
    background-attachment: fixed !important;
    min-height: 100vh !important;
}

/* ── Ambient Orbs ── */
.gradio-container::before {
    content: '';
    position: fixed;
    top: -10%; left: -10%;
    width: 600px; height: 600px;
    background: radial-gradient(circle, #4f46e5, transparent 70%);
    border-radius: 50%;
    filter: blur(100px);
    opacity: 0.35;
    pointer-events: none;
    z-index: 0;
    animation: floatOrb 15s infinite ease-in-out alternate;
}
.gradio-container::after {
    content: '';
    position: fixed;
    bottom: -10%; right: -10%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, #c084fc, transparent 70%);
    border-radius: 50%;
    filter: blur(100px);
    opacity: 0.3;
    pointer-events: none;
    z-index: 0;
    animation: floatOrb 15s infinite ease-in-out alternate;
    animation-delay: -5s;
}
@keyframes floatOrb {
    0%   { transform: translate(0,0)       scale(1);   }
    100% { transform: translate(50px,50px) scale(1.1); }
}

/* ── Scanlines ── */
body::after {
    content: '';
    position: fixed;
    inset: 0;
    background: linear-gradient(
        to bottom,
        rgba(255,255,255,0), rgba(255,255,255,0) 50%,
        rgba(0,0,0,0.18) 50%, rgba(0,0,0,0.18)
    );
    background-size: 100% 4px;
    pointer-events: none;
    z-index: 9998;
    opacity: 0.55;
}

/* ── App Title ── */
#app-title {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    font-size: clamp(2rem, 5vw, 3.5rem) !important;
    background: linear-gradient(to right, #818cf8, #c084fc, #f472b6) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-shadow: none !important;
    text-align: center !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: 3px !important;
}

#app-subtitle {
    text-align: center !important;
    color: var(--slate-400) !important;
    font-size: 1.1rem !important;
    margin-bottom: 2.5rem !important;
}

/* ── Panels / Cards ── */
.gr-panel, .gr-block, .gr-box,
div[class*="block"], .panel, .form {
    background: rgba(30,41,59,0.6) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 20px !important;
    box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5) !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}
div[class*="block"]:hover {
    border-color: rgba(129,140,248,0.3) !important;
}

/* ── Labels ── */
label span, .gr-form label, fieldset > div > label span {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: var(--slate-200) !important;
}

/* ── Image Upload Boxes ── */
.gr-image, .svelte-1ed2p3z, [data-testid="image"],
div[class*="image-container"] {
    background: var(--slate-900) !important;
    border: 2px dashed var(--slate-700) !important;
    border-radius: 14px !important;
    min-height: 260px !important;
    color: var(--slate-400) !important;
    transition: border-color 0.3s !important;
}
.gr-image:hover, [data-testid="image"]:hover {
    border-color: var(--indigo) !important;
}

/* ── Slider ── */
input[type="range"] {
    -webkit-appearance: none !important;
    appearance: none !important;
    width: 100% !important;
    height: 12px !important;
    background: linear-gradient(to right, #818cf8 0%, #c084fc 100%, rgba(30,41,59,0.8) 100%) !important;
    border-radius: 15px !important;
    outline: none !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    box-shadow: inset 0 2px 5px rgba(0,0,0,0.5) !important;
}
input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important;
    height: 24px !important;
    width: 24px !important;
    border-radius: 50% !important;
    background: #fff !important;
    cursor: pointer !important;
    box-shadow: 0 0 10px #c084fc, 0 0 20px #818cf8 !important;
    border: 3px solid var(--slate-800) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
input[type="range"]::-webkit-slider-thumb:hover {
    box-shadow: 0 0 18px #c084fc, 0 0 35px #818cf8 !important;
    transform: scale(1.15) !important;
}

/* ── Generate / Submit Button ── */
button.primary, button[variant="primary"],
.gr-button-primary, button.lg.primary {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    border: none !important;
    padding: 14px 48px !important;
    font-size: 1.05rem !important;
    font-family: 'Orbitron', sans-serif !important;
    border-radius: 50px !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    color: #fff !important;
    box-shadow: 0 10px 20px -10px rgba(168,85,247,0.55) !important;
    animation: glowing 2s infinite !important;
    transition: all 0.3s !important;
    cursor: pointer !important;
}
button.primary:hover, button[variant="primary"]:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%) !important;
    box-shadow: 0 20px 30px -10px rgba(168,85,247,0.65) !important;
    transform: translateY(-2px) !important;
}
@keyframes glowing {
    0%   { box-shadow: 0 0 5px #a855f7; }
    50%  { box-shadow: 0 0 20px #a855f7, 0 0 10px #6366f1; }
    100% { box-shadow: 0 0 5px #a855f7; }
}

/* ── Secondary / Clear Buttons ── */
button.secondary, button[variant="secondary"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: var(--slate-200) !important;
    border-radius: 50px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s !important;
}
button.secondary:hover {
    background: rgba(129,140,248,0.2) !important;
    border-color: var(--indigo) !important;
}

/* ── Inputs / Textboxes ── */
input[type="number"], input[type="text"], textarea, select {
    background: var(--slate-800) !important;
    border: 1px solid var(--slate-700) !important;
    color: var(--slate-50) !important;
    border-radius: 8px !important;
}
input:focus, textarea:focus {
    border-color: var(--indigo) !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.25) !important;
    outline: none !important;
}

/* ── Output Image ── */
#output-image img {
    border-radius: 16px !important;
    border: 2px solid rgba(16,185,129,0.4) !important;
    box-shadow: 0 0 30px rgba(16,185,129,0.15) !important;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--slate-200);
    text-align: center;
    padding: 14px;
    background: rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px 18px 0 0;
    margin: -12px -12px 16px -12px;
}

/* ── FAQ Accordion ── */
.faq-item {
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    margin-bottom: 10px;
    overflow: hidden;
}
.faq-question {
    background: rgba(15,23,42,0.75);
    color: var(--slate-200);
    font-weight: 600;
    padding: 14px 18px;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: background 0.2s;
}
.faq-question:hover { background: rgba(129,140,248,0.15); }
.faq-answer {
    color: var(--slate-400);
    padding: 12px 18px 16px;
    font-size: 0.95rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    display: none;
}
.faq-item.open .faq-answer  { display: block; }
.faq-item.open .faq-icon    { transform: rotate(180deg); }
.faq-icon { transition: transform 0.3s; }

/* ── Examples Section ── */
.example-flow {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 1.25rem;
    border-radius: 18px;
    background: rgba(15,23,42,0.45);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}
.example-inputs { flex: 1; display: grid; gap: 1rem; }
.example-output { flex: 1; }
.example-arrow {
    color: #a5b4fc;
    font-size: 1.75rem;
    font-weight: 700;
}
.example-role {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #cbd5e1;
    margin-bottom: 6px;
    text-transform: uppercase;
    text-align: center;
}
.example-img {
    width: 100%;
    height: 180px;
    object-fit: contain;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(15,23,42,0.8);
}

/* ── Footer ── */
#footer {
    background: rgba(15,23,42,0.95);
    border-top: 1px solid rgba(255,255,255,0.05);
    padding: 3rem 2rem 1.5rem;
    margin-top: 4rem;
    color: var(--slate-400);
    font-size: 0.9rem;
    text-align: center;
}
#footer .footer-brand {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.4rem;
    background: linear-gradient(to right, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
    display: inline-block;
}
#footer .footer-links a {
    color: var(--slate-400);
    text-decoration: none;
    margin: 0 12px;
    transition: color 0.3s;
}
#footer .footer-links a:hover { color: #818cf8; }
.social-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px; height: 36px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
    color: #f8fafc;
    margin: 0 5px;
    text-decoration: none;
    transition: background 0.3s;
    font-size: 0.9rem;
}
.social-icon:hover { background: #818cf8; }

/* ── Neural Canvas ── */
#neural-canvas {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 10px; }
::-webkit-scrollbar-track { background: var(--slate-900); }
::-webkit-scrollbar-thumb { background: var(--slate-700); border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: #818cf8; }

/* ── Fade-in ── */
@keyframes fadeInUp {
    from { opacity:0; transform:translateY(30px); }
    to   { opacity:1; transform:translateY(0);    }
}
.fade-in { animation: fadeInUp 0.8s ease-out forwards; }

/* ── Typing animation for subtitle ── */
@keyframes typing      { from { width:0 } to { width:100% } }
@keyframes blink-caret { from,to { border-color:transparent } 50% { border-color:rgba(255,255,255,0.75); } }
"""

# =========================
# EXTRA HTML (canvas + FAQ + footer)
# =========================
neural_canvas_html = """
<canvas id="neural-canvas"></canvas>
<script>
(function(){
    const canvas = document.getElementById('neural-canvas');
    if(!canvas) return;
    const ctx = canvas.getContext('2d');
    let W, H, particles=[], mouse={x:null,y:null,radius:150};
    function resize(){ W=canvas.width=window.innerWidth; H=canvas.height=window.innerHeight; }
    class P {
        constructor(){
            this.x=Math.random()*W; this.y=Math.random()*H;
            this.vx=(Math.random()-.5)*.5; this.vy=(Math.random()-.5)*.5;
            this.size=Math.random()*3+1.5; this.hue=Math.floor(Math.random()*10)*36;
            this.density=(Math.random()*30)+1;
        }
        update(){
            this.x+=this.vx; this.y+=this.vy;
            if(mouse.x!=null){
                let dx=mouse.x-this.x,dy=mouse.y-this.y,d=Math.sqrt(dx*dx+dy*dy);
                if(d<mouse.radius){
                    let f=(mouse.radius-d)/mouse.radius;
                    this.x-=(dx/d)*f*this.density;
                    this.y-=(dy/d)*f*this.density;
                }
            }
            if(this.x<0||this.x>W)this.vx*=-1;
            if(this.y<0||this.y>H)this.vy*=-1;
            this.hue=(this.hue+.2)%360;
        }
        draw(){
            ctx.fillStyle=`hsla(${this.hue},70%,60%,0.3)`;
            ctx.beginPath(); ctx.arc(this.x,this.y,this.size,0,Math.PI*2); ctx.fill();
        }
    }
    function init(){ particles=[]; let n=Math.floor(window.innerWidth/15); for(let i=0;i<n;i++) particles.push(new P()); }
    function animate(){
        ctx.clearRect(0,0,W,H);
        particles.forEach((p,i)=>{
            p.update(); p.draw();
            for(let j=i+1;j<particles.length;j++){
                const p2=particles[j], d=Math.hypot(p.x-p2.x,p.y-p2.y);
                if(d<100){
                    ctx.strokeStyle=`hsla(${p.hue},70%,60%,${.15*(1-d/100)})`;
                    ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(p2.x,p2.y); ctx.stroke();
                }
            }
        });
        requestAnimationFrame(animate);
    }
    window.addEventListener('resize',()=>{resize();init();});
    window.addEventListener('mousemove',e=>{mouse.x=e.clientX;mouse.y=e.clientY;});
    resize(); init(); animate();
})();
</script>
"""

faq_html = """
<div style="margin-top:2.5rem;" class="fade-in">
  <div style="background:rgba(30,41,59,0.6);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.1);border-radius:20px;padding:1.5rem 1.75rem;box-shadow:0 25px 50px -12px rgba(0,0,0,0.5);">
    <div style="font-family:'Orbitron',sans-serif;font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;color:#e2e8f0;text-align:center;padding:10px 0 18px;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:1.25rem;">
      FAQs
    </div>
    <div class="faq-item">
      <div class="faq-question" onclick="this.parentElement.classList.toggle('open')">
        1. Is it a pretrained model? <span class="faq-icon">▼</span>
      </div>
      <div class="faq-answer">No, we train a model ourselves using PyTorch and AdaIN architecture.</div>
    </div>
    <div class="faq-item">
      <div class="faq-question" onclick="this.parentElement.classList.toggle('open')">
        2. Is it a free platform? <span class="faq-icon">▼</span>
      </div>
      <div class="faq-answer">This demo is currently available as a free experience.</div>
    </div>
    <div class="faq-item">
      <div class="faq-question" onclick="this.parentElement.classList.toggle('open')">
        3. Which styles of painting can be used? <span class="faq-icon">▼</span>
      </div>
      <div class="faq-answer">You can use almost any painting style image — impressionist, abstract, cubist, watercolor, sketch, and more — as the style reference.</div>
    </div>
    <div class="faq-item">
      <div class="faq-question" onclick="this.parentElement.classList.toggle('open')">
        4. Tech stack of the project? <span class="faq-icon">▼</span>
      </div>
      <div class="faq-answer">Python, PyTorch, Gradio — with a custom CSS theme inspired by the original NeuralArt design.</div>
    </div>
    <div class="faq-item">
      <div class="faq-question" onclick="this.parentElement.classList.toggle('open')">
        5. Which dataset / how big data? <span class="faq-icon">▼</span>
      </div>
      <div class="faq-answer">The model is trained on large-scale image datasets covering diverse artistic styles.</div>
    </div>
  </div>
</div>
"""

footer_html = """
<div id="footer" class="fade-in">
  <div class="footer-brand">🧠 NeuralArt</div>
  <p style="margin:0.5rem 0 1rem;font-size:0.9rem;">"Every artist dips his brush in his own soul, and paints his own nature into his pictures."</p>
  <div class="footer-links" style="margin-bottom:1rem;">
    <a href="#">Home</a>
    <a href="#">About Us</a>
    <a href="#">Showcase</a>
    <a href="#">API Docs</a>
  </div>
  <div style="margin-bottom:1rem;">
    <a href="#" class="social-icon" title="Twitter">🐦</a>
    <a href="#" class="social-icon" title="GitHub">🐙</a>
    <a href="#" class="social-icon" title="Instagram">📸</a>
    <a href="#" class="social-icon" title="LinkedIn">💼</a>
  </div>
  <hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0;">
  <p style="font-size:0.78rem;color:#64748b;">© 2025 NeuralArt Inc. All rights reserved. &nbsp;|&nbsp; Privacy Policy &nbsp;|&nbsp; Terms of Service</p>
</div>
"""

# =========================
# GRADIO BLOCKS UI
# =========================
with gr.Blocks(
    css=custom_css,
    title="StyleForge AI — Neural Style Transfer",
    theme=gr.themes.Base()
) as demo:

    # Neural canvas (background animation)
    gr.HTML(neural_canvas_html)

    # ── Header ──
    gr.HTML("""
    <div style="padding:5.5rem 0 2.5rem;text-align:center;position:relative;z-index:1;">
        <h1 id="app-title">STYLEFORGE AI</h1>
        <p id="app-subtitle" style="
            border-right:2px solid rgba(255,255,255,0.75);
            white-space:nowrap;overflow:hidden;width:0;
            animation:typing 3.5s steps(40,end) forwards, blink-caret .75s step-end infinite;
            margin:0 auto;display:inline-block;">
            Redefine Reality with AI-Powered Artistry
        </p>
    </div>
    """)

    # ── Main Form ──
    with gr.Row(equal_height=True):
        with gr.Column():
            gr.HTML('<div class="section-header">Content Source</div>')
            content_input = gr.Image(
                type="pil",
                label="Content Image",
                elem_id="content-image"
            )

        with gr.Column():
            gr.HTML('<div class="section-header">Style Reference</div>')
            style_input = gr.Image(
                type="pil",
                label="Style Image",
                elem_id="style-image"
            )

    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            alpha_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.1,
                label="🎨  Style Strength (Alpha)",
                elem_id="alpha-slider"
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button(
                "✦  Generate Style  ✦",
                variant="primary",
                elem_id="generate-btn"
            )

    # ── Output ──
    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="section-header">Stylized Result</div>')
            output_image = gr.Image(
                type="pil",
                label="Output",
                elem_id="output-image"
            )

    # ── Connect function ──
    submit_btn.click(
        fn=style_transfer,
        inputs=[content_input, style_input, alpha_slider],
        outputs=output_image
    )

    # ── FAQ ──
    gr.HTML(faq_html)

    # ── Footer ──
    gr.HTML(footer_html)

# =========================
# LAUNCH
# =========================
demo.launch()