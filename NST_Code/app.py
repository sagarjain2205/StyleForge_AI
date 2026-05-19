import os
import torch
from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# =========================
# FLASK APP
# =========================

app = Flask(__name__)

app.config['SECRET_KEY'] = 'supersecretkey'

app.config['UPLOAD_FOLDER'] = 'static/uploads'

app.config['ALLOWED_EXTENSIONS'] = {
    'png',
    'jpg',
    'jpeg'
}

Bootstrap(app)

os.makedirs(
    app.config['UPLOAD_FOLDER'],
    exist_ok=True
)


# =========================
# FORM
# =========================

class UploadForm(FlaskForm):

    content = FileField('Content Image')

    style = FileField('Style Image')

    content_path = HiddenField()

    style_path = HiddenField()

    alpha = FloatField(
        'Alpha',
        default=1.0
    )

    submit = SubmitField(
        'Transfer Style'
    )


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# =========================
# LOAD MODELS
# =========================

encoder = VGGEncoder(
    'vgg_normalised.pth'
).to(device)

decoder = Decoder().to(device)

decoder.load_state_dict(
    torch.load(
        r'D:\final_nst_project\experiment\experiment1\decoder_epoch_20.pth',
        map_location=device
    )
)

encoder.eval()

decoder.eval()

print("\nModel Loaded Successfully!")


# =========================
# UTILITIES
# =========================

def allowed_file(filename):

    return (
        '.' in filename and
        filename.rsplit('.', 1)[1]
        .lower() in app.config['ALLOWED_EXTENSIONS']
    )


def style_transfer(
    content_image,
    style_image,
    encoder,
    decoder,
    alpha,
    device
):

    transform = transforms.Compose([

        transforms.Resize((512, 512)),

        transforms.ToTensor()

    ])

    content_image = (
        transform(content_image)
        .unsqueeze(0)
        .to(device)
    )

    style_image = (
        transform(style_image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        content_feat = encoder(
            content_image,
            is_test=True
        )

        style_feat = encoder(
            style_image,
            is_test=True
        )

        t = adaptive_instance_normalization(
            content_feat,
            style_feat
        )

        t = (
            alpha * t
            + (1 - alpha) * content_feat
        )

        stylized_image = decoder(t)

    return stylized_image


def save_image(image, path):

    image = image.cpu().clone()

    image = image.squeeze(0)

    image = image.clamp(0, 1)

    image = transforms.ToPILImage()(image)

    image.save(path)


# =========================
# ROUTES
# =========================

@app.route('/', methods=['GET', 'POST'])
def index():

    form = UploadForm()

    result_image = None

    content_filename = None

    style_filename = None

    error = None

    if form.validate_on_submit():

        # CONTENT IMAGE

        if (
            form.content.data and
            form.content.data.filename
        ):

            if allowed_file(
                form.content.data.filename
            ):

                content_filename = secure_filename(
                    form.content.data.filename
                )

                content_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    content_filename
                )

                form.content.data.save(
                    content_path
                )

                form.content_path.data = (
                    content_filename
                )

        else:

            content_filename = (
                form.content_path.data
            )

        # STYLE IMAGE

        if (
            form.style.data and
            form.style.data.filename
        ):

            if allowed_file(
                form.style.data.filename
            ):

                style_filename = secure_filename(
                    form.style.data.filename
                )

                style_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    style_filename
                )

                form.style.data.save(
                    style_path
                )

                form.style_path.data = (
                    style_filename
                )

        else:

            style_filename = (
                form.style_path.data
            )

        # STYLE TRANSFER

        if (
            content_filename and
            style_filename
        ):

            try:

                content_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    content_filename
                )

                style_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    style_filename
                )

                content_image = (
                    Image.open(content_path)
                    .convert('RGB')
                )

                style_image = (
                    Image.open(style_path)
                    .convert('RGB')
                )

                alpha = float(
                    form.alpha.data
                )

                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    encoder,
                    decoder,
                    alpha,
                    device
                )

                result_filename = (
                    'stylized_' +
                    content_filename
                )

                result_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    result_filename
                )

                save_image(
                    stylized_image,
                    result_path
                )

                result_image = result_filename

            except Exception as e:

                error = str(e)

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# =========================
# UPLOAD IMAGES ROUTE
# =========================

@app.route('/uploads/<filename>')
def send_image(filename):

    return send_from_directory(
        'static/uploads',
        filename
    )


# =========================
# EXAMPLE IMAGES ROUTE
# =========================

@app.route('/examples/<filename>')
def send_example(filename):

    return send_from_directory(
        'examples',
        filename
    )


# =========================
# MAIN
# =========================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )