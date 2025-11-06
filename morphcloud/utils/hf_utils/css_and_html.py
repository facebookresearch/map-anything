# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
CSS and HTML content for the MapAnything Gradio application.
This module contains all the CSS styles and HTML content blocks
used in the Gradio interface.
"""

# CSS Styles for the Gradio interface
GRADIO_CSS = """
.custom-log * {
    font-style: italic;
    font-size: 22px !important;
    background-image: linear-gradient(120deg, #ffb366 0%, #ffa366 60%, #ff9966 100%);
    -webkit-background-clip: text;
    background-clip: text;
    font-weight: bold !important;
    color: transparent !important;
    text-align: center !important;
}

.example-log * {
    font-style: italic;
    font-size: 16px !important;
    background-image: linear-gradient(120deg, #ffb366 0%, #ffa366 60%, #ff9966 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
}

#my_radio .wrap {
    display: flex;
    flex-wrap: nowrap;
    justify-content: center;
    align-items: center;
}

#my_radio .wrap label {
    display: flex;
    width: 50%;
    justify-content: center;
    align-items: center;
    margin: 0;
    padding: 10px 0;
    box-sizing: border-box;
}

/* Align navigation buttons with dropdown bottom */
.navigation-row {
    display: flex !important;
    align-items: flex-end !important;
    gap: 8px !important;
}

.navigation-row > div:nth-child(1),
.navigation-row > div:nth-child(3) {
    align-self: flex-end !important;
}

.navigation-row > div:nth-child(2) {
    flex: 1 !important;
}

/* Make thumbnails clickable with pointer cursor */
.clickable-thumbnail img {
    cursor: pointer !important;
}

.clickable-thumbnail:hover img {
    cursor: pointer !important;
    opacity: 0.8;
    transition: opacity 0.3s ease;
}

/* Make thumbnail containers narrower horizontally */
.clickable-thumbnail {
    padding: 5px 2px !important;
    margin: 0 2px !important;
}

.clickable-thumbnail .image-container {
    margin: 0 !important;
    padding: 0 !important;
}

.scene-info {
    text-align: center !important;
    padding: 5px 2px !important;
    margin: 0 !important;
}
"""


def get_header_html(logo_base64=None):
    """
    Generate the main header HTML with logo and title.

    Args:
        logo_base64 (str, optional): Base64 encoded logo image

    Returns:
        str: HTML string for the header
    """
    logo_style = "display: none;" if not logo_base64 else ""
    logo_src = logo_base64 or ""

    return f"""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <img src="{logo_src}" alt="WAI Logo" style="height: 40px; margin-right: 15px; {logo_style}">
        <h1 style="margin: 0;"><span style="color: #ffb366;">MapAnything:</span> <span style="color: #555555;">Metric 3D Scene Reconstruction</span></h1>
    </div>
    <p>
    <a href="https://github.com/facebookresearch/map-anything">🌟 GitHub Repository</a> |
    <a href="https://map-anything.github.io/">🚀 Project Page</a>
    </p>
    """


def get_description_html():
    """
    Generate the main description and getting started HTML.

    Returns:
        str: HTML string for the description
    """
    return """
    <div style="font-size: 16px; line-height: 1.5;">
    <p>Upload a video or a set of images to create a 3D reconstruction of a scene or object. MapAnything takes these images and generates 3D point clouds directly from multi-view images.</p>
    <p>This demo demonstrates the use of image inputs only. However, MapAnything is extremely flexible and supports any combination of inputs (images, calibration, poses & depth). For trying out memory efficient inference or additional inputs like cameras & depth, please check out the code in our <a href="https://github.com/facebookresearch/map-anything">Github repo</a>.</p>

    <h3>Getting Started:</h3>
    <ol>
        <li><strong>Upload Your Data:</strong> Use the "Upload Video" or "Upload Images" buttons on the left to provide your input. Videos will be automatically split into individual frames (one frame per second).</li>
        <li><strong>Preview:</strong> Your uploaded images will appear in the gallery on the left.</li>
        <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction process.</li>
        <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file. Note the visualization of 3D points may be slow for a large number of input images.</li>
        <li>
        <strong>Adjust Reconstruction & Visualization (Optional):</strong>
        You can fine-tune the visualization using the options below the viewer
        <details style="display:inline;">
            <summary style="display:inline;">(<strong>click to expand</strong>):</summary>
            <ul>
            <li><em>Show Camera:</em> Toggle the display of estimated camera positions.</li>
            <li><em>Show Mesh:</em> Use meshes for the prediction visualization.</li>
            <li><em>Show Points from Frame:</em> Select specific frames to display in the viewer.</li>
            <li><em>Filter Black Background:</em> Remove black background pixels.</li>
            <li><em>Filter White Background:</em> Remove white background pixels.</li>
            </ul>
        </details>
        </li>
    </ol>
    <p><strong style="color: #555555;">Please note:</strong> <span style="color: #555555;">The inference time changes based on the amount of input images, for e.g., less than 1 second for up to 50 views. However, downloading model weights and visualizing 3D points may take tens of seconds. Please be patient or, for faster visualization, use a local machine to run our demo from our <a href="https://github.com/facebookresearch/map-anything">GitHub repository</a>. </span></p>
    </div>
    """


def get_acknowledgements_html():
    """
    Generate the acknowledgements section HTML.

    Returns:
        str: HTML string for the acknowledgements
    """
    return """
    <hr style="margin-top: 40px; margin-bottom: 20px;">
    <div style="text-align: center; font-size: 14px; color: #666; margin-bottom: 20px;">
        <h3>Acknowledgements</h3>
        <p>This site builds upon code from:</p>
        <ul style="list-style: none; padding: 0;">
            <li>🔗 <a href="https://github.com/microsoft/MoGe">MoGe (and MoGe2) on GitHub (and HuggingFace)</a></li>
            <li>🔗 <a href="https://github.com/facebookresearch/vggt">VGGT on GitHub</a></li>
        </ul>
        <p>We extend our gratitude to these projects for their valuable contributions to the research community.</p>
    </div>
    """


def get_gradio_theme():
    """
    Get the configured Gradio theme.

    Returns:
        gr.themes.Base: Configured Gradio theme
    """
    import gradio as gr

    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c100="#ffedd5",
            c200="#ffddb3",
            c300="rgba(242.78125, 182.89427563548466, 120.32579495614034, 1)",
            c400="#fb923c",
            c50="#fff7ed",
            c500="#f97316",
            c600="#ea580c",
            c700="#c2410c",
            c800="#9a3412",
            c900="#7c2d12",
            c950="#6c2e12",
        ),
        secondary_hue="amber",
    )


# Measure tab instructions HTML
MEASURE_INSTRUCTIONS_HTML = """
### Click on the image to measure the distance between two points.
"""
