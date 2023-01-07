import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
import tempfile
import time
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

st.title("smith is an good boy")
st.markdown(
    """
    <style>
    [data-testid="stSlider"][aria-expanded="true"]>div:first-child{
        width:350px
        margin-left=-350px
    }

    [data-testid="stSlider"][aria-expanded="false"]>div:first-child{
        width:350px
        margin-left=-350px
    }
    </style>
    """,
    unsafe_allow_html=True

)
st.sidebar.title("Slider menu")
st.sidebar.subheader("parameters")


@st.cache()
def img_reshape(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[0]
    # image.shape[1] would return the width of the image in pixels,
    # and image.shape[2] would return the number of channels in the image
    # (typically 3 for RGB images and 1 for grayscale images).
    if width and height is None:
        return image

    if width is None:
        r = width / float(w)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resize = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    return resize


app_mode = st.sidebar.selectbox("choose the mode",
                                ["About app", "Run on image", "Run on video"]
                                )

# if unsafe_allow_html is set to True, the function will return the text as-is,
# including any HTML tags that are included.
# If unsafe_allow_html is set to False (the default),
if app_mode == "About app":
    st.markdown("In this we are using **mediapipe** for identification")

    st.markdown(
        """
        <style>
        [data-testid="stSlider"][aria-expanded="true"]>div:first-child{
            width:350px
            margin-left=-350px
        }

        [data-testid="stSlider"][aria-expanded="false"]>div:first-child{
            width:350px
            margin-left=-350px
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.video("https://www.youtube.com/watch?v=uYtqlXHGrvw")

    st.markdown("""

        # About This \n

        MediaPipe is an open-source framework for building cross-platform multimodal applied machine learning pipelines.\n

        Also used in: \n

        -It was developed by Google and is designed to make it easy to build and deploy machine learning models
        -that can process and analyze a wide range of input types, 
        including\n
        - [audio](https://www.youtube.com/channel/UCwj3IQlO3ny-5mEo6Z0hDqQ) \n
        -video, and \n
        -sensor data.\n

        MediaPipe provides a set of pre-built, reusable components for tasks such as""")




elif app_mode == "Run on image":

    drawing_spec = mp_drawing.DrawingSpec(thickness=int(0.2), circle_radius=int(1))
    st.sidebar.markdown("----")
    # this helps to make our inter fave look same
    st.markdown(
        """
        <style>
        [data-testid="stSlider"][aria-expanded="true"]>div:first-child{
            width:350px
            margin-left=-350px
        }

        [data-testid="stSlider"][aria-expanded="false"]>div:first-child{
            width:350px
            margin-left=-350px
        }
        </style>
        """,unsafe_allow_html=True)

    st.markdown("**Face Detected**")
    kpi1_text = st.markdown("0")
    max_face = st.sidebar.number_input("Number of faces", value=2, min_value=1, )
    st.sidebar.markdown("---")
    confidence = st.sidebar.slider("Confidence Level", min_value=0.0, value=0.5, max_value=1.0)
    st.sidebar.markdown("---")

    img_uploader = st.sidebar.file_uploader("Upload the image", type=["jpg", "jpeg", "png"])

    if img_uploader is not None:
        image = np.array(Image.open(img_uploader))
        st.sidebar.text("Uploaded Image")
    else:
        # auto upload the image
        demo = "demo.jpg"
        image = np.array(Image.open(demo))
        st.sidebar.text("Orginal Image")

    st.sidebar.image(image)
    face_count = 0

    ##dashboad
    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=max_face,
    min_detection_confidence=confidence) as face_mesh:
        result = face_mesh.process(image)
        output_img = image.copy()

        for land_mask_draw in result.multi_face_landmarks:
            face_count += 1

            mp_drawing.draw_landmarks(
            image=output_img,
            landmark_list=land_mask_draw,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec)
            kpi1_text.write(f"<h1 style= 'text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)

        st.subheader("output")
        st.image(output_img, use_column_width=True)
# the function will return the text with any HTML tags escaped,
# so that they are displayed as plain text rather than being interpreted as HTML.



elif app_mode == "Run on video":
    st.set_option("deprecation.showfileUploaderEncoding",False)

    use_webcam= st.button("use webcam")
    record=st.sidebar.button("Record Video")

    if record:
        st.checkbox("recording...",value=True)

    st.sidebar.markdown("----")
    # this helps to make our inter fave look same
    st.markdown(
        """
        <style>
        [data-testid="stSlider"][aria-expanded="true"]>div:first-child{
            width:350px
            margin-left=-350px
        }

        [data-testid="stSlider"][aria-expanded="false"]>div:first-child{
            width:350px
            margin-left=-350px
        }
        </style>
        """,unsafe_allow_html=True)

    max_face = st.sidebar.number_input("Number of faces", value=2, min_value=1, )
    st.sidebar.markdown("---")
    confidence = st.sidebar.slider("Confidence Level", min_value=0.0, value=0.5, max_value=1.0)
    st.sidebar.markdown("---")
    tracking_conf=st.sidebar.slider("Tracking Level", min_value=0.0, value=0.5, max_value=1.0)
    st.sidebar.markdown("---")

    st.markdown("## output")

    stframe= st.empty
    video_in = st.sidebar.file_uploader("upload video",type=["mp4","mov","avi"])
    temp_file=tempfile.NamedTemporaryFile(delete=False)

    if not video_in:
        if use_webcam:
            video =cv2.VideoCapture(0)
        else:
            video = cv2.VideoCapture("output.mp4")
            temp_file.name= "output.mp4"

    else:
        temp_file.write(video_in.radio())
        video=cv2.VideoCapture(temp_file.name)


    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=int(video.get(cv2.CAP_PROP_FPS))


    #RECODIN PART
    code=cv2.VideoWriter_fourcc("M","J","P","G")
    output_vdieo= cv2.VideoWriter("output_vdieo",code,fps,(width,height))

    st.sidebar.text("Input image")
    st.sidebar.video(temp_file.name)


    # ##dashboad
    # with mp_face_mesh.FaceMesh(
    # static_image_mode=True,
    # max_num_faces=max_face,
    # min_detection_confidence=confidence) as face_mesh:
    #     result = face_mesh.process(image)
    #     output_img = image.copy()
    #
    #     for land_mask_draw in result.multi_face_landmarks:
    #         face_count += 1
    #
    #         mp_drawing.draw_landmarks(
    #         image=output_img,
    #         landmark_list=land_mask_draw,
    #         connections=mp_face_mesh.FACEMESH_CONTOURS,
    #         landmark_drawing_spec=drawing_spec)
    #         kpi1_text.write(f"<h1 style= 'text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
    #
    #     st.subheader("output")
    #     st.image(output_img, use_column_width=True)