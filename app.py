import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Load YOLOv8n model (trained, in best.pt)
@st.cache_resource
def load_model():
    return YOLO('best.pt')  # make sure 'best.pt' is in the same folder

model = load_model()


sign_descriptions = {
    "Speed Limit 20 kmph": "Maximum speed allowed is 20 kilometers per hour.",
    "Speed Limit 30 kmph": "Maximum speed allowed is 30 kilometers per hour.",
    "Speed Limit 50 kmph": "Maximum speed allowed is 50 kilometers per hour.",
    "Speed Limit 60 kmph": "Maximum speed allowed is 60 kilometers per hour.",
    "Speed Limit 70 kmph": "Maximum speed allowed is 70 kilometers per hour.",
    "Speed Limit 80 kmph": "Maximum speed allowed is 80 kilometers per hour.",
    "End of Speed Limit 80 kmph": "End of the 80 km/h speed restriction.",
    "Speed Limit 100 kmph": "Maximum speed allowed is 100 kilometers per hour.",
    "Speed Limit 120 kmph": "Maximum speed allowed is 120 kilometers per hour.",
    "No Passing": "Overtaking is prohibited for all vehicles.",
    "No Passing vehicle over 3.5 ton": "Vehicles over 3.5 tons are not allowed to overtake.",
    "Right-of-way at intersection": "Drivers have the right-of-way at the upcoming intersection.",
    "Priority road": "This road has priority over intersecting roads.",
    "Yield": "Slow down and give way to other vehicles or pedestrians.",
    "Stop": "Come to a complete stop before proceeding.",
    "No vehicles": "No motor vehicles are allowed beyond this point.",
    "Veh > 3.5 tons prohibited": "Vehicles over 3.5 tons are prohibited.",
    "No entry": "Entry is prohibited for all vehicles.",
    "General caution": "Be alert for potential hazards ahead.",
    "Dangerous curve left": "Sharp curve to the left ahead.",
    "Dangerous curve right": "Sharp curve to the right ahead.",
    "Double curve": "Two consecutive curves ahead, first to the left.",
    "Bumpy road": "Road surface is uneven or bumpy.",
    "Slippery road": "Road may be slippery when wet or icy.",
    "Road narrows on the right": "Right side of the road narrows ahead.",
    "Road work": "Construction or maintenance work ahead.",
    "Traffic signals": "Traffic lights are ahead; be prepared to stop.",
    "Pedestrians": "Pedestrian crossing ahead; slow down.",
    "Children crossing": "Watch out for children crossing the road.",
    "Bicycles crossing": "Cyclists may be crossing ahead.",
    "Beware of ice/snow": "Road may be icy or snowy; drive carefully.",
    "Wild animals crossing": "Animals may cross the road; stay alert.",
    "End speed + passing limits": "End of speed and overtaking restrictions.",
    "Turn right ahead": "Prepare to turn right ahead.",
    "Turn left ahead": "Prepare to turn left ahead.",
    "Ahead only": "Only straight-ahead movement is permitted.",
    "Go straight or right": "You may go straight or turn right.",
    "Go straight or left": "You may go straight or turn left.",
    "Keep right": "Keep to the right side of the road.",
    "Keep left": "Keep to the left side of the road.",
    "Roundabout mandatory": "You must enter the roundabout ahead.",
    "End of no passing": "End of overtaking prohibition.",
    "End no passing vehicle > 3.5 tons": "End of overtaking ban for vehicles over 3.5 tons."
}




st.title("🔍 YOLOv8 Traffic Sign Recognition (CPU - Streamlit App)")
st.write("Upload a image below  and see what traffic sign it is using our custom model.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Optional resize to 416x416 for speed (if not already)
    image_resized = image.resize((416, 416))
    
    st.image(image_resized, caption="Uploaded Image (416x416)", use_column_width=True)

    with st.spinner("Running detection..."):
        # img_np = np.array(image_resized)
        results = model(image_resized, device="cpu")
        # force CPU
        annotated_img = results[0].plot()  # get numpy array with bounding boxes

        st.image(annotated_img, caption="Detected Objects", use_column_width=True)

        # Optional: show labels and confidences
        st.subheader("📋 Detected Classes")
        names = model.names
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            st.write(f"• {names[cls_id]} ({conf:.2%})")
            class_name = names[cls_id]
            description = sign_descriptions.get(class_name, "No description available.")
            st.write(f" Description: {description}")

