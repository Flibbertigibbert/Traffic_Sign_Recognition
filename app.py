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
    "Speed Limit 20 kmph": "This sign indicates that vehicles must not exceed a speed of 20 kilometers per hour. It is typically found in areas with high pedestrian activity, school zones, or narrow residential streets where slow driving is essential for safety.",
    "Speed Limit 30 kmph": "Maximum speed allowed is 30 km/h. Commonly used in urban areas, near playgrounds, or zones with frequent pedestrian crossings to reduce accident risks.",
    "Speed Limit 50 kmph": "Vehicles must not exceed 50 km/h. This is a standard limit in many city streets and suburban roads, balancing traffic flow and safety.",
    "Speed Limit 60 kmph": "Indicates a speed limit of 60 km/h, often used on wider urban roads or semi-residential areas where moderate speed is acceptable.",
    "Speed Limit 70 kmph": "Drivers must maintain a speed below 70 km/h. Typically seen on arterial roads or highways entering urban zones.",
    "Speed Limit 80 kmph": "Maximum speed is 80 km/h. Common on rural roads or expressways with moderate traffic and fewer intersections.",
    "End of Speed Limit 80 kmph": "This sign marks the end of the 80 km/h speed restriction. Drivers may resume the default speed limit unless otherwise posted.",
    "Speed Limit 100 kmph": "Vehicles must not exceed 100 km/h. Usually found on national highways or expressways with controlled access.",
    "Speed Limit 120 kmph": "Indicates a high-speed zone where vehicles can travel up to 120 km/h, typically on motorways or autobahns with minimal pedestrian or cyclist presence.",
    "No Passing": "This sign prohibits overtaking other vehicles. It is used in areas with limited visibility, curves, or high accident risk to prevent dangerous maneuvers.",
    "No Passing vehicle over 3.5 ton": "Heavy vehicles over 3.5 tons are not allowed to overtake. This helps maintain traffic flow and safety on narrow or hilly roads.",
    "Right-of-way at intersection": "Drivers approaching this sign have the right-of-way at the intersection. Other vehicles must yield, ensuring smoother traffic coordination.",
    "Priority road": "This road has priority over intersecting roads. Vehicles on this route do not need to yield, reducing confusion at junctions.",
    "Yield": "Drivers must slow down and give way to traffic on the main road or at the intersection. It ensures safe merging and prevents collisions.",
    "Stop": "A mandatory stop is required. Drivers must come to a complete halt and proceed only when the road is clear, often used at critical junctions.",
    "No vehicles": "All motorized vehicles are prohibited beyond this point. This sign is used in pedestrian zones, parks, or restricted areas.",
    "Veh > 3.5 tons prohibited": "Vehicles exceeding 3.5 tons are not allowed. Common in residential areas or bridges with weight restrictions.",
    "No entry": "Entry is forbidden for all vehicles. Typically placed at one-way streets or restricted zones to prevent wrong-way driving.",
    "General caution": "Warns drivers of potential hazards ahead. It may be accompanied by additional signs specifying the nature of the danger.",
    "Dangerous curve left": "A sharp curve to the left is ahead. Drivers should reduce speed and stay alert to maintain control.",
    "Dangerous curve right": "A sharp curve to the right is ahead. Slowing down and careful steering are advised.",
    "Double curve": "Two consecutive curves ahead, first to the left then to the right. Drivers should be cautious and maintain a safe speed.",
    "Bumpy road": "The road surface is uneven or damaged. Reduce speed to avoid loss of control or vehicle damage.",
    "Slippery road": "Road conditions may cause vehicles to skid, especially in wet or icy weather. Drive slowly and avoid sudden maneuvers.",
    "Road narrows on the right": "The right side of the road will narrow ahead. Drivers should adjust their position and be cautious of merging traffic.",
    "Road work": "Construction or maintenance is taking place ahead. Expect delays, detours, and workers on or near the road.",
    "Traffic signals": "Traffic lights are ahead. Be prepared to stop or yield based on the signal status.",
    "Pedestrians": "Pedestrian crossing ahead. Drivers must slow down and yield to people crossing the road.",
    "Children crossing": "Children may be crossing the road, often near schools or playgrounds. Extra caution and reduced speed are essential.",
    "Bicycles crossing": "Cyclists may be crossing or sharing the road. Watch for bike lanes and yield when necessary.",
    "Beware of ice/snow": "Road may be icy or snowy, increasing the risk of skidding. Drive slowly and maintain a safe distance.",
    "Wild animals crossing": "Animals may cross the road unexpectedly. Common in forested or rural areas; reduce speed and stay alert.",
    "End speed + passing limits": "Marks the end of both speed and overtaking restrictions. Drivers may resume normal driving behavior.",
    "Turn right ahead": "Mandatory right turn ahead. Drivers must prepare to turn and follow the indicated direction.",
    "Turn left ahead": "Mandatory left turn ahead. Drivers must prepare to turn and follow the indicated direction.",
    "Ahead only": "Only forward movement is allowed. No turns permitted at the upcoming junction.",
    "Go straight or right": "Drivers may either continue straight or turn right. Useful at intersections with limited options.",
    "Go straight or left": "Drivers may either continue straight or turn left. Indicates permitted directions at the junction.",
    "Keep right": "Stay to the right side of the road or obstacle. Often used in divided roads or construction zones.",
    "Keep left": "Stay to the left side of the road or obstacle. Ensures safe navigation around barriers or medians.",
    "Roundabout mandatory": "Drivers must enter the roundabout and follow its circular flow. Yield to traffic already in the roundabout.",
    "End of no passing": "Overtaking is now permitted. This sign marks the end of a no-passing zone.",
    "End no passing vehicle > 3.5 tons": "Heavy vehicles over 3.5 tons may now overtake. Marks the end of a restricted overtaking zone."
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
            st.write(f"{description}")



