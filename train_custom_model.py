import os
from roboflow import Roboflow
from ultralytics import YOLO

def main():
    print("🚗 Step 1: Downloading Dataset from Roboflow...")
    rf = Roboflow(api_key="8joVH7TL2k5V6uScFBEy")
    project = rf.workspace("moin").project("car_license_plates")
    version = project.version(2)
    dataset = version.download("yolov8")
    
    # The dataset object contains the path to the downloaded folder
    data_yaml_path = f"{dataset.location}/data.yaml"
    
    print(f"\n✅ Dataset downloaded successfully to: {dataset.location}")
    print(f"📄 Using YAML configuration: {data_yaml_path}")
    
    print("\n🚀 Step 2: Starting YOLO Model Training...")
    # Load a pre-trained base model. We'll use standard yolov8n
    # You can change 'yolov8n.pt' to 'weights/best.pt' if you want to fine-tune your existing one
    model = YOLO('yolov8n.pt') 

    # Start training
    # epochs=50 is a good starting point. Increase to 100 if accuracy is still low.
    results = model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs/detect",
        name="custom_license_plates"
    )

    print("\n🎉 Training Complete!")
    print("Your new, highly accurate weights are saved at: runs/detect/custom_license_plates/weights/best.pt")
    print("Copy that file and replace your current 'weights/best.pt' to use it in the Streamlit app!")

if __name__ == '__main__':
    main()
