# Object Detection with Faster R-CNN

<p align="left">
  This project implements <b>Object Detection</b> using <b>Faster R-CNN with ResNet50-FPN</b>, a state-of-the-art deep learning model for real-time object localization and classification.  
  The model is fine-tuned on a custom dataset to accurately detect multiple object classes in images, combining <b>region proposal networks (RPN)</b> with a powerful <b>ResNet-50 Feature Pyramid Network</b>.
</p>



<h2 id="-model-architecture" align="left">🏗️ Model Architecture</h2>

- **Backbone:** ResNet-50 with Feature Pyramid Network (FPN)  
- **Detector Head:** Faster R-CNN  
- **Framework:** PyTorch + Torchvision  
- **Loss Function:** Classification + Bounding Box Regression  
- **Optimization:** SGD / Adam with learning rate scheduler  


<h2 id="-dataset--training" align="left">📦 Dataset & Training</h2>

- **Dataset:** Custom dataset prepared for object detection (COCO-style format)  
- **Classes:** Multiple object categories (e.g., car, laptop, person, bicycle, etc.)  
- **Input Size:** 224×224  
- **Data Split:** 80% training / 20% validation  
- **Epochs:** 10–15  
- **Batch Size:** 4  
- **Hardware:** CPU-compatible, GPU-accelerated optional  

Training Pipeline:
```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = len(dataset.classes)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

<h3 align="left">📸 Visual Detection Results</h3>
<img width="515" height="372" alt="image" src="https://github.com/user-attachments/assets/aeddb831-4467-49d3-a3a1-40a2b9c6cdc0" />
<img width="515" height="372" alt="image" src="https://github.com/user-attachments/assets/6092a50d-9b68-4b2b-b45c-d6092dd0a6a1" />
<img width="446" height="411" alt="image" src="https://github.com/user-attachments/assets/2b156b75-8dc8-472b-8d93-c92cad3e4ff4" />



<h2 id="-future-improvements" align="left">🚧 Future Improvements</h2>

- [ ] Convert model to ONNX or TorchScript for deployment  
- [ ] Integrate real-time video detection  
- [ ] Add custom UI for object annotation  
- [ ] Experiment with MobileNet-FPN for faster inference  



<h2 id="-acknowledgements" align="left">🙏 Acknowledgements</h2>

- PyTorch & Torchvision team for open-source detection models  
- COCO Dataset for reference annotation format  
- NVIDIA and Kaggle for providing GPU resources  

<h2 align="left">💼 Libraries & Tools</h2>

Object Detection using <b>Faster R-CNN (ResNet50-FPN)</b> is powered by a robust deep learning stack — optimized for precision, scalability, and research-ready deployment.

🧠 Every library here plays a vital role — from feature extraction and region proposal to visualization and performance tracking.  
🔗 Together, they enable an end-to-end detection pipeline that fuses computer vision and deep learning excellence.

## 👤 Author

**HOSEN ARAFAT**  

**Software Engineer, China**  

**GitHub:** https://github.com/arafathosense

**Researcher: Artificial Intelligence, Machine Learning, Deep Learning, Computer Vision, Image Processing**
